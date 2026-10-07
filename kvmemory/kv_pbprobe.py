"""kv_pbprobe.py -- X5 (paper App. J): is the sixteen-construction write-through profile
(paper Table 2, kv_parabank) a WRITE limit or a READ limit?

Unit of work = (design, construction): N items, one prefill per item with the source(s) present, then
  V probes   served V (no RoPE) of three rows, every 4th layer, mean + last token:
               car   the carrier row (slot 23)
               prev  the filler row at slot 22 (after the source(s), before the carrier: cannot see the carrier)
               dec   the decoy row at slot 0 (before any source: causal floor)
  read-time  hidden state at the last question token while serving {decoy, carrier}, every 4th layer
  readouts   kv  = serve {decoy, carrier}; txt = serve every event (comprehension control)
               gen  the paper protocol (trace prompt + 160 tokens + pick-last, exactly as kv_parabank); it is the
                    slow part, so: kv arm on every A item and the first --b_gen_n B items, txt arm on the first
                    --txt_gen_n items of each unit
               fc   forced choice on every item: first-token logits of the two state words after "Reply with the
                    state word only." (margin kept; AUC is threshold-free, the paper calibrated on the text arm)
Designs
  A bank  kv_parabank's items (one source); target = the source's state.
  B bind  two sources with OPPOSITE states; the carrier names one of them at random; target = the named
          source's state. target = c XOR s1 with c independent of everything before slot 23, so the slot-22
          row is at chance by construction, and a LINEAR probe reads the target off the carrier row only if that
          row computed the named source's state -- separate "which one is named" and "S1's state" features
          cannot express an XOR.
Probe: PCA (<=128 dims) + L2 logistic regression, 4-fold CV grouped by item, on GPU.
Units are claimed with an atomic mkdir under --qdir, so processes on several workers share one model's queue.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_pbprobe --tag qwen3 --qdir ./out/x5/q_qwen3 --out_dir ./out/x5/qwen3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES
from kvmemory.kv_parabank import BANK_MIRROR, BANK_FLAG, ev, pickw_last

STATUS2 = ["ONLINE", "OFFLINE"]
FLAG2 = ["EXCEEDED", "NORMAL"]
N_BODY, DEC, PREV, CAR = 24, 0, 22, 23
TRACE = " Locate the relevant statement, then end your answer with the final state word."
FC = " Reply with the state word only."
BANK = [("mirror", k, t) for k, t in BANK_MIRROR] + [("flag", k, t) for k, t in BANK_FLAG]
DECOY = "register {} completed a routine self-test"


def ids_of(llm, text):
    return list(llm.tok(text, add_special_tokens=False).input_ids)


def verdict_pair(r, verdict):
    while True:
        g, t = r.randrange(100, 999), r.randrange(100, 999)
        if abs(g - t) >= 30 and (g > t) == (verdict == "EXCEEDED"):
            return g, t


def skeleton(design, fam, i):
    """Item skeleton shared by the eight constructions of a family (as in kv_parabank)."""
    r = random.Random(f"pb-{design}-{fam}-{i}")
    if fam == "mirror":
        q = "\n\nQuestion: What state is register {} in? Answer ONLINE or OFFLINE."
        if design == "A":
            D, S, T = r.sample(NAMES, 3)
            gold = r.choice(STATUS2)
            m = r.randrange(8, 16)
            return dict(sp={DEC: DECOY.format(D), m: f"register {S} set to state {gold}"}, used=[D, S, T],
                        fmt=dict(a=T, b=S), target=STATUS2.index(gold), words=STATUS2, c=-1, q=q.format(T))
        D, S1, S2, T = r.sample(NAMES, 4)
        s1 = r.choice(STATUS2)
        s2 = STATUS2[1 - STATUS2.index(s1)]
        m1, m2 = r.sample(range(8, 16), 2)
        c = r.randrange(2)
        return dict(sp={DEC: DECOY.format(D), m1: f"register {S1} set to state {s1}",
                        m2: f"register {S2} set to state {s2}"}, used=[D, S1, S2, T],
                    fmt=dict(a=T, b=(S1, S2)[c]), target=STATUS2.index((s1, s2)[c]), words=STATUS2, c=c,
                    q=q.format(T))
    q = "\n\nQuestion: Was sensor {} above the alert threshold? Answer EXCEEDED or NORMAL."
    if design == "A":
        D, S = r.sample(NAMES, 2)
        v = r.choice(FLAG2)
        g, t = verdict_pair(r, v)
        m = r.randrange(8, 14)
        return dict(sp={DEC: DECOY.format(D), m: f"sensor {S} measured {g}; the alert threshold is {t}",
                        m + 1: f"alert check result for sensor {S}: threshold {v}"}, used=[D, S],
                    fmt=dict(r=S), target=FLAG2.index(v), words=FLAG2, c=-1, q=q.format(S))
    D, Sa, Sb = r.sample(NAMES, 3)
    va = r.choice(FLAG2)
    vb = FLAG2[1 - FLAG2.index(va)]
    while True:
        ma, mb = r.randrange(8, 15), r.randrange(8, 15)
        if abs(ma - mb) >= 2:
            break
    sp = {DEC: DECOY.format(D)}
    for S, v, m in ((Sa, va, ma), (Sb, vb, mb)):
        g, t = verdict_pair(r, v)
        sp[m] = f"sensor {S} measured {g}; the alert threshold is {t}"
        sp[m + 1] = f"alert check result for sensor {S}: threshold {v}"
    c = r.randrange(2)
    Sc = (Sa, Sb)[c]
    return dict(sp=sp, used=[D, Sa, Sb], fmt=dict(r=Sc), target=FLAG2.index((va, vb)[c]), words=FLAG2, c=c,
                q=q.format(Sc))


def events(sk, ctpl, rf):
    others = [x for x in NAMES if x not in sk["used"]]
    body = dict(sk["sp"])
    body[CAR] = ctpl.format(**sk["fmt"])
    for i in range(N_BODY):
        if i not in body:
            body[i] = f"register {rf.choice(others)} refreshed to {rf.randrange(100, 999)}"
    return [ev(i, body[i]) for i in range(N_BODY)]


def vrow(kv, span, base, layers):
    a, b = span[0] - base, span[1] - base
    out = []
    for l in layers:
        V = kv[l][1][0, :, a:b].float()                      # [n_kv, len, d]
        v = V.permute(1, 0, 2).reshape(V.shape[1], -1)       # [len, n_kv*d]
        out.append(torch.cat([v.mean(0), v[-1]]).half())
    return out                                               # list: layer widths may differ (Gemma-4)


@torch.no_grad()
def fc_forward(llm, blocks, qids, hl):
    cache, pos = assemble(llm, blocks)
    cl, L = pos.shape[0], qids.shape[1]
    nxt = int(pos.max().item()) + 1
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(nxt, nxt + L, device=llm.device).unsqueeze(0),
                    cache_position=torch.arange(cl, cl + L, device=llm.device),
                    attention_mask=torch.ones(1, cl + L, dtype=torch.long, device=llm.device),
                    output_hidden_states=True)
    if out.hidden_states is None:
        raise RuntimeError("model returned no hidden_states")
    lg = out.logits[0, -1].float()
    hs = [out.hidden_states[l][0, -1].half().cpu() for l in hl]
    del cache, out
    return lg, hs


def probe_cv(X, y, groups, k, dev, folds=4, dim=128, lam=1e-3, seed=0):
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    X, y = X.to(dev).float(), y.to(dev)
    accs = []
    for f in range(folds):
        test_g = set(g[f::folds])
        te = torch.tensor([gi in test_g for gi in groups], device=dev)
        Xtr, Xte, ytr, yte = X[~te], X[te], y[~te], y[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        q = min(dim, Xtr.shape[0] - 1, Xtr.shape[1])
        _, _, V = torch.pca_lowrank(Xtr, q=q, center=False)
        Ztr, Zte = Xtr @ V, Xte @ V
        W = torch.zeros(q, k, device=dev, requires_grad=True)
        b = torch.zeros(k, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Ztr @ W + b, ytr) + lam * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            accs.append(float(((Zte @ W + b).argmax(1) == yte).float().mean()))
    return round(sum(accs) / len(accs), 4)


def auc(scores, labels):
    """P(score of a label-0 item > score of a label-1 item); ties count half."""
    pos = [s for s, l in zip(scores, labels) if l == 0]
    neg = [s for s, l in zip(scores, labels) if l == 1]
    if not pos or not neg:
        return None
    w = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(w / (len(pos) * len(neg)), 4)


def run_unit(llm, ctx, design, fam, key, ctpl, n, txt_gen_n, kv_gen_n):
    H_ids, H, H_kv, tail, hl, kl, wid = ctx
    feats = {"V_car": [], "V_prev": [], "V_dec": [], "H_read": []}
    rows = []
    for i in range(n):
        sk = skeleton(design, fam, i)
        evs = events(sk, ctpl, random.Random(f"fill-{design}-{fam}-{i}-{key}"))
        eids = [ids_of(llm, t) for t in evs]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        full_kv = encode_block(llm, H_ids + [t for e in eids for t in e], list(range(cur)), keep_a=H)
        feats["V_car"].append(vrow(full_kv, spans[CAR], H, kl))
        feats["V_prev"].append(vrow(full_kv, spans[PREV], H, kl))
        feats["V_dec"].append(vrow(full_kv, spans[DEC], H, kl))
        w0, w1 = wid[fam]
        gold = sk["words"][sk["target"]]
        rec = {"i": i, "target": sk["target"], "c": sk["c"]}
        for arm, served in (("kv", [DEC, CAR]), ("txt", list(range(N_BODY)))):
            crows = [pp for j in served for pp in range(*spans[j])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            blocks = [(H_kv, list(range(H))), (sub, crows)]
            lg, hs = fc_forward(llm, blocks, llm._ids(sk["q"] + FC + tail), hl)
            margin = float(max(lg[t] for t in w0) - max(lg[t] for t in w1))
            if arm == "kv":
                feats["H_read"].append(hs)
            rec[f"{arm}_margin"] = round(margin, 3)
            rec[f"{arm}_fc"] = int((0 if margin > 0 else 1) == sk["target"])
            if i >= (txt_gen_n if arm == "txt" else kv_gen_n):   # generation readout on the first items only
                rec[f"{arm}_gen"] = rec[f"{arm}_pick"] = None
                continue
            c, p = assemble(llm, blocks)
            o, _, _ = llm._greedy_pos(c, p, llm._ids(sk["q"] + TRACE + tail), 160)
            del c, sub
            pick = pickw_last(o, sk["words"])
            rec[f"{arm}_gen"] = int(pick == gold)
            rec[f"{arm}_pick"] = pick
            if i < 3:
                rec[f"{arm}_out"] = o[-240:]
        rows.append(rec)
        del full_kv
    torch.cuda.empty_cache()
    y = torch.tensor([r["target"] for r in rows])
    groups = [r["i"] for r in rows]
    probe = {"chance": max(int((y == 0).sum()), int((y == 1).sum())) / n}
    for fname, fl in feats.items():
        probe[fname] = [probe_cv(torch.stack([f[li] for f in fl]).float(), y, groups, 2, llm.device)
                        for li in range(len(fl[0]))]
    tg = [r["txt_gen"] for r in rows if r["txt_gen"] is not None]
    kg = [r["kv_gen"] for r in rows if r["kv_gen"] is not None]
    labels = [r["target"] for r in rows]
    return {"design": design, "fam": fam, "key": key, "n": n, "rows": rows, "probe": probe,
            "kv_gen": sum(kg) / max(1, len(kg)), "kv_gen_n": len(kg),
            "txt_gen": sum(tg) / max(1, len(tg)), "txt_gen_n": len(tg),
            "kv_fc": sum(r["kv_fc"] for r in rows) / n, "txt_fc": sum(r["txt_fc"] for r in rows) / n,
            "kv_fc_auc": auc([r["kv_margin"] for r in rows], labels),
            "txt_fc_auc": auc([r["txt_margin"] for r in rows], labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--qdir", default="")
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--n_a", type=int, default=96)
    ap.add_argument("--n_b", type=int, default=256)
    ap.add_argument("--txt_gen_n", type=int, default=48)
    ap.add_argument("--b_gen_n", type=int, default=64, help="design B: kv-arm generation readout on the first items only")
    ap.add_argument("--designs", default="B,A")
    ap.add_argument("--smoke", type=int, default=0, help=">0: this many items of B/follows and A/consist, no queue")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = ids_of(llm, head + HDR)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    cfg = getattr(llm.model.config, "text_config", None) or llm.model.config
    hl = list(range(4, cfg.num_hidden_layers + 1, 4))       # hidden_states index (0 = embeddings)
    kl = [l - 1 for l in hl]                                 # decoder layer index for the KV
    wid = {}
    for fam, words in (("mirror", STATUS2), ("flag", FLAG2)):
        sets = [sorted({ids_of(llm, w)[0], ids_of(llm, " " + w)[0]}) for w in words]
        assert not set(sets[0]) & set(sets[1]), (fam, sets)
        wid[fam] = sets
    ctx = (H_ids, H, H_kv, tail, hl, kl, wid)
    tpl = {key: t for _, key, t in BANK}
    fam_of = {key: fam for fam, key, _ in BANK}
    if args.smoke:
        for d, key in (("B", "follows"), ("A", "consist")):
            t0 = time.time()
            res = run_unit(llm, ctx, d, fam_of[key], key, tpl[key], args.smoke, args.txt_gen_n, args.smoke)
            json.dump(res, open(f"./out/x5_smoke_{args.tag}_{d}_{key}.json", "w"), indent=1)
            print(f"[smoke {args.tag} {d}/{key}] kv_gen {res['kv_gen']:.2f} txt_gen {res['txt_gen']:.2f} "
                  f"kv_fc {res['kv_fc']:.2f} txt_fc {res['txt_fc']:.2f} | {(time.time() - t0) / args.smoke:.1f}s/item",
                  flush=True)
            for r in res["rows"][:2]:
                print("   ", {k: v for k, v in r.items() if not k.endswith("_out")}, "|", repr(r.get("kv_out", "")[-120:]),
                      flush=True)
        print("X5_SMOKE_OK", flush=True)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.qdir, exist_ok=True)
    for d in args.designs.split(","):
        for fam, key, ctpl in BANK:
            out = f"{args.out_dir}/{d}_{key}.json"
            if os.path.exists(out):
                continue
            try:
                os.mkdir(f"{args.qdir}/{d}_{key}")
            except FileExistsError:
                continue
            with open(f"{args.qdir}/{d}_{key}/owner", "w") as fh:
                fh.write(f"{socket.gethostname()} {os.getpid()} {os.environ.get('CUDA_VISIBLE_DEVICES')} {time.strftime('%H:%M:%S')}\n")
            t0 = time.time()
            n = args.n_a if d == "A" else args.n_b
            res = run_unit(llm, ctx, d, fam, key, ctpl, n, args.txt_gen_n, n if d == "A" else args.b_gen_n)
            res.update(tag=args.tag, layers_hidden=hl, layers_kv=kl, secs=round(time.time() - t0))
            json.dump(res, open(out + ".tmp", "w"), indent=1)
            os.replace(out + ".tmp", out)
            pb = res["probe"]
            print(f"[{args.tag} {d}/{fam}/{key}] n={n} kv_gen {res['kv_gen']:.2f} txt_gen {res['txt_gen']:.2f} "
                  f"kv_fc_auc {res['kv_fc_auc']} txt_fc_auc {res['txt_fc_auc']} | V_car {max(pb['V_car']):.2f} "
                  f"V_prev {max(pb['V_prev']):.2f} V_dec {max(pb['V_dec']):.2f} H_read {max(pb['H_read']):.2f} "
                  f"| {res['secs']}s", flush=True)
    print("X5_DONE", flush=True)


if __name__ == "__main__":
    main()
