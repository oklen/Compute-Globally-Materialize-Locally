"""kv_payprobe3.py -- = kv_payprobe2 with per-layer V features kept as a list (Gemma-4 mixes layer types whose
K/V widths differ, so they cannot be stacked across layers).

kv_payprobe2.py -- X3 in one pass, any checkpoint, probes trained on GPU.

Question: is Materialize's capacity envelope a WRITE limit or a READ limit? Per item (same synthetic
items and rng sequence as kv_payprobe):
  hidden  carrier span's hidden states (mean + last token) at every 4th layer
          harv: [H; prefix; carrier]   iso: [H; carrier] at the same positions, no prefix
  V       the SERVED V rows (no RoPE, so no absolute-position channel), every 4th layer
          harv: carrier rows from the prefill with the prefix present
          neutral: an unrelated row at the carrier slot, prefix present (is the carrier special?)
          iso: carrier encoded with no prefix
  readout the model's own menu readout (lettered options; num3: gold + 9 independent decoys, and a
          coarse "above 500?" two-option question), served three ways: harv / full / iso
Probe: PCA-128 + L2 multinomial logistic regression, grouped 4-fold CV by register-name pair, on GPU.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_payprobe2 --tag qwen3 --kinds k8 --n 512 --out ./out/x3m/qwen3_k8.json
"""
from __future__ import annotations

import argparse
import json
import random
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_phantom import HDR, ids_of
from kvmemory.kv_payprobe import PAYLOADS, make_items, texts, hidden

NEUTRAL = "<step 3>\naction: idle()\nobservation: nothing new to report this cycle.\n"
LET = "ABCDEFGHIJ"


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


def vfeat(kv, layers, a=0):
    out = []
    for l in layers:
        V = kv[l][1][0, :, a:].float()                       # [n_kv, len, d]
        v = V.permute(1, 0, 2).reshape(V.shape[1], -1)       # [len, n_kv*d]
        out.append(torch.cat([v.mean(0), v[-1]]).half())
    return out                                               # list: layer widths may differ


@torch.no_grad()
def next_logits(llm, blocks, new_ids):
    cache, pos = assemble(llm, blocks)
    cl, L = pos.shape[0], new_ids.shape[1]
    nxt = int(pos.max().item()) + 1
    out = llm.model(input_ids=new_ids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(nxt, nxt + L, device=llm.device).unsqueeze(0),
                    cache_position=torch.arange(cl, cl + L, device=llm.device),
                    attention_mask=torch.ones(1, cl + L, dtype=torch.long, device=llm.device))
    lg = out.logits[0, -1].float()
    del cache
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--kinds", default="bin,k4,k8,num3")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    llm = HFBackend()
    llm.warmup()
    dev = llm.device
    head, tail = split_wrap_nothink(llm)
    H_ids = ids_of(llm, head + HDR)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    cfg = getattr(llm.model.config, "text_config", None) or llm.model.config
    nl = cfg.num_hidden_layers
    hl = list(range(4, nl + 1, 4))          # hidden_states index (0 = embeddings)
    kl = [l - 1 for l in hl]                 # decoder layer index for the KV
    let_ids = [llm.tok.encode(c, add_special_tokens=False)[0] for c in LET]
    n_ids = ids_of(llm, NEUTRAL)
    rng, rng2 = random.Random(0), random.Random(1)
    res = {"tag": args.tag, "layers_hidden": hl, "layers_kv": kl, "n": args.n, "kinds": {}}
    for kind in PAYLOADS:                    # advance the shared rng exactly as kv_payprobe does
        items = make_items(args.n, kind, rng)
        if kind not in kinds:
            continue
        F_h, F_hi, F_v, F_vn, F_vi = [], [], [], [], []
        rd = {"harv": 0, "full": 0, "iso": 0}
        gt = {"harv": 0, "full": 0, "iso": 0}
        for it in items:
            pre, car, _ = texts(it)
            p_ids, c_ids = ids_of(llm, pre), ids_of(llm, car)
            P, C = len(p_ids), len(c_ids)
            est, een = H + P, H + P + C
            allpos = list(range(H + P)) + list(range(est, een))
            hv = hidden(llm, H_ids + p_ids + c_ids, list(range(een)), {"c": (est, een)}, hl)["c"]
            hi = hidden(llm, H_ids + c_ids, list(range(H)) + list(range(est, een)), {"c": (H, H + C)}, hl)["c"]
            fk = encode_block(llm, H_ids + p_ids + c_ids, allpos, keep_a=H)          # prefix + carrier rows
            hk = [(K[:, :, P:], V[:, :, P:]) for K, V in fk]                          # carrier rows only
            nk = encode_block(llm, H_ids + p_ids + n_ids, list(range(H + P)) + list(range(est, est + len(n_ids))), keep_a=H + P)
            ik = encode_block(llm, H_ids + c_ids, list(range(H)) + list(range(est, een)), keep_a=H)
            F_h.append(hv); F_hi.append(hi)
            F_v.append(vfeat(hk, kl)); F_vn.append(vfeat(nk, kl)); F_vi.append(vfeat(ik, kl))
            # the model's own menu readout
            if kind == "num3":
                pool = set()
                while len(pool) < 9:
                    d = str(rng2.randrange(100, 1000))
                    if d != it["v"]:
                        pool.add(d)
                opts = list(pool) + [it["v"]]
            else:
                opts = list(PAYLOADS[kind])
            rng2.shuffle(opts)
            gold = opts.index(it["v"])
            menu = " ".join(f"{LET[i]}) {o}" for i, o in enumerate(opts))
            qids = llm._ids(f"\n\nQuestion: What is the value of register {it['M']}? Options: {menu}. Answer with the letter only." + tail)
            arms = {"harv": [(H_kv, list(range(H))), (hk, list(range(est, een)))],
                    "full": [(H_kv, list(range(H))), (fk, list(range(H, een)))],
                    "iso": [(H_kv, list(range(H))), (ik, list(range(est, een)))]}
            for a, blocks in arms.items():
                lg = next_logits(llm, blocks, qids)
                rd[a] += int(int(torch.stack([lg[t] for t in let_ids[:len(opts)]]).argmax()) == gold)
            if kind == "num3":
                o2 = ["above 500", "500 or below"]
                rng2.shuffle(o2)
                g2 = o2.index("above 500" if int(it["v"]) > 500 else "500 or below")
                q2 = llm._ids(f"\n\nQuestion: Is the value of register {it['M']} above 500? Options: A) {o2[0]} B) {o2[1]}. Answer with the letter only." + tail)
                for a, blocks in arms.items():
                    lg = next_logits(llm, blocks, q2)
                    gt[a] += int(int(torch.stack([lg[let_ids[0]], lg[let_ids[1]]]).argmax()) == g2)
        groups = [f"{it['S']}-{it['M']}" for it in items]
        if kind == "num3":
            targets = {"hundreds": [int(it["v"][0]) - 1 for it in items], "tens": [int(it["v"][1]) for it in items],
                       "ones": [int(it["v"][2]) for it in items], "gt500": [int(int(it["v"]) > 500) for it in items]}
        else:
            targets = {"value": [PAYLOADS[kind].index(it["v"]) for it in items]}
        out = {"readout": {a: rd[a] / args.n for a in rd}, "probe": {}}
        if kind == "num3":
            out["readout_gt500"] = {a: gt[a] / args.n for a in gt}
        feats = {"hidden_harv": F_h, "hidden_iso": F_hi, "V_harv": F_v, "V_neutral": F_vn, "V_iso": F_vi}
        for tname, ylist in targets.items():
            y = torch.tensor(ylist)
            k = max(ylist) + 1
            out["probe"][tname] = {"chance": max(ylist.count(c) for c in set(ylist)) / args.n}
            for fname, fl in feats.items():
                nlay = len(fl[0])
                out["probe"][tname][fname] = [probe_cv(torch.stack([f[li] for f in fl]).float(), y, groups, k, dev) for li in range(nlay)]
            pt = out["probe"][tname]
            print(f"[{args.tag} {kind}/{tname}] chance {pt['chance']:.3f} | hidden harv {max(pt['hidden_harv']):.3f} iso {max(pt['hidden_iso']):.3f} "
                  f"| V harv {max(pt['V_harv']):.3f} neutral {max(pt['V_neutral']):.3f} iso {max(pt['V_iso']):.3f} | read {out['readout']}", flush=True)
        res["kinds"][kind] = out
        json.dump(res, open(args.out, "w"), indent=1)
    print("X3M_DONE", flush=True)


if __name__ == "__main__":
    main()
