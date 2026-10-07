"""kv_cite.py -- counterfactual citation verification on HotpotQA (bridge questions).

Wallat et al. 2024 (2412.18004) define citation FAITHFULNESS (the cited document causally
influenced the statement) and list two necessary conditions; they ran only the second (insert
irrelevant documents, see whether they get cited).  This module runs the first:

    if the relevant information in the cited document is ALTERED, the statement should change.

Value-swap, not removal: the passage stays in place at the same length/position; only the value
is rewritten (R3, word-boundary, residual asserted 0).  Readout is the model's own answer
sentence s scored against s' (v -> v'), teacher-forced:

    l(W)  = log p(s | prompt(W)) - log p(s' | prompt(W))
    Delta = l(W_orig) - l(W_swapped)          large => the sentence depends on that passage

Per bridge item (answer v stated in supporting passage d_A; bridge entity b = d_A's title,
mentioned in the other supporting passage d_B; 8 distractors):

    W0     original context                                   l0
    W1     d_A: v -> v'                                       DeltaA   = l0 - l1   (cited or not)
    W2o/W2 bridge test: add d_C = copy of d_A renamed b->X with v->v';
           W2o keeps d_B, W2 rewrites d_B: b -> X             DeltaB   = l2o - l2  (bridge load-bearing?)
    W3     placebo: a distractor's title entity swapped        DeltaP   = l0 - l3   (must be ~0)
    W5     no passages                                         l_none   (parametric prior)
    W6     double source: d_A's answer sentence copied into a distractor d_E;
           then swap d_A only / d_E only / both                 which one did the model use?
    PLANT  v -> v* in d_A BEFORE generation (model cannot know v*), then swap v* -> v**:
           positive control of generation + readout, immune to parametric knowledge

Negatives are borrowed from other items (same shape class, absent from the context) and the
answer negative is chosen by the no-context probe (|l_none| minimal) -- E3's R1 used an LLM
instead; flagged as a limitation in the proposal.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_cite --data ./data/hotpotqa/hotpot_dev_distractor.jsonl \\
        --limit 500 --out ./out/cite.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_e3ctrl import shape_class
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest4 import recog_q
import math

INSTR = ("Answer the question in ONE sentence using the passages. After the sentence, cite the "
         "passages you relied on as [i] (you may cite several).")
CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def wb(v):
    pb = re.escape(v)
    if v[:1].isalnum():
        pb = r"\b" + pb
    if v[-1:].isalnum():
        pb = pb + r"\b"
    return re.compile(pb, re.I)


def strip_paren(t):
    return re.sub(r"\s*\(.*?\)\s*$", "", t).strip()


def load_items(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "bridge":
            continue
        ts, ss = r["context"]["title"], r["context"]["sentences"]
        sup = list(dict.fromkeys(r["supporting_facts"]["title"]))
        if len(sup) != 2 or any(t not in ts for t in sup):
            continue
        paras = {t: " ".join(ss[ts.index(t)]) for t in ts}
        pa = wb(r["answer"])
        A = [t for t in sup if pa.search(paras[t])]
        if len(A) != 1:
            continue
        tA = A[0]
        tB = [t for t in sup if t != tA][0]
        b = strip_paren(tA)
        if len(b) < 3 or not wb(b).search(paras[tB]):
            continue
        # the sentence of d_A that states the answer (for the double-source insertion)
        sentA = next((s for s in ss[ts.index(tA)] if pa.search(s)), None)
        if sentA is None:
            continue
        out.append(dict(id=r["id"], q=r["question"], v=r["answer"], tA=tA, tB=tB, b=b,
                        titles=ts, paras=paras, sentA=sentA))
    return out


def cites_of(text):
    got = set()
    for m in CITE.finditer(text):
        for x in m.group(1).split(","):
            got.add(int(x.strip()))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/hotpotqa/hotpot_dev_distractor.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--gen_tokens", type=int, default=64)
    ap.add_argument("--no_plant", action="store_true")
    ap.add_argument("--negs", default="./out/hnegs.json",
                    help="kv_hnegs output: LLM-written same-type negatives (E3 defect C1). "
                         "Borrowed shape-class negatives are the fallback when an item is absent.")
    ap.add_argument("--done_from", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    HN = json.load(open(args.negs))["negs"] if args.negs and os.path.exists(args.negs) else {}
    print("[cite] LLM negatives for %d items" % sum(1 for r in HN.values() if r.get("ans")), flush=True)

    items = load_items(args.data)
    print("[cite] %d usable bridge items" % len(items), flush=True)
    rnd.shuffle(items)
    items = items[: args.limit]
    DONE = set()
    if args.done_from:
        import glob
        for g in args.done_from.split(":"):
            for f in glob.glob(g):
                for line in open(f, errors="ignore"):
                    m = re.match(r"^\[cite\] (\S+) \|", line)
                    if m:
                        DONE.add(m.group(1))
        print("[cite] resume: %d done" % len(DONE), flush=True)

    # negative pools: answers by shape class, bridge names
    all_items = load_items(args.data)
    by_shape = {}
    for it in all_items:
        by_shape.setdefault(shape_class(it["v"]), []).append(it["v"])
    bridge_pool = [it["b"] for it in all_items]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    dev = llm.device
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H_kv = encode_block(llm, H_ids, list(range(len(H_ids))))
    HPOS = list(range(len(H_ids)))

    def log_odds(pa, pb, ga, floor=1e-12):
        g, n = (pa, pb) if ga else (pb, pa)
        return math.log(max(g, floor)) - math.log(max(n, floor))

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        return cand.get("A", 0.0), cand.get("B", 0.0)

    def body_of(paras_list):
        return INSTR + "\n\n" + "\n\n".join("[%d] %s: %s" % (i + 1, t, p)
                                              for i, (t, p) in enumerate(paras_list))

    def lo_menu(paras_list, q, gold, neg, ga):
        """Prefill header+passages+question ONCE, then read the forced-choice gold-vs-neg log-odds.
        Bounded per-token, identical readout to kv_region and kv_ans -- not a sentence log-prob sum."""
        pre = head + body_of(paras_list) + "\n\nQuestion: " + q
        pid = llm._ids(pre)
        kv = encode_block(llm, pid[0].tolist(), list(range(pid.shape[1])))
        c, pp = assemble(llm, [(kv, list(range(pid.shape[1])))])
        a, b = (gold, neg) if ga else (neg, gold)
        pa, pb = probe(c, pp, llm._ids(recog_q(q, a, b) + tail + "Answer:"))
        del c
        return round(log_odds(pa, pb, ga), 4)

    def prompt(paras_list, q):
        body = INSTR + "\n\n" + "\n\n".join("[%d] %s: %s" % (i + 1, t, p) for i, (t, p) in enumerate(paras_list))
        return head + body + "\n\nQuestion: " + q + tail

    def prompt_none(q):
        return head + "Answer the question in ONE sentence.\n\nQuestion: " + q + tail

    @torch.no_grad()
    def gen(text, n):
        from transformers import DynamicCache
        o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), n)
        return o.strip()

    @torch.no_grad()
    def seq_lp(prefix_ids, cont_text):
        cont = llm._ids(cont_text)
        ids = torch.cat([prefix_ids, cont], 1)
        out = llm.model(input_ids=ids, use_cache=False)
        P = prefix_ids.shape[1]
        lp = torch.log_softmax(out.logits[0, P - 1:-1].float(), -1)
        return float(lp.gather(1, cont[0].unsqueeze(1)).sum()), cont.shape[1]

    def lo(world_text, s, sp):
        pid = llm._ids(world_text)
        a, na = seq_lp(pid, s)
        b, nb = seq_lp(pid, sp)
        return round(a - b, 4), na, nb

    rows, skip = [], Counter()
    for it in items:
        if it["id"] in DONE:
            continue
        q, v, tA, tB, b = it["q"], it["v"], it["tA"], it["tB"], it["b"]
        paras = it["paras"]
        order = list(it["titles"])
        rnd.shuffle(order)
        ctx = [(t, paras[t]) for t in order]
        iA, iB = order.index(tA) + 1, order.index(tB) + 1
        pv, pbr = wb(v), wb(b)
        ctx_text = " ".join(p for _, p in ctx) + " " + q

        # ---- negatives: LLM-written first (E3 C1), shape-class pool as fallback ----
        hn = HN.get(it["id"], {})
        llm_negs = [x for x in hn.get("ans", []) if x.lower() != v.lower()
                    and not wb(x).search(ctx_text) and not pv.search(x)]
        pool_negs = [x for x in by_shape.get(shape_class(v), []) if x.lower() != v.lower()
                     and not wb(x).search(ctx_text) and not pv.search(x) and 0 < len(x) <= 2 * len(v) + 10]
        rnd.shuffle(pool_negs)
        cands = list(dict.fromkeys(llm_negs + pool_negs))[: args.pool]
        Xs = [x for x in (hn.get("bridge", []) + bridge_pool) if x != b
              and not wb(x).search(ctx_text) and abs(len(x) - len(b)) < 20]
        if len(cands) < 2 or not Xs:
            skip["no_negatives"] += 1
            continue
        X = rnd.choice(Xs)
        rec_negsrc = "llm" if llm_negs else "pool"

        # ---- generate with citations on the original world ----
        w0 = prompt(ctx, q)
        ans = gen(w0, args.gen_tokens)
        cited = cites_of(ans)
        s_full = re.sub(r"\s*\[[\d,\s]+\]", "", ans).strip()   # the sentence without markers
        if not pv.search(s_full):
            skip["answer_lacks_gold"] += 1
            rec = dict(id=it["id"], q=q, v=v, ans=ans, cited=sorted(cited), iA=iA, iB=iB, status="answer_lacks_gold")
            rows.append(rec)
            print("[cite] %s | status=answer_lacks_gold ans=%r" % (it["id"], ans[:80]), flush=True)
            continue
        # choose v' by the no-context forced-choice prior (|lo| minimal), E3-style
        best = None
        for vp in cands:
            for ga in (True, False):
                a, b2 = (v, vp) if ga else (vp, v)
                c, pp = assemble(llm, [(H_kv, HPOS)])
                _pa, _pb = probe(c, pp, llm._ids(recog_q(q, a, b2) + tail + "Answer:"))
                del c
                lon = log_odds(_pa, _pb, ga)
                if best is None or abs(lon) < abs(best[0]):
                    best = (lon, vp, ga)
        l_none, vp, ga = best
        pvp = wb(vp)

        def swap_para(p, pat, rep):
            out = pat.sub(rep, p)
            assert not pat.search(out), "residual after swap"
            return out

        # ---- worlds ----
        LM = lambda plist: lo_menu(plist, q, v, vp, ga)
        pA_sw = swap_para(paras[tA], pv, vp)
        l0 = LM(ctx)
        l1 = LM([(t, pA_sw if t == tA else p) for t, p in ctx])
        na = nb = 0
        # bridge: d_C = d_A renamed b->X with v->v', appended as an extra passage
        dC_title = X + (tA[len(b):] if tA.startswith(b) else "")
        dC = swap_para(swap_para(paras[tA], pbr, X) if pbr.search(paras[tA]) else paras[tA], pv, vp)
        l2o = LM(ctx + [(dC_title, dC)])
        pB_sw = swap_para(paras[tB], pbr, X)
        l2 = LM([(t, pB_sw if t == tB else p) for t, p in ctx] + [(dC_title, dC)])
        # placebo: swap a distractor's title entity (not the answer, not the bridge) for a pool name
        dis = [t for t in order if t not in (tA, tB) and len(strip_paren(t)) >= 3
               and wb(strip_paren(t)).search(paras[t])]
        if dis:
            tD = rnd.choice(dis)
            eD = strip_paren(tD)
            Xd = rnd.choice([x for x in bridge_pool if x != eD and not wb(x).search(ctx_text)] or [X])
            l3 = LM([(t, swap_para(p, wb(eD), Xd) if t == tD else p) for t, p in ctx])
        else:
            l3 = None
        # double source: copy d_A's answer sentence into a distractor d_E (skip if no distractor)
        dEs = [t for t in order if t not in (tA, tB)]
        if dEs:
            dE = rnd.choice(dEs)
            pE_dup = paras[dE] + " " + it["sentA"]
            l6 = LM([(t, pE_dup if t == dE else p) for t, p in ctx])
            l6a = LM([(t, pE_dup if t == dE else (pA_sw if t == tA else p)) for t, p in ctx])
            l6b = LM([(t, swap_para(pE_dup, pv, vp) if t == dE else p) for t, p in ctx])
            l6c = LM([(t, swap_para(pE_dup, pv, vp) if t == dE else (pA_sw if t == tA else p)) for t, p in ctx])
        else:
            l6 = l6a = l6b = l6c = None

        rec = dict(id=it["id"], q=q, v=v, vp=vp, b=b, X=X, ans=ans, s=s_full, cited=sorted(cited),
                   iA=iA, iB=iB, citedA=int(iA in cited), citedB=int(iB in cited), n_cited=len(cited),
                   neg_src=rec_negsrc, l_none=round(l_none, 4), l0=l0, l1=l1, dA=round(l0 - l1, 4),
                   l2o=l2o, l2=l2, dB=round(l2o - l2, 4), l3=l3, dP=(round(l0 - l3, 4) if l3 is not None else None),
                   l6=l6, l6a=l6a, l6b=l6b, l6c=l6c, d6a=(round(l6 - l6a, 4) if l6 is not None else None), d6b=(round(l6 - l6b, 4) if l6 is not None else None), d6c=(round(l6 - l6c, 4) if l6 is not None else None),
                   status="ok")

        # ---- planted positive control: the model cannot know v*, so dependence must be ~1 ----
        if not args.no_plant and len(cands) >= 2:
            vs = cands[1] if cands[0] == vp else cands[0]     # v* planted into d_A (model can't know it)
            vss = next((x for x in cands if x not in (vp, vs)), None)
            if vss:
                pA_st = swap_para(paras[tA], pv, vs)
                ans_s = gen(prompt([(t, pA_st if t == tA else p) for t, p in ctx], q), args.gen_tokens)
                # dependence on d_A when its value is planted: menu vs* / vss under W0s and (d_A: vs->vss)
                pl0 = lo_menu([(t, pA_st if t == tA else p) for t, p in ctx], q, vs, vss, True)
                pl1 = lo_menu([(t, swap_para(pA_st, wb(vs), vss) if t == tA else p) for t, p in ctx], q, vs, vss, True)
                rec.update(plant_v=vs, plant_ans=ans_s, plant_cited=sorted(cites_of(ans_s)),
                           plant_l0=pl0, plant_l1=pl1, plant_d=round(pl0 - pl1, 4),
                           plant_citedA=int(iA in cites_of(ans_s)),
                           plant_answered=int(bool(wb(vs).search(ans_s))))
        rows.append(rec)
        fmt = lambda x: ("%+.2f" % x) if x is not None else "na"
        print("[cite] %s | citedA=%d citedB=%d n_cited=%d | l_none=%+.2f l0=%+.2f dA=%+.2f dB=%+.2f dP=%s | dup a/b/both=%s/%s/%s | plant_d=%s | v=%r vp=%r"
              % (it["id"], rec["citedA"], rec["citedB"], rec["n_cited"], l_none, l0, rec["dA"], rec["dB"],
                 fmt(rec["dP"]), fmt(rec["d6a"]), fmt(rec["d6b"]), fmt(rec["d6c"]),
                 fmt(rec.get("plant_d")), v[:24], vp[:24]), flush=True)
        torch.cuda.empty_cache()

    json.dump({"data": args.data, "seed": args.seed, "rows": rows, "skip": dict(skip)}, open(args.out, "w"), indent=1)
    ok = [r for r in rows if r.get("status") == "ok"]
    print("[cite] rows=%d ok=%d skip=%s" % (len(rows), len(ok), dict(skip)))
    if ok:
        import statistics as st
        m = lambda k: st.fmean([r[k] for r in ok if r.get(k) is not None])
        print("  citedA rate %.2f  citedB rate %.2f  mean n_cited %.2f" % (m("citedA"), m("citedB"), m("n_cited")))
        print("  dA %+.2f  dB %+.2f  dP %+.2f  l_none %+.2f | dup: swapA %+.2f swapE %+.2f both %+.2f | plant_d %+.2f (n=%d)"
              % (m("dA"), m("dB"), m("dP"), m("l_none"), m("d6a"), m("d6b"), m("d6c"),
                 m("plant_d") if any(r.get("plant_d") is not None for r in ok) else float("nan"),
                 sum(r.get("plant_d") is not None for r in ok)))
    print("CITE_DONE", flush=True)


if __name__ == "__main__":
    main()
