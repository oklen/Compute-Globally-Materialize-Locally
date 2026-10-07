"""kv_e3probe.py -- does the forced-choice readout actually SEE the options?

kv_e3ctrl picks each item's negative by minimising |p(gold) - 0.5| under the no-context arm, and on
the smoke items it lands at exactly .500.  Two very different worlds produce that number:

  (a) the readout is content-sensitive and R1 genuinely found a confusable negative -- the option
      prior really has been neutralised, which is what gate G1 is supposed to certify; or
  (b) the "Answer:" forced choice is nearly blind without context, so ANY negative reads ~.5 and
      G1 certifies nothing.  Under (b) the whole repair is cosmetic.

So: probe the no-context arm over the WHOLE candidate pool plus three deliberately absurd negatives
that no reader could confuse with the gold.  Under (a) the pool spans a wide range and the absurd
negatives sit near p(gold)=1.0; under (b) everything piles up at .5.  This is the instrument test,
run before any E3 number is believed -- it needs no prefill, so it costs about a minute.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_e3probe --ds all --out ./out/e3probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest3 import load_realtalk, load_locomo, select_qa
from kvmemory.kv_harvest4 import N_INJECT, recog_q, hard_neg
from kvmemory.kv_e3ctrl import neg_pool, shape_class
from kvmemory.kv_e3negs import qkey, eligible

ABSURD = ["a banana", "Genghis Khan", "the year 1215"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["realtalk", "locomo", "all"], default="all")
    ap.add_argument("--rt_path", default="./data/REALTALK")
    ap.add_argument("--lc_path", default="./data/locomo10.json")
    ap.add_argument("--negs", default="./out/e3_negs.json")
    ap.add_argument("--out", default="./out/e3probe.json")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    HPOS = list(range(H))
    NEGS = json.load(open(args.negs))["negs"] if os.path.exists(args.negs) else {}

    tasks = []
    if args.ds in ("locomo", "all"):
        tasks += [("locomo", c) for c in load_locomo(args.lc_path)]
    if args.ds in ("realtalk", "all"):
        tasks += [("realtalk", c) for c in load_realtalk(args.rt_path)]
    golds_of = {"realtalk": set(), "locomo": set()}
    for ds, cv in tasks:
        for x in select_qa(cv):
            golds_of[ds].add(x["gold"].strip())
    for k in golds_of:
        golds_of[k] = sorted(golds_of[k])

    def p_gold(question, gold, neg, gold_is_a):
        a, b = (gold, neg) if gold_is_a else (neg, gold)
        c, p = assemble(llm, [(H_kv, HPOS)])
        _, cand = answer_with_probs(llm, c, p,
                                    llm._ids(recog_q(question, a, b) + tail + "Answer:"),
                                    ["A", "B"], ntok=2)
        del c
        pa, pb = cand.get("A", 0.0), cand.get("B", 0.0)
        tot = pa + pb
        return ((pa if gold_is_a else pb) / tot if tot > 0 else 0.5), tot

    rows = []
    for ds, cv in tasks:
        conv_low = "\n".join(t[2] for t in cv["turns"]).lower()
        qas = select_qa(cv)
        for qa in eligible(ds, cv):
            gold = qa["gold"].strip()
            ga = int(hashlib.sha1(("inj" + qa["q"]).encode()).hexdigest(), 16) % 2 == 0
            gen = NEGS.get(qkey(ds, cv["cid"], qa["q"]), [])
            pool, seen = [], set()
            for c in gen + neg_pool(gold, conv_low, golds_of[ds]):
                if c.lower() not in seen:
                    seen.add(c.lower())
                    pool.append(c)
            vals = []
            for cand in pool[:32]:
                for order in (True, False):
                    v, m = p_gold(qa["q"], gold, cand, order)
                    vals.append(v)
            frz = hard_neg(qas, qa)
            p_frz = p_gold(qa["q"], gold, frz, ga)[0] if frz else None
            p_abs = [p_gold(qa["q"], gold, a, ga)[0] for a in ABSURD]
            rec = {"ds": ds, "conv": cv["cid"], "gold": gold, "shape": shape_class(gold),
                   "n_pool": len(vals), "p_min": round(min(vals), 4),
                   "p_med": round(statistics.median(vals), 4), "p_max": round(max(vals), 4),
                   "p_frozen": round(p_frz, 4) if p_frz is not None else None,
                   "p_absurd": [round(x, 4) for x in p_abs],
                   "frac_near_half": round(sum(1 for v in vals if abs(v - .5) < .05) / len(vals), 3)}
            rows.append(rec)
            print(f"[probe] {cv['cid']} {gold[:22]!r} pool[{rec['p_min']:.2f} "
                  f"{rec['p_med']:.2f} {rec['p_max']:.2f}] frozen={rec['p_frozen']} "
                  f"absurd={rec['p_absurd']} near.5={rec['frac_near_half']}", flush=True)

    json.dump({"rows": rows, "absurd": ABSURD}, open(args.out, "w"), indent=1)
    n = len(rows)
    if n:
        m = lambda k: statistics.fmean(r[k] for r in rows if r[k] is not None)
        print(f"\n===== readout sensitivity (n={n}) =====")
        print(f"  candidate pool  p(gold): min {m('p_min'):.3f} | median {m('p_med'):.3f} "
              f"| max {m('p_max'):.3f}")
        print(f"  frozen negative p(gold): {m('p_frozen'):.3f}")
        p_abs = statistics.fmean(x for r in rows for x in r["p_absurd"])
        print(f"  absurd negative p(gold): {p_abs:.3f}")
        print(f"  fraction of the pool within .05 of chance: {m('frac_near_half'):.3f}")
        print("  READOUT IS CONTENT-SENSITIVE (absurd negatives are rejected)" if p_abs > .8
              else "  WARNING: readout may be option-blind -- G1 would certify nothing")
    print("E3PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
