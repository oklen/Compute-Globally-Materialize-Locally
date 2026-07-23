"""kv_bicap.py -- is the ~1-bit capacity PER ROW or PER BOUND VARIABLE? (user question)

The capacity law so far: a served row recovers a bound binary state at ~.93-.98, a 4-way label
at chance. But dir4 is 2 bits in ONE variable -- it cannot distinguish "the row's total budget
is 1 bit" from "each bound variable gets ~1 bit but a variable must be binary". TWO independent
binary variables separate the readings:

  one         step 24: "T1 mirrors S1"                                  -- calibration (~.93)
  two_in_one  step 24: "T1 mirrors S1; T2 mirrors S2"  (ONE observation binding TWO states)
  two_sep     step 23: "T1 mirrors S1" / step 24: "T2 mirrors S2"      (one binding per event)

Both sources are dropped; v1, v2, and the decoy's state are all independent draws. Same base
item (names, values, fillers) rendered in all three variants -> paired comparisons. Queries ask
each target separately, with the menu in the question. sel_txt is the per-variant floor (served
text carries no value); cross-talk is measured on the v1 != v2 half as P(answer_T1 == v2).

Outcomes:
  both recovered in two_in_one -> capacity is per-binding (law: the constraint is per-variable
                                  entropy, not per-row total)
  two_in_one collapses, two_sep survives -> capacity is per-event/sentence (~1 bit each)
  both collapse -> a shared budget across served rows

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_bicap --items 32 --seed 2400 --out ./out/bc.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX = 0
VARIANTS = ["one", "two_in_one", "two_sep"]


def pick(o):
    u = o.upper()
    a, b = u.find("ONLINE"), u.find("OFFLINE")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "OFFLINE"
    if b < 0:
        return "ONLINE"
    return "ONLINE" if a < b else "OFFLINE"


def gen_base(rng, swap=0):
    """swap=1 puts S1's source LATER than S2's while keeping clause/query order fixed --
    separates source-recency from clause-order as the driver of the first-binding decay."""
    D, S1, T1, S2, T2 = rng.sample(NAMES, 5)
    v1, v2, dm = (rng.choice(STATUS2) for _ in range(3))
    if swap:
        m2 = rng.randrange(8, 12)
        m1 = rng.randrange(12, 16)
    else:
        m1 = rng.randrange(8, 12)
        m2 = rng.randrange(12, 16)
    others = [n for n in NAMES if n not in (D, S1, T1, S2, T2)]
    fill = {}
    for i in range(N_BODY):
        if i in (DEC_IDX, m1, m2, 22, 23):
            continue
        fill[i] = f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
    fill22 = f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
    A = " Answer ONLINE or OFFLINE:"
    return {"D": D, "S1": S1, "T1": T1, "S2": S2, "T2": T2, "v1": v1, "v2": v2,
            "dm": dm, "m1": m1, "m2": m2, "fill": fill, "fill22": fill22,
            "q1": f"\n\nQuestion: What state is register {T1} in?{A}",
            "q2": f"\n\nQuestion: What state is register {T2} in?{A}"}


def events(b, variant):
    body = dict(b["fill"])
    body[DEC_IDX] = f"register {b['D']} set to state {b['dm']}"
    body[b["m1"]] = f"register {b['S1']} set to state {b['v1']}"
    body[b["m2"]] = f"register {b['S2']} set to state {b['v2']}"
    if variant == "one":
        body[22] = b["fill22"]
        body[23] = f"register {b['T1']} mirrors register {b['S1']}"
        served = [DEC_IDX, 23]
    elif variant == "two_in_one":
        body[22] = b["fill22"]
        body[23] = (f"register {b['T1']} mirrors register {b['S1']}; "
                    f"register {b['T2']} mirrors register {b['S2']}")
        served = [DEC_IDX, 23]
    else:
        body[22] = f"register {b['T1']} mirrors register {b['S1']}"
        body[23] = f"register {b['T2']} mirrors register {b['S2']}"
        served = [DEC_IDX, 22, 23]
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
           for i in range(N_BODY)]
    return evs, served


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2400)
    ap.add_argument("--swap", type=int, default=0)
    ap.add_argument("--out", default="./out/bc.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    rows = []

    for it in range(args.items):
        b = gen_base(rng, swap=args.swap)
        rec = {"it": it, "v1": b["v1"], "v2": b["v2"], "dm": b["dm"],
               "diff": int(b["v1"] != b["v2"])}
        for variant in VARIANTS:
            evs, served = events(b, variant)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
            spans, cur = [], H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
            crows = [pp for i in served for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]

            qs = [("q1", b["q1"])] if variant == "one" else \
                 [("q1", b["q1"]), ("q2", b["q2"])]
            for qk, qt in qs:
                t = (head_full + "".join(evs[i] for i in served) + qt + tail)
                o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(t), 8)
                rec[f"{variant}_{qk}_sel"] = pick(o)
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(qt + tail), 8)
                del c
                rec[f"{variant}_{qk}_harv"] = pick(o)
            del full_kv, sub
            torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[bc] it{it} v1={b['v1'][:2]} v2={b['v2'][:2]} | " + " ".join(
            f"{v}:{rec.get(f'{v}_q1_harv','')[:2]}/{rec.get(f'{v}_q2_harv','-')[:2]}"
            for v in VARIANTS), flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    print(f"\n===== BIT CAPACITY: per row or per binding? (n={n}, chance .50) =====")
    for v in VARIANTS:
        qs = ["q1"] if v == "one" else ["q1", "q2"]
        line = f"{v:11s} "
        for qk in qs:
            gold = "v1" if qk == "q1" else "v2"
            ha = sum(1 for r in rows if r[f"{v}_{qk}_harv"] == r[gold]) / n
            se = sum(1 for r in rows if r[f"{v}_{qk}_sel"] == r[gold]) / n
            line += f"{qk}: sel={se:.3f} harv={ha:.3f}   "
        if v != "one":
            both = sum(1 for r in rows if r[f"{v}_q1_harv"] == r["v1"]
                       and r[f"{v}_q2_harv"] == r["v2"]) / n
            dd = [r for r in rows if r["diff"]]
            xt = sum(1 for r in dd if r[f"{v}_q1_harv"] == r["v2"]) / max(1, len(dd))
            line += f"BOTH={both:.3f} xtalk(T1<-v2|diff)={xt:.2f}"
        print(line)
    print("BC_DONE", flush=True)


if __name__ == "__main__":
    main()
