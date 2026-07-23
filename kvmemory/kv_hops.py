"""kv_hops.py -- WHERE does the value live in a mirror chain? (user-ordered discrimination)

dg's flat-in-depth chained readout (.71/.70/.72) admits two mechanisms:
  (A) write-time TRANSITIVE resolution -- every carrier's rows bind the already-resolved v1,
      so read-time needs no chain walk;
  (B) TEXT walk + terminal bit -- the reader walks the chain in the served carriers' TEXT and
      takes the value from the KV of the one carrier that bound the source directly (C1).

Fixed depth-3 world: S=v1 at m (DROPPED); C1="M1 mirrors S"@13; C2="M2 mirrors M1"@18;
C3="T mirrors M2"@23; decoy@0 (independent). ONE causal prefill; arms serve subsets:

  A1 {C1} ask M1     1-hop positive control (binding law, expect ~.9)
  A2 {C2} ask M2     one-step transitivity: does C2 bind resolved v1 or just a pointer?
  A3 {C3} ask T      two-step transitivity: does the leaf independently carry v1?
  A4 {C2,C3} ask T   drop C1: if the value lives only in C1's rows, this collapses
  A5 {C1,C2,C3} ask T   dg replication (~.72)
  floor: A5's events re-read as TEXT, ask T (no value in text -> default/chance; leak check)

Readout: trace ("reason briefly then end with the final state word"), 160 tokens, menu in the
question, score the LAST state word. Verdict table:
  A3 high            -> full write-time transitivity; chained KV readout in the strongest form
  A3 low, A2 high    -> one-step transitivity only
  A2,A3,A4 low, A5 high -> the text-walk+terminal-bit reading (user's alternative) wins
  gradient A3<A4<A5  -> partial transitivity with decay

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_hops --items 32 --seed 2900 --out ./out/hp.json
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
C1, C2, C3 = 13, 18, 23
TRACE = (" Trace the mirror chain briefly (one line per hop), then end your answer with the "
         "final state word.")


def pick_last(o):
    u = o.upper()
    a, b = u.rfind("ONLINE"), u.rfind("OFFLINE")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "OFFLINE"
    if b < 0:
        return "ONLINE"
    return "ONLINE" if a > b else "OFFLINE"


def gen_item(rng):
    D, S, M1, M2, T = rng.sample(NAMES, 5)
    v1 = rng.choice(STATUS2)
    dm = rng.choice(STATUS2)
    m = rng.randrange(8, 12)
    others = [n for n in NAMES if n not in (D, S, M1, M2, T)]
    body = {DEC_IDX: f"register {D} set to state {dm}",
            m: f"register {S} set to state {v1}",
            C1: f"register {M1} mirrors register {S}",
            C2: f"register {M2} mirrors register {M1}",
            C3: f"register {T} mirrors register {M2}"}
    for i in range(N_BODY):
        if i not in body:
            body[i] = (f"register {rng.choice(others)} refreshed to "
                       f"{rng.randrange(100, 999)}")
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
           for i in range(N_BODY)]
    A = " Answer ONLINE or OFFLINE."

    def q(reg):
        return f"\n\nQuestion: What state is register {reg} in?{A}{TRACE}"
    arms = {
        "A1_c1_M1": ([DEC_IDX, C1], q(M1)),
        "A2_c2_M2": ([DEC_IDX, C2], q(M2)),
        "A3_c3_T": ([DEC_IDX, C3], q(T)),
        "A4_c23_T": ([DEC_IDX, C2, C3], q(T)),
        "A5_all_T": ([DEC_IDX, C1, C2, C3], q(T)),
    }
    return evs, arms, v1, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2900)
    ap.add_argument("--out", default="./out/hp.json")
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
        evs, arms, v1, dm = gen_item(rng)
        eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        total = cur
        flat = [t for e in eids for t in e]
        full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
        rec = {"it": it, "v1": v1, "dm": dm}
        for name, (served, qt) in arms.items():
            crows = [pp for i in served for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(qt + tail), 160)
            del c, sub
            rec[name] = pick_last(o)
        served, qt = arms["A5_all_T"]
        t = head_full + "".join(evs[i] for i in served) + qt + tail
        o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(t), 160)
        rec["floor_txt_T"] = pick_last(o)
        del full_kv
        torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[hp] it{it} v1={v1[:2]} | " + " ".join(
            f"{k.split('_')[0]}={'1' if rec[k] == v1 else '0'}"
            for k in list(arms) + ["floor_txt_T"]), flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    print(f"\n===== WHERE DOES THE VALUE LIVE (n={n}, chance .50) =====")
    for k in ["A1_c1_M1", "A2_c2_M2", "A3_c3_T", "A4_c23_T", "A5_all_T", "floor_txt_T"]:
        print(f"  {k:12s} acc={sum(1 for r in rows if r[k] == r['v1'])/n:.3f}")
    print("HP_DONE", flush=True)


if __name__ == "__main__":
    main()
