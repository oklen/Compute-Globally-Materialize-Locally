"""kv_causal3.py -- X5c (verdict-4): the downstream span is a TRUE REFERENCING EDGE.

kv_causal2's downstream was a value-free mention, licensing only "root is the only tested
span with a stable value-aligned native causal effect". To support a "only sufficient
carrier"-grade claim the downstream must be an actual referencing edge. Here
DW = "register T mirrors register M1" (references the root's register; value-free surface,
identical across donors). Cells per item (2 encodes, donor v in {ONLINE, OFFLINE}):

  2x2      serve {DEC, C1@i, DW@j}, ask T   -- readout THROUGH the served edge; prediction:
           answer follows ROOT donor i, edge donor j inert (equivalence bound +/-.075)
  rootonly serve {DEC, C1@i},        ask M1  -- kv_causal bridge (replication)
  edgeonly serve {DEC, DW@j},        ask T   -- does the edge span's OWN KV carry the value?
           (prediction from §3.1 leaf-only .48: no; donor-transition classes reported)

Same operative null as kv_causal2 (KV-no-effect + greedy => sensitive pairs <= bf16 floor).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_causal3 --items 1 --seed 999 --out ./out/cz3.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES
from kvmemory.kv_causal2 import pick_last, encode_variant, sub_rows

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX, SRC_LO, SRC_HI, C1_IDX, DW_IDX = 0, 8, 12, 13, 18
TRACE = (" Locate the relevant statement, then end your answer with the final state word.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=9500)
    ap.add_argument("--out", default="./out/cz3.json")
    ap.add_argument("--menu", action="store_true")
    ap.add_argument("--gen", type=int, default=160)
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
        D, S, M1, T = rng.sample(NAMES, 4)
        dm = rng.choice(STATUS2)
        m = rng.randrange(SRC_LO, SRC_HI)
        others = [n for n in NAMES if n not in (D, S, M1, T)]
        fill = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
                for i in range(N_BODY)}
        src_budget = max(len(llm.tok(f"<step {m+1}>\naction: reg_op()\nobservation: "
                                     f"register {S} set to state {v}\n",
                                     add_special_tokens=False).input_ids)
                         for v in STATUS2)
        enc = {}
        for v in STATUS2:
            body = dict(fill)
            body[DEC_IDX] = f"register {D} set to state {dm}"
            body[m] = f"register {S} set to state {v}"
            body[C1_IDX] = f"register {M1} mirrors register {S}"
            body[DW_IDX] = f"register {T} mirrors register {M1}"
            evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                   for i in range(N_BODY)]
            enc[v] = encode_variant(llm, H_ids, H, evs, {m: src_budget})
        if args.menu:
            qT = (f"\n\nQuestion: What state is register {T} in? "
                  "Answer with the single state word only (ONLINE or OFFLINE):")
            qM = (f"\n\nQuestion: What state is register {M1} in? "
                  "Answer with the single state word only (ONLINE or OFFLINE):")
        else:
            qT = f"\n\nQuestion: What state is register {T} in? Answer ONLINE or OFFLINE." + TRACE
            qM = f"\n\nQuestion: What state is register {M1} in? Answer ONLINE or OFFLINE." + TRACE
        rec = {"it": it, "dm": dm}
        for i in STATUS2:
            kv_i, rm_i, sp_i, ei_i = enc[i]
            sub_i, cr_i = sub_rows(kv_i, rm_i, sp_i, ei_i, [DEC_IDX, C1_IDX])
            for j in STATUS2:
                kv_j, rm_j, sp_j, ei_j = enc[j]
                sub_j, cr_j = sub_rows(kv_j, rm_j, sp_j, ei_j, [DW_IDX])
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub_i, cr_i), (sub_j, cr_j)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(qT + tail), args.gen)
                rec[f"t_{i[:2]}_{j[:2]}"] = pick_last(o)
                del c, sub_j
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub_i, cr_i)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(qM + tail), args.gen)
            rec[f"m_{i[:2]}"] = pick_last(o)
            del c, sub_i
        for j in STATUS2:
            kv_j, rm_j, sp_j, ei_j = enc[j]
            sub_j, cr_j = sub_rows(kv_j, rm_j, sp_j, ei_j, [DEC_IDX, DW_IDX])
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub_j, cr_j)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(qT + tail), args.gen)
            rec[f"e_{j[:2]}"] = pick_last(o)
            del c, sub_j
        del enc
        torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[cz3] it{it} T2x2={rec['t_ON_ON']}/{rec['t_ON_OF']}/{rec['t_OF_ON']}/"
              f"{rec['t_OF_OF']} M={rec['m_ON']}/{rec['m_OF']} E={rec['e_ON']}/{rec['e_OF']}",
              flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    fr = {(i, j): sum(1 for r in rows if r[f"t_{i[:2]}_{j[:2]}"] == i)/n
          for i in STATUS2 for j in STATUS2}
    print("\n===== ask-T through served edge: follow-ROOT rates =====")
    for i in STATUS2:
        print(f"  root={i:7s}: j=ON {fr[(i,'ONLINE')]:.2f}  j=OFF {fr[(i,'OFFLINE')]:.2f}")
    ms = sum(1 for r in rows if (r["m_ON"], r["m_OF"]) == ("ONLINE", "OFFLINE"))/n
    es = sum(1 for r in rows if r["e_ON"] != r["e_OF"])
    print(f"rootonly M follow-pair={ms:.2f} | edgeonly donor-sensitive={es}/{n}")
    print("CZ3_DONE", flush=True)


if __name__ == "__main__":
    main()
