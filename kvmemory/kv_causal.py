"""kv_causal.py -- causal nails for "grounded payload + symbolic edges" (collision-map #3).

Two parts, one script:

A. PAIRED DONOR SWAP (depth-3 chain). Each base is rendered twice, differing ONLY in the
   dropped source's value v1 (the source slot gets a fixed position budget, so every SERVED
   token is identical across the two members). Arms per member, trace readout:
     root  serve {decoy, C1}          ask M1
     full  serve {decoy, C1, C2, C3}  ask T
   If the payload lives in the grounded root and the chain merely routes to it, then
   (i) each arm's answer flips with the donor, and (ii) root and full flip TOGETHER at the
   item level (pair separation co-occurrence), which upgrades the A1~A5 mean-level match to a
   causal, item-level claim.

B. EDGE SWAP (per-edge counterfactual). One log holds TWO independent grounded chains:
     Sa=va@6 -> C1a@13 "M1 mirrors Sa"      Sb=vb@8 -> C1b@15 "M2 mirrors Sb"
   Leaf variants at slot 23 differ only in which chain they reference:
     LA "T mirrors M1"   vs   LB "T mirrors M2"     (slot length-padded across variants)
   Serve {decoy, C1a, C1b, leaf}; ask T. On the va != vb half, tracking rate
   P(answer == referenced root's value) >> .5 shows the SERVED TEXT EDGE causally selects
   which root's KV payload is read out -- the "symbolic edges" half made counterfactual.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_causal --items 24 --seed 3900 --out ./out/cz.json
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
TRACE = (" Locate the mirror statement for that register, then end your answer with the "
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


def ask(llm, H_ids, H, H_kv, tail, full_kv, rowmap, spans, eids, served, qt):
    crows = [pp for i in served for pp in range(spans[i][0], spans[i][0] + len(eids[i]))]
    rt = torch.tensor([rowmap[pp] for pp in crows], dtype=torch.long)
    sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
    o, _, _ = llm._greedy_pos(c, p, llm._ids(qt + tail), 160)
    del c, sub
    return pick_last(o)


def encode_variant(llm, H_ids, H, evs, var_slots):
    """Encode a log whose events at var_slots get position budgets = the max token length
    across that item's variants (budgets dict passed via var_slots {slot: budget})."""
    eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
    spans, cur = [], H
    for i, e in enumerate(eids):
        spans.append((cur, cur + len(e)))
        cur += var_slots.get(i, len(e))
    pos, rowmap, r = [], {}, 0
    for i in range(N_BODY):
        for pp in range(spans[i][0], spans[i][0] + len(eids[i])):
            pos.append(pp)
            rowmap[pp] = r
            r += 1
    flat = [t for e in eids for t in e]
    kv = encode_block(llm, H_ids + flat, list(range(H)) + pos, keep_a=H)
    return kv, rowmap, spans, eids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3900)
    ap.add_argument("--out", default="./out/cz.json")
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
    A = " Answer ONLINE or OFFLINE." + TRACE

    for it in range(args.items):
        # ---------- Part A: paired donor swap, depth-3 ----------
        D, S, M1, M2, T = rng.sample(NAMES, 5)
        dm = rng.choice(STATUS2)
        m = rng.randrange(8, 12)
        C1, C2, C3 = 13, 18, 23
        others = [n for n in NAMES if n not in (D, S, M1, M2, T)]
        fill = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
                for i in range(N_BODY)}
        recA = {"it": it, "part": "A", "dm": dm}
        src_budget = max(len(llm.tok(f"<step {m+1}>\naction: reg_op()\nobservation: "
                                     f"register {S} set to state {v}\n",
                                     add_special_tokens=False).input_ids)
                         for v in STATUS2)
        for v1 in STATUS2:
            body = dict(fill)
            body[DEC_IDX] = f"register {D} set to state {dm}"
            body[m] = f"register {S} set to state {v1}"
            body[C1] = f"register {M1} mirrors register {S}"
            body[C2] = f"register {M2} mirrors register {M1}"
            body[C3] = f"register {T} mirrors register {M2}"
            evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                   for i in range(N_BODY)]
            kv, rowmap, spans, eids = encode_variant(llm, H_ids, H, evs, {m: src_budget})
            recA[f"root_{v1}"] = ask(llm, H_ids, H, H_kv, tail, kv, rowmap, spans, eids,
                                     [DEC_IDX, C1],
                                     f"\n\nQuestion: What state is register {M1} in?{A}")
            recA[f"full_{v1}"] = ask(llm, H_ids, H, H_kv, tail, kv, rowmap, spans, eids,
                                     [DEC_IDX, C1, C2, C3],
                                     f"\n\nQuestion: What state is register {T} in?{A}")
            del kv
            torch.cuda.empty_cache()
        rows.append(recA)

        # ---------- Part B: edge swap, two grounded chains ----------
        D2, Sa, Sb, Ma, Mb, T2 = rng.sample(NAMES, 6)
        va, vb = rng.choice(STATUS2), rng.choice(STATUS2)
        others2 = [n for n in NAMES if n not in (D2, Sa, Sb, Ma, Mb, T2)]
        fill2 = {i: f"register {rng.choice(others2)} refreshed to {rng.randrange(100, 999)}"
                 for i in range(N_BODY)}
        leaf = {"LA": f"register {T2} mirrors register {Ma}",
                "LB": f"register {T2} mirrors register {Mb}"}
        leaf_budget = max(len(llm.tok(f"<step 24>\naction: reg_op()\nobservation: {t}\n",
                                      add_special_tokens=False).input_ids)
                          for t in leaf.values())
        recB = {"it": it, "part": "B", "va": va, "vb": vb, "diff": int(va != vb)}
        for lk, ltxt in leaf.items():
            body = dict(fill2)
            body[DEC_IDX] = f"register {D2} set to state {rng.choice(STATUS2)}"
            body[6] = f"register {Sa} set to state {va}"
            body[8] = f"register {Sb} set to state {vb}"
            body[13] = f"register {Ma} mirrors register {Sa}"
            body[15] = f"register {Mb} mirrors register {Sb}"
            body[23] = ltxt
            evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                   for i in range(N_BODY)]
            kv, rowmap, spans, eids = encode_variant(llm, H_ids, H, evs, {23: leaf_budget})
            recB[lk] = ask(llm, H_ids, H, H_kv, tail, kv, rowmap, spans, eids,
                           [DEC_IDX, 13, 15, 23],
                           f"\n\nQuestion: What state is register {T2} in?{A}")
            del kv
            torch.cuda.empty_cache()
        rows.append(recB)
        print(f"[cz] it{it} A: root {recA['root_ONLINE']}/{recA['root_OFFLINE']} "
              f"full {recA['full_ONLINE']}/{recA['full_OFFLINE']} | "
              f"B({va[:2]},{vb[:2]}): LA={recB['LA']} LB={recB['LB']}", flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    ga = [r for r in rows if r["part"] == "A"]
    n = len(ga)
    rsep = sum(1 for r in ga if r["root_ONLINE"] == "ONLINE"
               and r["root_OFFLINE"] == "OFFLINE") / n
    fsep = sum(1 for r in ga if r["full_ONLINE"] == "ONLINE"
               and r["full_OFFLINE"] == "OFFLINE") / n
    both = sum(1 for r in ga if r["root_ONLINE"] == "ONLINE"
               and r["root_OFFLINE"] == "OFFLINE"
               and r["full_ONLINE"] == "ONLINE"
               and r["full_OFFLINE"] == "OFFLINE") / n
    print(f"\nA: pairs={n} root-sep={rsep:.2f} full-sep={fsep:.2f} both-sep={both:.2f} "
          f"(donor-blind floor .25)")
    gb = [r for r in rows if r["part"] == "B" and r["diff"]]
    if gb:
        trk = sum((r["LA"] == r["va"]) + (r["LB"] == r["vb"]) for r in gb) / (2 * len(gb))
        wrg = sum((r["LA"] == r["vb"]) + (r["LB"] == r["va"]) for r in gb) / (2 * len(gb))
        print(f"B: diff-pairs={len(gb)} edge-tracking={trk:.2f} wrong-chain={wrg:.2f} "
              f"(floor .5)")
    print("CZ_DONE", flush=True)


if __name__ == "__main__":
    main()
