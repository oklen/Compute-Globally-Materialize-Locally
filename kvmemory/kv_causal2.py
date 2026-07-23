"""kv_causal2.py -- 2x2 root-donor x downstream-donor factorial (verdict-3 P0-3).

kv_causal proved the root carrier's KV carries a causal donor imprint (99/99 donor-sensitive
pairs flip with the donor, 0 anti-flips). What it did NOT test: whether a DOWNSTREAM event's
KV (an event that also attended to the source during encoding) carries a behaviorally used
imprint too. "The grounded root is the only behaviorally sufficient carrier" needs the full
factorial: independently swap the root span's donor and the downstream span's donor while
every SERVED surface token stays identical.

Design (frozen):
  Log of 24 events. slot 0 decoy ("register D set to state dm", dm independent). slot m in
  [8,12) source "register S set to state {v}" -- position-budgeted so both donor variants
  occupy identical position ranges; source NEVER served. slot 13 root carrier C1
  "register M1 mirrors register S". slot 18 downstream mention DW "register M1 was re-checked
  during the routine audit; no anomalies were noted" (references M1, value-free, identical
  surface across donors). Fillers numeric.
  Encode the full log once per donor v in {ONLINE, OFFLINE}. Serve {DEC, C1@donor_i,
  DW@donor_j} for all four (i, j) cells; DEC sits before the source so its rows are
  donor-invariant. Ask M1 with trace readout (160 tok, pick-last, menu).

Predictions + pre-registered tests:
  - answer follows the ROOT donor: orientation among root-sensitive pairs ~100% follow.
  - downstream donor has NO effect: per-item concordance ans(i,A)==ans(i,B); pre-registered
    equivalence bound on the downstream marginal |P(follow-root | j=i) - P(follow-root |
    j!=i)| < 0.075 (TOST at alpha=.05 on the paired difference).
  Null for sensitivity counts: KV-no-effect + greedy determinism => identical answers across
  donors => zero sensitive pairs (NOT the .25 independent-guessing floor).

AMENDMENT v2 (documented stimulus fix, 2026-07-18 14:30). The v1 run used DW text "...no
anomalies were noted", whose health implication is menu-valenced: Qwen3-8B collapsed to
const-ONLINE in 123/128 items (root-sensitivity 2.7% vs kv_causal's 51.6% without any DW),
while Gemma2-9B (65.1% sensitive, 124/1 follow) proved the harness itself sound. v1's
DOWNSTREAM-equivalence conclusion is unaffected (the pull comes from DW's donor-independent
token content, not its donor imprint; both models passed TOST in v1). v2 changes: (a) DW text
replaced by a valence-free mention; (b) two bridge cells j=none (serve {DEC, C1} only) added
per root donor, replicating kv_causal's root arm in-run and quantifying the mention's
content-pull directly (three-point comparison: no-DW / neutral-DW / v1's valenced-DW).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_causal2 --items 1 --seed 999 --out ./out/cz2.json
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

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX, SRC_LO, SRC_HI, C1_IDX, DW_IDX = 0, 8, 12, 13, 18
TRACE = (" Locate the relevant statement, then end your answer with the final state word.")
DW_STYLES = {
    "neutral": "register {m} was listed in the scheduled audit inventory for this cycle",
    "audit_v1": "register {m} was re-checked during the routine audit; no anomalies were noted",
}


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


def encode_variant(llm, H_ids, H, evs, var_slots):
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


def sub_rows(full_kv, rowmap, spans, eids, served):
    crows = [pp for i in served for pp in range(spans[i][0], spans[i][0] + len(eids[i]))]
    rt = torch.tensor([rowmap[pp] for pp in crows], dtype=torch.long)
    sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
    return sub, crows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=8200)
    ap.add_argument("--dw", default="neutral", choices=sorted(DW_STYLES))
    ap.add_argument("--menu", action="store_true",
                    help="menu/direct readout (no CoT trace); for checkpoints whose "
                         "free-generation interface is unreliable, e.g. Gemma-4")
    ap.add_argument("--gen", type=int, default=160)
    ap.add_argument("--out", default="./out/cz2.json")
    args = ap.parse_args()
    dw_txt = DW_STYLES[args.dw]
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
        D, S, M1 = rng.sample(NAMES, 3)
        dm = rng.choice(STATUS2)
        m = rng.randrange(SRC_LO, SRC_HI)
        others = [n for n in NAMES if n not in (D, S, M1)]
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
            body[DW_IDX] = dw_txt.format(m=M1)
            evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                   for i in range(N_BODY)]
            enc[v] = encode_variant(llm, H_ids, H, evs, {m: src_budget})
        if args.menu:
            q = (f"\n\nQuestion: What state is register {M1} in? "
                 "Answer with the single state word only (ONLINE or OFFLINE):")
        else:
            q = f"\n\nQuestion: What state is register {M1} in? Answer ONLINE or OFFLINE." + TRACE
        rec = {"it": it, "dm": dm}
        for i in STATUS2:
            kv_i, rm_i, sp_i, ei_i = enc[i]
            sub_i, crows_i = sub_rows(kv_i, rm_i, sp_i, ei_i, [DEC_IDX, C1_IDX])
            for j in STATUS2:
                kv_j, rm_j, sp_j, ei_j = enc[j]
                sub_j, crows_j = sub_rows(kv_j, rm_j, sp_j, ei_j, [DW_IDX])
                c, p = assemble(llm, [(H_kv, list(range(H))),
                                      (sub_i, crows_i), (sub_j, crows_j)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), args.gen)
                rec[f"a_{i[:2]}_{j[:2]}"] = pick_last(o)
                del c, sub_j
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub_i, crows_i)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), args.gen)
            rec[f"a_{i[:2]}_NO"] = pick_last(o)
            del c, sub_i
        del enc
        torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[cz2] it{it} ON_ON={rec['a_ON_ON']} ON_OF={rec['a_ON_OF']} "
              f"OF_ON={rec['a_OF_ON']} OF_OF={rec['a_OF_OF']} "
              f"none={rec['a_ON_NO']}/{rec['a_OF_NO']}", flush=True)

    json.dump({"rows": rows, "dw": args.dw}, open(args.out, "w"), indent=1)
    n = len(rows)
    fr = {(i, j): sum(1 for r in rows if r[f"a_{i[:2]}_{j[:2]}"] == i) / n
          for i in STATUS2 for j in STATUS2}
    print(f"\n===== 2x2 follow-root rates (dw={args.dw}; root donor i, downstream donor j) =====")
    for i in STATUS2:
        print(f"  root={i:7s}: j=ONLINE {fr[(i,'ONLINE')]:.2f}  j=OFFLINE {fr[(i,'OFFLINE')]:.2f}  "
              f"j=none {sum(1 for r in rows if r[f'a_{i[:2]}_NO'] == i)/n:.2f}")
    rs = sum(1 for r in rows
             if (r["a_ON_ON"], r["a_ON_OF"]) != (r["a_OF_ON"], r["a_OF_OF"]))
    ds = sum(1 for r in rows
             if r["a_ON_ON"] != r["a_ON_OF"] or r["a_OF_ON"] != r["a_OF_OF"])
    print(f"root-sensitive items={rs}/{n}  downstream-sensitive items={ds}/{n}")
    print("CZ2_DONE", flush=True)


if __name__ == "__main__":
    main()
