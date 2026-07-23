"""kv_mater3.py -- X10 (verdict-6 P0-3): serve-set ablation for active materialization.

X9 served {decoy, unrelated, carrier, downstream-review} together, so a donor-following
answer could be carried by the shared downstream-review row rather than the manipulated
carrier. This isolates them. Per (item, arm) we run THREE serve-sets on the same
donor-paired full-log cache:

  car   = {decoy, unrelated, carrier}          -- carrier row only (no review)
  rev   = {decoy, unrelated, review}           -- review row only  (no carrier)
  both  = {decoy, unrelated, carrier, review}  -- = X9

Prediction that would VALIDATE "the carrier event carries state":
  follow(car) >> follow(rev) ~ floor  -> the manipulated carrier row is the vehicle.
Prediction that would CONFIRM the confound (rename to "carrier bundle"):
  follow(rev) ~ follow(both) and follow(car) ~ floor -> the common review row is enough.

Carrier arms: pas/bind/dir/nld (latent, byte-identical across donors) + exp (upper bound).
Readouts: qmain (subject-addressed) + qtgt (target-addressed, for the readout-contract
point). Donor-paired; anti/const/other transition classes reported.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 ./out/tf5env/bin/python -m kvmemory.kv_mater3 \
        --items 2 --seed 999 --out ./out/x10.json
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
from kvmemory.kv_mater2 import ARMS, T_ARMS, DONORS, DEC_IDX, U_IDX, DST_IDX, N_BODY, pickv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1500)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--out", default="./out/x10.json")
    args = ap.parse_args()
    llm = HFBackend()
    print("MODEL_CLASS", type(llm.model).__name__, flush=True)
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    rows = []

    for it in range(args.items):
        tpl = it % 2
        D, S, T, U = rng.sample(NAMES, 4)
        us = rng.choice(["ONLINE", "OFFLINE"])
        thr = rng.randrange(200, 800)
        gv = {"hi": rng.randrange(thr + 30, 999), "lo": rng.randrange(100, thr - 29)}
        m = rng.randrange(N_BODY // 3, (2 * N_BODY) // 3)
        others = [n for n in NAMES if n not in (D, S, T, U)]
        fill = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
                for i in range(N_BODY)}
        fill[DEC_IDX] = f"sensor {D} logged a routine idle reading"
        fill[U_IDX] = f"register {U} set to state {us}"
        fill[DST_IDX] = f"the operator reviewed sensor {S} against the alert threshold"
        qmain = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
                 "Answer EXCEEDED or NORMAL:")
        qtgt = (f"\n\nQuestion: What verdict is recorded for status register {T}? "
                "Answer EXCEEDED or NORMAL:")
        src_txts = {d: f"sensor {S} measured {gv[d]}; the alert threshold is {thr}"
                    for d, _ in DONORS}
        src_budget = max(len(llm.tok(f"<step {m+1}>\naction: reg_op()\nobservation: "
                                     f"{src_txts[d]}\n", add_special_tokens=False).input_ids)
                         for d, _ in DONORS)
        rec = {"it": it, "tpl": tpl}
        from kvmemory.kv_mater2 import ARMS as ARMDEF
        for arm, tpls in ARMDEF.items():
            cars = {d: tpls[tpl].format(S=S, T=T, V=gold) for d, gold in DONORS}
            if arm != "exp":
                assert cars["hi"] == cars["lo"], arm
            car_budget = max(len(llm.tok(f"<step {m+2}>\naction: reg_op()\nobservation: "
                                         f"{cars[d]}\n", add_special_tokens=False).input_ids)
                             for d, _ in DONORS)
            setmap = {"car": [DEC_IDX, U_IDX, m + 1],
                      "rev": [DEC_IDX, U_IDX, DST_IDX],
                      "both": [DEC_IDX, U_IDX, m + 1, DST_IDX]}
            for d, gold in DONORS:
                body = dict(fill)
                body[m] = src_txts[d]
                body[m + 1] = cars[d]
                evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                       for i in range(N_BODY)]
                kv, rm, sp, ei = encode_variant(llm, H_ids, H, evs,
                                                {m: src_budget, m + 1: car_budget})
                for sname, served in setmap.items():
                    sub, cr = sub_rows(kv, rm, sp, ei, served)
                    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, cr)])
                    o, _, _ = llm._greedy_pos(c, p, llm._ids(qmain + tail), args.gen)
                    rec[f"{arm}_{sname}_h_{d}"] = pickv(o)
                    del c
                    if arm in T_ARMS:
                        c, p = assemble(llm, [(H_kv, list(range(H))), (sub, cr)])
                        o, _, _ = llm._greedy_pos(c, p, llm._ids(qtgt + tail), args.gen)
                        rec[f"{arm}_{sname}_t_{d}"] = pickv(o)
                        del c
                del kv
                torch.cuda.empty_cache()
        rows.append(rec)
        def fol(a, s):
            return int(rec[f"{a}_{s}_h_hi"] == "EXCEEDED" and rec[f"{a}_{s}_h_lo"] == "NORMAL")
        print(f"[x10] it{it} " + " ".join(
            f"{a}:car{fol(a,'car')}/rev{fol(a,'rev')}/both{fol(a,'both')}"
            for a in ARMDEF), flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    print(f"\n===== X10 serve-set ablation n={n} (follow-donor per set) =====")
    for a in ARMS:
        line = f"{a:4s} "
        for s in ("car", "rev", "both"):
            fo = sum(1 for r in rows
                     if r[f"{a}_{s}_h_hi"] == "EXCEEDED" and r[f"{a}_{s}_h_lo"] == "NORMAL")
            line += f"{s}={fo:3d} "
        print(line)
    print("X10_DONE", flush=True)


if __name__ == "__main__":
    main()
