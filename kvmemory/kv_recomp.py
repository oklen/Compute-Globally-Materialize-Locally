"""kv_recomp.py -- the recompute-necessity matrix (reviewer-ordered).

Three write-time states x {stale / source delta / derived delta / recompute oracle}, all in
the out-of-band-update world. The matrix attributes each failure to stale KV, query-time
computation, or DELTA CONTENT INSUFFICIENCY -- and tests the corrected claim that a delta
carrying a high-entropy value works fine (the old ">1 bit -> patch useless" claim is wrong).

  Row A  materialized binary   src "S set to state v1", carrier "T mirrors S" (binds v1)
         update S:=v2.  Ask current state of T (menu).                    readout: direct
  Row B  operands-only         src "sensor S measured gv; threshold thr" (DROPPED), carrier
         "operator reviewed S against the alert threshold" (verdict NOT materialized, .48)
         update: NEW MEASUREMENT gv2 arrives out-of-band. Ask above-threshold now?
         source delta carries gv2 but NOT thr -> predicted FAIL (missing operand, not stale KV)
         derived delta carries the verdict -> predicted OK               readout: trace
  Row C  high-cardinality      src "S initialized to num1", carrier "T copied from S"
         (carries nothing at this entropy), update S:=num2. Ask value of T.
         source delta carries num2 -> query-time derivation from carrier TEXT + delta
         -> predicted OK (the delta supplies the high-entropy value)     readout: direct

Arms per row: stale (no delta) / d_src / d_der / oracle (delta restating source AND carrier
as fresh steps -- the ceiling any recompute could reach by append).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_recomp --items 32 --seed 3700 --out ./out/rx.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
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
DEC_IDX, CAR_IDX = 0, 23
ROWS = ["A_mat", "B_oper", "C_high"]
ARMS = ["stale", "d_src", "d_mtn", "d_der", "oracle"]
TRACE = (" Reason briefly, then end your answer with the final answer word or number.")


def pstep(body):
    return f"<step {N_BODY+1}>\naction: memory_patch()\nobservation: {body}\n"


def pick2(o, a, b):
    u = o.upper()
    ia, ib = u.rfind(a), u.rfind(b)
    if ia < 0 and ib < 0:
        return None
    if ia < 0:
        return b
    if ib < 0:
        return a
    return a if ia > ib else b


def gen_item(rng, row):
    D, S, T = rng.sample(NAMES, 3)
    others = [n for n in NAMES if n not in (D, S, T)]
    body = {}
    if row == "A_mat":
        v1 = rng.choice(STATUS2)
        v2 = rng.choice(STATUS2)
        dm = rng.choice(STATUS2)
        body[DEC_IDX] = f"register {D} set to state {dm}"
        m0 = rng.randrange(8, 16)
        body[m0] = f"register {S} set to state {v1}"
        body[CAR_IDX] = f"register {T} mirrors register {S}"
        q = (f"\n\nQuestion: What state is register {T} in now? Answer ONLINE or OFFLINE.")
        gold = v2
        deltas = {
            "d_src": pstep(f"register {S} set to state {v2}"),
            "d_mtn": pstep(f"[STATE UPDATE] register {S} \u2192 {v2}; overrides any "
                           "earlier value and conclusion"),
            "d_der": pstep(f"register {S} set to state {v2}; therefore register {T} is "
                           f"now {v2}"),
            "oracle": pstep(f"register {S} set to state {v2}") +
                      f"<step {N_BODY+2}>\naction: reg_op()\nobservation: register {T} "
                      f"mirrors register {S}\n",
        }
        meta = {"v1": v1, "v2": v2, "stale_gold": v1}
        score = lambda o, g: int(pick2(o, "ONLINE", "OFFLINE") == g)
    elif row == "B_oper":
        thr = rng.randrange(100, 999)
        gv = rng.randrange(100, 999)
        while abs(gv - thr) < 30:
            gv = rng.randrange(100, 999)
        gv2 = rng.randrange(100, 999)
        while abs(gv2 - thr) < 30:
            gv2 = rng.randrange(100, 999)
        gold = "EXCEEDED" if gv2 > thr else "NORMAL"
        body[DEC_IDX] = f"sensor {D} logged a routine idle reading"
        m0 = rng.randrange(8, 16)
        body[m0] = f"sensor {S} measured {gv}; the alert threshold is {thr}"
        body[CAR_IDX] = f"the operator reviewed sensor {S} against the alert threshold"
        q = (f"\n\nQuestion: Is sensor {S} above the alert threshold now? "
             "Answer EXCEEDED or NORMAL." + TRACE)
        deltas = {
            "d_src": pstep(f"sensor {S} measured {gv2}"),
            "d_mtn": pstep(f"[STATE UPDATE] sensor {S} measurement \u2192 {gv2}; overrides "
                           "any earlier value and conclusion"),
            "d_der": pstep(f"sensor {S} measured {gv2}; the verdict is now {gold}"),
            "oracle": pstep(f"sensor {S} measured {gv2}; the alert threshold is {thr}"),
        }
        meta = {"thr": thr, "gv2": gv2,
                "stale_gold": "EXCEEDED" if gv > thr else "NORMAL"}
        score = lambda o, g: int(pick2(o, "EXCEEDED", "NORMAL") == g)
    else:
        n1 = rng.randrange(100, 999)
        n2 = rng.randrange(100, 999)
        while n2 == n1:
            n2 = rng.randrange(100, 999)
        dm = rng.randrange(100, 999)
        body[DEC_IDX] = f"register {D} initialized to {dm}"
        m0 = rng.randrange(8, 16)
        body[m0] = f"register {S} initialized to {n1}"
        body[CAR_IDX] = f"register {T} copied from register {S}"
        q = (f"\n\nQuestion: What is the value of register {T} now? "
             "Answer with the number only.")
        gold = str(n2)
        deltas = {
            "d_src": pstep(f"register {S} set to {n2}"),
            "d_mtn": pstep(f"[STATE UPDATE] register {S} \u2192 {n2}; overrides any "
                           "earlier value and conclusion"),
            "d_der": pstep(f"register {S} set to {n2}; therefore register {T} is now {n2}"),
            "oracle": pstep(f"register {S} set to {n2}") +
                      f"<step {N_BODY+2}>\naction: reg_op()\nobservation: register {T} "
                      f"copied from register {S}\n",
        }
        meta = {"n1": n1, "n2": n2, "stale_gold": str(n1)}
        score = lambda o, g: int(g in re.findall(r"\d+", o))
    for i in range(N_BODY):
        if i not in body:
            body[i] = (f"register {rng.choice(others)} refreshed to "
                       f"{rng.randrange(100, 999)}")
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
           for i in range(N_BODY)]
    return evs, q, gold, deltas, meta, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3700)
    ap.add_argument("--out", default="./out/rx.json")
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

    dtok = int(os.environ.get("SPRAG_DIRECT_NTOK", "12"))
    for row in ROWS:
        ntok = 160 if row == "B_oper" else dtok
        for it in range(args.items):
            evs, q, gold, deltas, meta, score = gen_item(rng, row)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
            spans, cur = [], H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
            crows = [pp for i in (DEC_IDX, CAR_IDX) for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            rec = {"row": row, "it": it, "gold": gold, **meta}
            for arm in ARMS:
                pre = "" if arm == "stale" else deltas[arm]
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(pre + q + tail), ntok)
                del c
                g = meta["stale_gold"] if arm == "stale" else gold
                rec[arm] = score(o, g)
            del full_kv, sub
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[rx] {row} it{it} | " + " ".join(f"{a}={rec[a]}" for a in ARMS),
                  flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = args.items
    print("\n===== RECOMPUTE-NECESSITY MATRIX =====")
    for row in ROWS:
        g = [r for r in rows if r["row"] == row]
        print(f"{row:8s} " + " ".join(
            f"{a}={sum(r[a] for r in g)/len(g):.2f}" for a in ARMS))
    print("RX_DONE", flush=True)


if __name__ == "__main__":
    main()
