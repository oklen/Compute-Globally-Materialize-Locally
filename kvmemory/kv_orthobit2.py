"""kv_orthobit2.py -- orthogonal bits, take 2: neutral labels (the YES/NO version collapsed
to an always-NO default -- YES rate 5%/0% -- so it could not distinguish "no info" from
"info present but YES-averse"). Labels RED/BLUE carry no default-class semantics and the
mapping is defined inside the question.

Behavioral failure of a 4-way readout cannot upper-bound the information inside the rows. So:
for the SAME served cache (dir4 substrate: decoy independent, source dropped, carrier
"T copied from S"), ask THREE readouts per item:

  q4      native 4-way menu (replicates the .223 cliff cell)
  bit_a   "If NORTH or SOUTH answer RED; if EAST or WEST answer BLUE."   (axis bit)
  bit_b   "If NORTH or EAST answer RED; if SOUTH or WEST answer BLUE."   (diagonal bit)

The two bits are orthogonal: (Y,Y)=N (Y,N)=S (N,Y)=E (N,N)=W. Response bias floors at .5 and
cannot fake success. sel_txt runs the same bits as the leak floor.

  bits ~.85+ each, reconstruction >> .22  ->  the information IS in the rows; the cliff is a
                                              READOUT limitation of native 4-way decoding
  bits ~.5                               ->  natively accessible information is genuinely
                                              narrow (the conservative reading survives)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_orthobit2 --items 32 --seed 3600 --out ./out/o4.json
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

DIR4 = ["NORTH", "SOUTH", "EAST", "WEST"]
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23
AX = {"NORTH": "RED", "SOUTH": "RED", "EAST": "BLUE", "WEST": "BLUE"}
DG = {"NORTH": "RED", "EAST": "RED", "SOUTH": "BLUE", "WEST": "BLUE"}
REC = {("RED", "RED"): "NORTH", ("RED", "BLUE"): "SOUTH",
       ("BLUE", "RED"): "EAST", ("BLUE", "BLUE"): "WEST"}


def pick4(o):
    u = o.upper()
    hits = [(u.find(w), w) for w in DIR4 if u.find(w) >= 0]
    return min(hits)[1] if hits else None


def pickyn(o):
    u = o.upper()
    a, b = u.find("RED"), u.find("BLUE")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "BLUE"
    if b < 0:
        return "RED"
    return "RED" if a < b else "BLUE"


def gen_item(rng):
    D, S, T = rng.sample(NAMES, 3)
    gold = rng.choice(DIR4)
    dm = rng.choice(DIR4)
    m = rng.randrange(8, 16)
    others = [n for n in NAMES if n not in (D, S, T)]
    body = {DEC_IDX: f"register {D} set to state {dm}",
            m: f"register {S} set to state {gold}",
            CAR_IDX: f"register {T} copied from register {S}"}
    for i in range(N_BODY):
        if i not in body:
            body[i] = (f"register {rng.choice(others)} refreshed to "
                       f"{rng.randrange(100, 999)}")
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
           for i in range(N_BODY)]
    qs = {
        "q4": (f"\n\nQuestion: What state is register {T} in? "
               "Answer NORTH, SOUTH, EAST, or WEST:"),
        "bit_a": (f"\n\nQuestion: If register {T}'s state is NORTH or SOUTH, answer RED. "
                  "If it is EAST or WEST, answer BLUE. Answer RED or BLUE:"),
        "bit_b": (f"\n\nQuestion: If register {T}'s state is NORTH or EAST, answer RED. "
                  "If it is SOUTH or WEST, answer BLUE. Answer RED or BLUE:"),
    }
    return evs, qs, gold, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3500)
    ap.add_argument("--out", default="./out/o4.json")
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
        evs, qs, gold, dm = gen_item(rng)
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
        rec = {"it": it, "gold": gold, "dm": dm}
        for qk, qt in qs.items():
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(qt + tail), 10)
            del c
            rec[f"harv_{qk}"] = (pick4 if qk == "q4" else pickyn)(o)
            t = (head_full + "".join(evs[i] for i in (DEC_IDX, CAR_IDX)) + qt + tail)
            o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(t), 10)
            rec[f"sel_{qk}"] = (pick4 if qk == "q4" else pickyn)(o)
        del full_kv, sub
        torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[o4] it{it} gold={gold[:2]} | q4={rec['harv_q4']} "
              f"a={rec['harv_bit_a']} b={rec['harv_bit_b']}", flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    print(f"\n===== ORTHOGONAL-BIT READOUT (n={n}) =====")
    q4 = sum(1 for r in rows if r["harv_q4"] == r["gold"]) / n
    ba = sum(1 for r in rows if r["harv_bit_a"] == AX[r["gold"]]) / n
    bb = sum(1 for r in rows if r["harv_bit_b"] == DG[r["gold"]]) / n
    rc = sum(1 for r in rows
             if REC.get((r["harv_bit_a"], r["harv_bit_b"])) == r["gold"]) / n
    print(f"harv: 4-way={q4:.3f} (chance .25) | bit_a={ba:.3f} bit_b={bb:.3f} "
          f"(chance .50) | reconstructed 4-way={rc:.3f}")
    sq4 = sum(1 for r in rows if r["sel_q4"] == r["gold"]) / n
    sa_ = sum(1 for r in rows if r["sel_bit_a"] == AX[r["gold"]]) / n
    sb_ = sum(1 for r in rows if r["sel_bit_b"] == DG[r["gold"]]) / n
    print(f"sel_txt floor: 4-way={sq4:.3f} bit_a={sa_:.3f} bit_b={sb_:.3f}")
    print("O4_DONE", flush=True)


if __name__ == "__main__":
    main()
