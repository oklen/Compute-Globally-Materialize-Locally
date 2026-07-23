"""kv_digit.py -- self-audit G3/X7: weak-signal probes for the three-digit payload.

The 0/128 exact-recall cell licenses "exact readout inaccessible", NOT "internally absent".
Probe the same serving condition (source with a 3-digit value DROPPED; mirror carrier served)
for partial information: (a) exact generation [replication], (b) binary quantile
"greater than 500?" via generation AND forced-position candidate-logit probe,
(c) first-digit via candidate-logit probe over "1".."9". txt arm (serve all) calibrates
each probe. Value sampling balanced across the 500 boundary.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_digit --items 1 --seed 999 --out ./out/dgt.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES
from kvmemory.kv_fail2 import answer_with_probs

N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23
DIGITS = [str(d) for d in range(1, 10)]


def build(rng, specials, n=N_BODY):
    others = [x for x in NAMES if x not in specials.get("_names", [])]
    body = {i: t for i, t in specials.items() if i != "_names"}
    for i in range(n):
        if i not in body:
            body[i] = f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
    return [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n" for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=9200)
    ap.add_argument("--out", default="./out/dgt.json")
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
        D, S, T = rng.sample(NAMES, 3)
        hi = it % 2 == 0
        val = rng.randrange(501, 999) if hi else rng.randrange(100, 500)
        m = rng.randrange(8, 16)
        sp = {DEC_IDX: f"register {D} completed a routine self-test",
              m: f"register {S} refreshed to {val}",
              CAR_IDX: f"register {T} mirrors register {S}", "_names": [D, S, T]}
        evs = build(rng, sp)
        eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
        rec = {"it": it, "val": val, "hi": int(hi)}
        for arm, served in (("kv", [DEC_IDX, CAR_IDX]), ("txt", list(range(len(evs))))):
            crows = [pp for i in served for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]

            def fresh():
                return assemble(llm, [(H_kv, list(range(H))), (sub, crows)])

            qx = (f"\n\nQuestion: What is the numeric value of register {T}? "
                  "Answer with the number only.")
            c, p = fresh()
            o, _, _ = llm._greedy_pos(c, p, llm._ids(qx + tail), 12)
            mnum = re.search(r"\d{3}", o)
            rec[f"{arm}_exact"] = int(bool(mnum) and int(mnum.group()) == val)
            del c
            qq = (f"\n\nQuestion: Is the numeric value of register {T} greater than 500? "
                  "Answer YES or NO.")
            c, p = fresh()
            raw, cp = answer_with_probs(llm, c, p, llm._ids(qq + tail + "Answer:"),
                                        ["YES", "NO"])
            gold_q = "YES" if hi else "NO"
            alt_q = "NO" if hi else "YES"
            rec[f"{arm}_quant_ok"] = int(cp.get(gold_q, 0) > cp.get(alt_q, 0))
            rec[f"{arm}_quant_pg"] = round(cp.get(gold_q, 0), 5)
            del c
            qd = (f"\n\nQuestion: What is the first digit of register {T}'s numeric value? "
                  "Answer with a single digit.")
            c, p = fresh()
            raw, cp = answer_with_probs(llm, c, p, llm._ids(qd + tail + "Answer:"), DIGITS)
            gold_d = str(val)[0]
            best = max(cp, key=cp.get)
            rec[f"{arm}_digit_ok"] = int(best == gold_d)
            rec[f"{arm}_digit_pg"] = round(cp.get(gold_d, 0), 5)
            del c, sub
        del full_kv
        torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[dgt] it{it} val={val} kv e{rec['kv_exact']}q{rec['kv_quant_ok']}"
              f"d{rec['kv_digit_ok']} | txt e{rec['txt_exact']}q{rec['txt_quant_ok']}"
              f"d{rec['txt_digit_ok']}", flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    for arm in ("kv", "txt"):
        print(f"{arm}: exact={sum(r[f'{arm}_exact'] for r in rows)/n:.2f} "
              f"quant={sum(r[f'{arm}_quant_ok'] for r in rows)/n:.2f} (chance .5) "
              f"digit={sum(r[f'{arm}_digit_ok'] for r in rows)/n:.2f}")
    print("DGT_DONE", flush=True)


if __name__ == "__main__":
    main()
