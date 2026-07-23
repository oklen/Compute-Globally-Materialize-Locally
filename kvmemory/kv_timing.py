"""kv_timing.py -- self-audit G5: real wall-clock for delta-append vs full recompute.

The paper's "delta = 1/17 of recompute" is a prefill-TOKEN ratio, not a latency claim.
Measure TTFT (query issued -> first output token, CUDA-synchronized) for the three serving
paths at two context scales:

  reuse        gather served rows -> assemble -> prefill question           (no update)
  reuse+delta  gather -> assemble -> prefill [36-tok patch + question]      (out-of-band fix)
  recompute    prefill [header + FULL log + question] from scratch

Scales: N=24 events (~0.7k tok, the synthetic protocol) and N=384 events (~10k tok, the
regime where recompute hurts). Warmup rep discarded; median of 5 measured reps.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_timing --out ./out/tm.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES


def build_log(rng, n_events):
    D, S, T = rng.sample(NAMES, 3)
    m = n_events // 2
    body = {}
    others = [x for x in NAMES if x not in (D, S, T)]
    for i in range(n_events):
        body[i] = f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
    body[0] = f"register {D} set to state OFFLINE"
    body[m] = f"register {S} set to state ONLINE"
    body[n_events - 1] = f"register {T} mirrors register {S}"
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
           for i in range(n_events)]
    patch = (f"<step {n_events+1}>\naction: memory_patch()\nobservation: register {S} "
             f"set to state OFFLINE   [{S} version 2]\n")
    q = f"\n\nQuestion: What state is register {T} in now? Answer ONLINE or OFFLINE:"
    return evs, patch, q, [0, n_events - 1]


def ttft(fn, reps=5):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out/tm.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(0)
    res = {}

    for n_events in (24, 384):
        evs, patch, q, served = build_log(rng, n_events)
        eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
        crows = [pp for i in served for pp in range(*spans[i])]
        rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
        q_ids = llm._ids(q + tail)
        pq_ids = llm._ids(patch + q + tail)
        full_ids = llm._ids(head_full + "".join(evs) + q + tail)
        n_served = len(crows)

        def path_reuse():
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            llm._greedy_pos(c, p, q_ids, 1)

        def path_delta():
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            llm._greedy_pos(c, p, pq_ids, 1)

        def path_recompute():
            llm._greedy(DynamicCache(), 0, full_ids, 1)

        r = {"ctx_tok": cur, "served_tok": n_served,
             "patch_tok": pq_ids.shape[1] - q_ids.shape[1],
             "reuse_ms": ttft(path_reuse), "delta_ms": ttft(path_delta),
             "recompute_ms": ttft(path_recompute)}
        r["delta_over_recompute"] = round(r["delta_ms"] / r["recompute_ms"], 3)
        r["delta_overhead_ms"] = round(r["delta_ms"] - r["reuse_ms"], 1)
        res[f"N{n_events}"] = r
        print(f"[tm] N={n_events} ctx={cur}tok served={n_served} | "
              f"reuse {r['reuse_ms']:.0f}ms  +delta {r['delta_ms']:.0f}ms  "
              f"recompute {r['recompute_ms']:.0f}ms  ratio={r['delta_over_recompute']}",
              flush=True)
        del full_kv
        torch.cuda.empty_cache()

    json.dump(res, open(args.out, "w"), indent=1)
    print("TM_DONE", flush=True)


if __name__ == "__main__":
    main()
