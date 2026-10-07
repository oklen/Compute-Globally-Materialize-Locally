"""kv_econ.py -- serving economics of sparse event-KV against keeping the same evidence as text.

The paper's timing claim is a single point (77 ms to append a patch vs 1033 ms to recompute at
L = 9.2k). That says reuse is cheaper per query; it does not say when reuse is cheaper *overall*,
because it prices neither the write nor the storage. This script prices all three and reports the
break-even query count.

Per served-evidence budget L, both arms answer the SAME question over the SAME evidence:

  text   store the trajectory as text; at query time re-prefill header + selected text + question
         C_read = TTFT of a cold prefill over the selected tokens          (the standard RAG path)
  kv     prefill the trajectory once; at query time gather the selected rows and prefill only the
         question                                                          (the paper's serving path)

Two regimes, because the write cost is not the same question in both:

  A  the cache is a BY-PRODUCT. The agent's history was prefilled once to run the agent (the paper's
     own premise), so keeping the rows costs no extra compute -- marginal write = 0, break-even is
     immediate, and the entire question is whether the storage is affordable.
  B  the memory is MATERIALIZED OFFLINE (rehydrated from logs, built by a memory service). The full
     prefill must be paid, and

         N* = C_write / (C_read_text - C_read_kv)

     queries against that memory are needed before it pays for itself.

Storage is reported as the ratio that decides regime A: KV bytes per token comes from the model
config (2 * layers * kv_heads * head_dim * 2 for bf16 K+V); text bytes are the actual UTF-8 length.
The ratio is the number the eviction literature does not print, and it is large enough that
"event-KV is cheaper memory" is not an available claim -- only "event-KV is a lower-latency read".

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_econ --targets 1000,4000,9200,16000 \
          --n_steps 900 --reps 5 --out ./out/econ.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import build_trajectory, build_full_ids, split_wrap_nothink


def cache_bytes_per_token(model) -> int:
    c = model.config
    L = c.num_hidden_layers
    kvh = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    hd = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    return 2 * L * kvh * hd * 2          # K + V, bf16


def timed(fn, reps: int):
    """Median of `reps` measured runs; one warmup discarded. CUDA-synchronised by the callee."""
    fn()
    return st.median([fn() for _ in range(reps)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="1000,4000,9200,16000")
    ap.add_argument("--n_steps", type=int, default=900)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--out", default="./out/econ.json")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    bpt = cache_bytes_per_token(llm.model)

    print(f"MODEL {os.environ.get('SPRAG_MODEL_PATH','?')}  attn={os.environ.get('SPRAG_ATTN_IMPL','default')}")
    print(f"KV CACHE: {bpt/1024:.1f} KB/token  |  {bpt/1e9*1000:.3f} GB per 1k tokens  |  "
          f"{bpt/1e9*100000:.1f} GB for a 100k-token history")
    print("TEXT:     UTF-8 bytes/token measured per row below (no assumed constant)\n")

    steps = build_trajectory(args.n_steps)
    header = head + "You are reviewing a completed agent trajectory. Answer precisely.\n\nTrajectory:\n"

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    full_cache, spans, total = llm.prefill_full(steps, header)
    torch.cuda.synchronize()
    t_write_full = time.perf_counter() - t0
    header_len = spans[0][0]
    print(f"prefilled {len(steps)} steps = {total} tokens in {t_write_full:.2f}s "
          f"(one-time write for regime B)\n")

    q = "\n\nQuestion: summarize what happened.\nAnswer concisely:" + tail
    qids = llm._ids(q)

    print(f"{'sel_tok':>8} {'kv_MB':>8} {'txt_KB':>8} {'txtB/tok':>9} {'ratio':>8} {'ttft_txt':>9} "
          f"{'ttft_kv':>8} {'gather':>8} {'spdup':>8} {'spdup-g':>9} {'N*':>8} {'N*-g':>8}")
    rows = []
    for tgt in [int(x) for x in args.targets.split(",")]:
        # prefill_full puts the header in [0, header_len) and does NOT give it a span, so
        # spans[i] IS steps[i] -- index them together, with no offset and nothing skipped.
        kept, acc = [], 0
        for i, (s, e) in enumerate(spans):
            kept.append(i)
            acc += (e - s)
            if acc >= tgt:
                break
        if acc < tgt * 0.8:
            print(f"{acc:8d}  (trajectory too short for {tgt}; skip)")
            continue

        sel_text = "".join(steps[i] for i in kept)
        txt_bytes = len(sel_text.encode("utf-8"))
        kv_bytes = bpt * acc

        # The gather is an index_select over the FULL cache per layer, so its cost tracks the whole
        # trajectory (95k here), not the selected slice. That is this prototype's copy overhead, not
        # an intrinsic cost of the KV path -- a paged-attention server would not copy at all. Time
        # the two phases separately so the copy can be priced out rather than silently charged.
        gather_t = []                                     # one entry per timed() call incl. warmup

        def read_kv():
            # TTFT only: _greedy_pos RETURNS the time to the first token, and wall-clocking the
            # call instead would charge the KV arm for `--decode` extra decode steps that the text
            # arm (decode=1) never pays -- an apples-to-oranges gap of ~175 ms at 8 tokens, which is
            # most of the apparent KV penalty at small L. Gather is real pre-first-token work, so it
            # is added; it is reported separately so the copy can also be priced out.
            torch.cuda.synchronize()
            t = time.perf_counter()
            sub, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
            torch.cuda.synchronize()
            g = time.perf_counter() - t
            gather_t.append(g)
            _, ttft, _ = llm._greedy_pos(sub, kpos, qids, args.decode)
            del sub
            torch.cuda.empty_cache()
            return g + ttft

        cids = build_full_ids(llm, header, steps, kept, q)

        def read_text():
            _, ttft, _ = llm._greedy(DynamicCache(), 0, cids, 1)
            return ttft

        t_kv = timed(read_kv, args.reps)
        t_gather = st.median(gather_t[1:]) if len(gather_t) > 1 else gather_t[0]
        t_txt = timed(read_text, args.reps)
        gain = t_txt - t_kv
        nstar = (t_write_full / gain) if gain > 0 else float("inf")
        gain_ng = t_txt - (t_kv - t_gather)               # copy priced out
        nstar_ng = (t_write_full / gain_ng) if gain_ng > 0 else float("inf")

        rows.append({"sel_tok": acc, "kv_bytes": kv_bytes, "txt_bytes": txt_bytes,
                     "txt_bytes_per_token": txt_bytes / acc,
                     "ratio": kv_bytes / txt_bytes, "ttft_text_s": t_txt, "ttft_kv_s": t_kv,
                     "t_gather_s": t_gather, "ttft_kv_nogather_s": t_kv - t_gather,
                     "speedup": (t_txt / t_kv) if t_kv > 0 else 0.0,
                     "speedup_nogather": (t_txt / (t_kv - t_gather)) if t_kv > t_gather else 0.0,
                     "t_write_full_s": t_write_full, "n_star_regimeB": nstar,
                     "n_star_regimeB_nogather": nstar_ng})
        sp_ng = (t_txt / (t_kv - t_gather)) if t_kv > t_gather else 0.0
        print(f"{acc:8d} {kv_bytes/1e6:8.1f} {txt_bytes/1e3:8.1f} {txt_bytes/acc:9.2f} "
              f"{kv_bytes/txt_bytes:7.0f}x {t_txt*1000:9.0f} {t_kv*1000:8.0f} "
              f"{t_gather*1000:8.0f} {(t_txt/t_kv if t_kv else 0):7.1f}x {sp_ng:8.1f}x "
              f"{nstar:8.1f} {nstar_ng:8.1f}")

    print("\nregime A (cache is a by-product of running the agent): marginal write = 0, so the")
    print("break-even is the FIRST query -- the whole cost is the storage ratio above.")
    print("regime B (memory built offline): N* queries against the same memory before it pays off.")
    json.dump({"args": vars(args), "bytes_per_token_kv": bpt,
               "t_write_full_s": t_write_full, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("ECON_DONE", flush=True)


if __name__ == "__main__":
    main()
