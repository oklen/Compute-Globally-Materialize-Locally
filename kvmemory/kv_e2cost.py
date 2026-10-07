"""kv_e2cost.py -- the cost axis (the latency microbenchmark of the real-dialog appendix).

Everything E2 reports so far is quality.  The deployable claim is not "harv is as good as self",
it is "harv is as good as self AND costs less at query time" -- and the E2 write-up says outright that this
needs MEASURED latency, not an estimate.  ([[gpu-shard-count-must-be-measured]]: on this project
an estimate was off by 2.9x.)

What is timed, per served set (the arms are constructed by the SAME code paths kv_e2hot2 uses --
a cost number for a different construction than the one we measured quality on would be useless):

    tx     re-prefill the served rows as TEXT                     the deployed text-memory path
    iso    encode each contiguous run alone, then assemble        read-time compute, no sharing
    self   encode all served rows in one block, then assemble     read-time compute, shared
    harv   index_select from the full-conversation prefill        NO forward pass at all
    bhot   harv for non-hot rows + prefill_fresh(hot tail)        what production runs

Two numbers per arm:
    T_build   producing (cache, positions) for the served rows -- this is where arms differ
    TTFT      T_build + question prefill + first generated token -- what a user actually waits for

and one number per conversation:
    T_write   the single full-conversation prefill that harv/bhot gather FROM.  This is a WRITE
              -side cost paid once per conversation, not per query, so it is reported separately
              and amortised; quoting harv as "free" without it would be dishonest.

Timing hygiene: warmup discarded, cuda.synchronize() on both sides of every region, MEDIAN of
`--reps` repeats (latency is right-skewed, the mean reports the worst outlier).  Must be run as a
SINGLE process on an otherwise idle GPU -- three shards sharing one card makes every number
meaningless.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa SPRAG_ENCODE_CHUNK=4096 \\
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_e2cost \\
        --sets ./out/e2_sets.json --limit 120 --out ./out/e2cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.locomo import load_locomo
from kvmemory.rA_roles import _utterance

INSTR = "Answer with as few words as possible (a name, a date, a short phrase)."


@torch.no_grad()
def prefill_fresh(llm, cache, cache_len, ids, positions):
    """Verbatim from kv_floor.py:50 -- same computation as the deployed b_hot."""
    dev = llm.device
    core = getattr(llm.model, "model", llm.model)
    core(input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
         past_key_values=cache, use_cache=True,
         position_ids=torch.tensor(positions, dtype=torch.long, device=dev).unsqueeze(0),
         cache_position=torch.arange(cache_len, cache_len + len(ids), device=dev),
         attention_mask=torch.ones(1, cache_len + len(ids), dtype=torch.long, device=dev))
    return cache


def timed(fn, reps, warmup=1):
    """median wall-clock of fn, with the GPU actually flushed on both sides."""
    for _ in range(warmup):
        r = fn()
        del r
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        del r
    return st.median(ts) * 1000.0          # ms


def runs_of(units):
    out, cur = [], [units[0]]
    for i in units[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hots", default="0,4,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    HOTS = [int(x) for x in args.hots.split(",") if x.strip()]
    S = json.load(open(args.sets))
    items = S["items"]
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    HPOS = list(range(H))
    eps = {str(e.conv_id): e for e in load_locomo(args.data)}

    by_conv = defaultdict(list)
    for it in items:
        by_conv[it["conv"]].append(it)

    rows, wrows = [], []
    done = 0
    for cid in sorted(by_conv, key=int):
        if done >= args.limit:
            break
        ep = eps[cid]
        turns = [_utterance(s) for s in ep.segments]
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids) for t in turns]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        NT = len(turns)

        # ---- WRITE SIDE: the one full-conversation prefill harv/bhot gather from ----
        tw = timed(lambda: encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H),
                   reps=max(2, args.reps // 2), warmup=1)
        full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
        wrows.append({"conv": cid, "turns": NT, "tokens": cur - H, "t_write_ms": tw})
        print("[cost] conv %s  turns=%d  tok=%d  WRITE full prefill %.1f ms (%.3f ms/1k tok)"
              % (cid, NT, cur - H, tw, tw / max(cur - H, 1) * 1000), flush=True)

        for it in by_conv[cid]:
            if done >= args.limit:
                break
            qt = f"\n\nQuestion: {it['question']}\n{INSTR}" + tail
            qids = llm._ids(qt)
            srv = sorted(it["serve"])[: args.k]
            for h in HOTS:
                hot_u = list(range(max(0, NT - h), NT))
                units = sorted(set(srv) | set(hot_u))
                hotset = set(hot_u)
                crows = [pp for i in units for pp in range(*spans[i])]
                ntok = len(crows)
                rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                text = "".join(turns[i] + "\n" for i in units)

                def b_iso():
                    parts = []
                    for run in runs_of(units):
                        a, b = spans[run[0]][0], spans[run[-1]][1]
                        parts.append((encode_block(llm, [t for i in run for t in eids[i]],
                                                   list(range(a, b))), list(range(a, b))))
                    return assemble(llm, [(H_kv, HPOS)] + parts)

                def b_self():
                    kv = encode_block(llm, [t for i in units for t in eids[i]], crows)
                    return assemble(llm, [(H_kv, HPOS), (kv, crows)])

                def b_harv():
                    sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
                    return assemble(llm, [(H_kv, HPOS), (sub, crows)])

                def b_bhot():
                    served = [i for i in units if i not in hotset]
                    blocks = [(H_kv, HPOS)]
                    if served:
                        cr = [pp for i in served for pp in range(*spans[i])]
                        r2 = torch.tensor([pp - H for pp in cr], dtype=torch.long)
                        blocks.append(([(K.index_select(2, r2), V.index_select(2, r2))
                                        for K, V in full_kv], cr))
                    c, p = assemble(llm, blocks)
                    hu = [i for i in units if i in hotset]
                    if hu:
                        hid = [t for i in hu for t in eids[i]]
                        hpos = [p2 for i in hu for p2 in range(*spans[i])]
                        c = prefill_fresh(llm, c, p.shape[0], hid, hpos)
                        p = torch.cat([p, torch.tensor(hpos, dtype=torch.long, device=llm.device)])
                    return c, p

                rec = {"conv": cid, "qi": it["qi"], "hot": h, "k": args.k,
                       "n_units": len(units), "n_ctx_tok": ntok,
                       "n_text_tok": len(llm._ids(text)), "n_q_tok": len(qids)}
                for name, fn in (("iso", b_iso), ("self", b_self),
                                 ("harv", b_harv), ("bhot", b_bhot)):
                    rec["build_%s_ms" % name] = timed(fn, args.reps)
                    # TTFT deliberately REBUILDS the cache each rep: what a user waits for is
                    # build + question prefill + first token, not the decode alone.
                    rec["ttft_%s_ms" % name] = timed(
                        lambda f=fn: llm._greedy_pos(*f(), qids, 1), args.reps)
                rec["build_tx_ms"] = float("nan")   # tx has no separable build step
                rec["ttft_tx_ms"] = timed(
                    lambda: llm._greedy(DynamicCache(), 0, llm._ids(head + text + qt), 1),
                    args.reps)
                rows.append(rec)
                torch.cuda.empty_cache()
            done += 1
            print("[cost] %s q%s | " % (cid, it["qi"])
                  + " ".join("h%d(%dtok) %s" % (
                      r["hot"], r["n_ctx_tok"],
                      " ".join("%s %.0f/%.0f" % (m, r["build_%s_ms" % m], r["ttft_%s_ms" % m])
                               for m in ("iso", "self", "harv", "bhot")))
                      + " tx -/%.0f" % r["ttft_tx_ms"]
                      for r in rows[-len(HOTS):]), flush=True)
        del full_kv
        torch.cuda.empty_cache()

    json.dump({"sets": args.sets, "k": args.k, "hots": HOTS, "reps": args.reps,
               "rows": rows, "write": wrows}, open(args.out, "w"), indent=1)

    print("\n===== COST (n=%d items, %d conversations, median of %d reps) ====="
          % (done, len(wrows), args.reps))
    print("\n-- WRITE side (once per conversation, amortised over its queries) --")
    print("  full-conversation prefill: median %.0f ms over %d convs (%.2f ms / 1k tokens)"
          % (st.median([w["t_write_ms"] for w in wrows]), len(wrows),
             st.median([w["t_write_ms"] / max(w["tokens"], 1) * 1000 for w in wrows])))
    for h in HOTS:
        sub = [r for r in rows if r["hot"] == h]
        if not sub:
            continue
        print("\n-- READ side, hot=%d  (%d items, %d ctx tokens median) --"
              % (h, len(sub), int(st.median([r["n_ctx_tok"] for r in sub]))))
        print("  %-6s %12s %12s   %s" % ("arm", "T_build ms", "TTFT ms", "vs tx"))
        tx = st.median([r["ttft_tx_ms"] for r in sub])
        for m in ("tx", "iso", "self", "harv", "bhot"):
            b = [r["build_%s_ms" % m] for r in sub if r["build_%s_ms" % m] == r["build_%s_ms" % m]]
            t = st.median([r["ttft_%s_ms" % m] for r in sub])
            print("  %-6s %12s %12.1f   %5.2fx"
                  % (m, ("%.1f" % st.median(b)) if b else "-", t, tx / t))
    print("E2COST_DONE", flush=True)


if __name__ == "__main__":
    main()
