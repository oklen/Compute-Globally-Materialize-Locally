"""kv_traj.py -- generate deep-research trajectories with REGION spans (paper section 9).

For each HotpotQA bridge item the agent (Qwen3-8B, greedy, no-think) searches its own 10-passage
pool with a BM25 tool and must write a NOTE after every result:

    SEARCH: <query>              tool_in
    RESULT: [title] text ...     tool_out   (inserted by us, BM25 top-2 from the pool)
    NOTE: <one line>             scratchpad
    ANSWER: <answer>             answer
plus the fixed system prompt (system), the question (user) and whatever the model writes before
its first SEARCH (plan).

The trajectory is the concatenation of generated ids and inserted ids, so every region is a token
span in ONE stream; re-prefilling that stream gives the in-situ KV the read-out modules harvest.
Nothing is measured here -- this module only writes trajectories + spans to JSON.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_traj --limit 400 --out ./out/traj.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_cite import load_items, wb
from kvmemory.kv_ans import bm25_top

SYSTEM = ("You are a research agent. You cannot see any passages until you search. To answer the "
          "question, issue searches one at a time on a line of the form `SEARCH: <query>`. After "
          "each result, write exactly one line `NOTE: <what you learned that matters for the "
          "question>`. When you know the answer, write `ANSWER: <short answer>` and stop. "
          "Use at most 4 searches. Think briefly in one line before the first search.")


@torch.no_grad()
def decode_until(llm, cache, cache_len, new_ids, max_new, stop_pats):
    """Greedy-decode on top of `cache`, feeding `new_ids` first (positions continue from cache_len).
    Stops when the decoded text (of THIS call) matches any stop pattern at a line end, or at EOS.
    Returns (generated_ids, text, stopped_by)."""
    dev = llm.device
    ids = torch.tensor([new_ids], dtype=torch.long, device=dev)
    L = len(new_ids)
    out = llm.model(input_ids=ids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(cache_len, cache_len + L, device=dev).unsqueeze(0),
                    cache_position=torch.arange(cache_len, cache_len + L, device=dev),
                    attention_mask=torch.ones(1, cache_len + L, dtype=torch.long, device=dev))
    cur = cache_len + L
    nxt = int(out.logits[0, -1].argmax())
    toks, text, why, unfed = [], "", "max", []
    for _ in range(max_new):
        if nxt in llm.eos_ids:
            why = "eos"
            break
        toks.append(nxt)
        text = llm.tok.decode(toks, skip_special_tokens=True)
        hit = next((name for name, pat in stop_pats if pat.search(text)), None)
        if hit:
            why, unfed = hit, [nxt]        # appended but NOT yet fed to the cache
            break
        out = llm.model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[cur]], device=dev),
                        cache_position=torch.tensor([cur], device=dev),
                        attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev))
        nxt = int(out.logits[0, -1].argmax())
        cur += 1
    return toks, text, why, unfed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/hotpotqa/hotpot_dev_distractor.jsonl")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--done_from", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    items = load_items(args.data)
    rnd.shuffle(items)
    items = items[: args.limit]
    DONE = set()
    if args.done_from:
        import glob
        for g in args.done_from.split(":"):
            for f in glob.glob(g):
                for line in open(f, errors="ignore"):
                    m = re.match(r"^\[traj\] (\S+) \|", line)
                    if m:
                        DONE.add(m.group(1))
        print("[traj] resume: %d done" % len(DONE), flush=True)

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    ids = lambda s: list(llm.tok(s, add_special_tokens=False).input_ids)
    STOP = [("search", re.compile(r"SEARCH:.*\n")), ("answer", re.compile(r"ANSWER:.*\n"))]
    STOP_NOTE = [("note", re.compile(r"NOTE:.*\n")), ("search", re.compile(r"SEARCH:.*\n")),
                 ("answer", re.compile(r"ANSWER:.*\n"))]

    from transformers import DynamicCache
    trajs, stats = [], Counter()
    for it in items:
        if it["id"] in DONE:
            continue
        q, v = it["q"], it["v"]
        pool = [(t, it["paras"][t]) for t in it["titles"]]
        # chat: system prompt + question in the user turn; the agent's transcript is the assistant turn
        user = SYSTEM + "\n\nQuestion: " + q
        stream, regions = [], []          # regions: (name, start, end, text)
        def add(name, tok_ids, text):
            s = len(stream); stream.extend(tok_ids); regions.append((name, s, len(stream), text))
        add("system", ids(head + SYSTEM), SYSTEM)
        add("user", ids("\n\nQuestion: " + q + tail), q)
        cache = DynamicCache()
        pending = list(stream)             # ids not yet fed to the cache
        fed = 0
        steps, answer, why = 0, None, None
        seen_titles = set()
        first = True
        while steps <= args.max_steps:
            toks, text, why, unfed = decode_until(llm, cache, fed, pending, 160, STOP if first else STOP_NOTE)
            fed += len(pending) + len(toks) - len(unfed)
            pending = list(unfed)
            if why == "eos" and "ANSWER:" in text:
                why = "answer"
            # split the generated text into plan / note / search / answer pieces
            if why == "answer":
                m = re.search(r"(.*?)(ANSWER:.*)$", text, re.S)
                pre, ans_line = (m.group(1), m.group(2)) if m else ("", text)
                pre_ids = ids(pre) if pre.strip() else []
                if pre.strip():
                    add("plan" if first else "scratchpad", pre_ids, pre.strip())
                ans_ids = toks[len(pre_ids):] if len(pre_ids) <= len(toks) else toks
                add("answer", ans_ids, ans_line.strip())
                answer = re.sub(r"^ANSWER:\s*", "", ans_line.strip())
                break
            if why == "search":
                m = re.search(r"(.*?)(SEARCH:.*\n)$", text, re.S)
                pre, s_line = m.group(1), m.group(2)
                # attribute generated tokens to pre (plan/note) and the SEARCH line by re-tokenising
                pre_ids = ids(pre) if pre.strip() else []
                if pre.strip():
                    add("plan" if first else "scratchpad", pre_ids, pre.strip())
                add("tool_in", toks[len(pre_ids):], s_line.strip())
                first = False
                query = re.sub(r"^SEARCH:\s*", "", s_line.strip())
                top = bm25_top(pool, query, 2)
                res = "RESULT: " + " ".join("[%s] %s" % (t, p) for t, p in top) + "\n"
                r_ids = ids(res)
                add("tool_out", r_ids, res.strip())
                seen_titles |= {t for t, _ in top}
                pending = pending + r_ids
                steps += 1
                continue
            if why == "note":
                add("scratchpad", toks, text.strip())
                first = False
                continue
            # eos / max without a marker
            if toks:
                add("scratchpad", toks, text.strip())
            break
        # sanity: the stream must equal the concatenation of region spans
        assert sum(e - s for _, s, e, _ in regions) == len(stream)
        # where does the gold first appear (verbatim) in the stream?  (source region)
        src = next(((n, s, e) for n, s, e, t in regions if n == "tool_out" and wb(v).search(t)), None)
        ok = answer is not None
        stats["done" if ok else "no_answer"] += 1
        stats["gold_in_tool_out" if src else "gold_never_retrieved"] += 1
        if ok and wb(v).search(answer):
            stats["answer_correct"] += 1
        trajs.append(dict(id=it["id"], q=q, v=v, tA=it["tA"], tB=it["tB"], b=it["b"], answer=answer,
                          steps=steps, why=why, n_tok=len(stream), stream=stream,
                          regions=[dict(name=n, s=s, e=e, text=t) for n, s, e, t in regions],
                          src=src, seen_titles=sorted(seen_titles),
                          note_states_gold=int(any(n == "scratchpad" and wb(v).search(t) for n, _, _, t in regions)),
                          answer_correct=int(bool(answer and wb(v).search(answer)))))
        print("[traj] %s | steps=%d tok=%d why=%s src=%s correct=%d note_states_gold=%d | ans=%r"
              % (it["id"], steps, len(stream), why, src[0] if src else None,
                 trajs[-1]["answer_correct"], trajs[-1]["note_states_gold"], (answer or "")[:50]), flush=True)
        del cache
        torch.cuda.empty_cache()
    json.dump({"data": args.data, "seed": args.seed, "system": SYSTEM, "trajs": trajs, "stats": dict(stats)},
              open(args.out, "w"))
    n = len(trajs)
    print("[traj] %d trajectories: %s" % (n, dict(stats)))
    if n:
        import statistics as st
        print("  tokens median %d  steps median %d  region token share: %s"
              % (st.median(t["n_tok"] for t in trajs), st.median(t["steps"] for t in trajs),
                 {k: round(v / sum(t["n_tok"] for t in trajs), 3) for k, v in
                  Counter({}).items()} or ""))
        share = Counter()
        for t in trajs:
            for r in t["regions"]:
                share[r["name"]] += r["e"] - r["s"]
        tot = sum(share.values())
        print("  region token share: %s" % {k: round(v / tot, 3) for k, v in sorted(share.items())})
    print("TRAJ_DONE", flush=True)


if __name__ == "__main__":
    main()
