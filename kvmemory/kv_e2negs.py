"""kv_e2negs.py -- hard negatives for E2's forced-choice readout.

Why E2 needs a second readout at all.  The pre-registered PRIMARY is harv - iso on judged accuracy,
a paired BINARY outcome.  Measured power on the real cluster structure (10 conversations, sizes
23..78, n=601) says that reaches 80% only around delta ~= .05-.07; at delta = .02 it is 25-36%.
E3 put the path's effect at roughly +.02 of probability mass with 4/111 items flipping, so a
binary readout is likely to sit below its own resolution and return an uninformative null.

A forced choice reads the probability mass instead of a coin flip, so the same items carry far more
information per item.  It needs a confusable negative per question -- generated the same way as
E3's (kv_e3negs), by a model that sees only the question and the gold, never the conversation, the
serve set or which arm a row came from.

    python -m kvmemory.kv_e2negs \\
        --sets ./out/e2_sets.json --workers 4 --out ./out/e2_negs.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.kv_e3negs import chat, parse, qkey, PROMPT, N_CAND


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(v, None)

    items = json.load(open(args.sets))["items"]
    from kvmemory.locomo import load_locomo
    from kvmemory.rA_roles import _utterance
    conv_low = {str(e.conv_id): "\n".join(_utterance(s) for s in e.segments).lower()
                for e in load_locomo(args.data)}

    cpath = args.out + ".jsonl"
    seen = {}
    if os.path.exists(cpath):
        for line in open(cpath):
            try:
                d = json.loads(line)
            except Exception:
                continue
            seen[d["k"]] = d["negs"]
        print(f"[e2negs] resuming: {len(seen)} cached", flush=True)
    todo = [it for it in items if qkey(it["conv"], str(it["qi"]), it["question"]) not in seen]
    print(f"[e2negs] {len(items)} items, {len(todo)} to fetch", flush=True)

    errs = Counter()
    lk = threading.Lock()
    fh = open(cpath, "a")

    def run(it):
        k = qkey(it["conv"], str(it["qi"]), it["question"])
        gold = it["gold"].strip()
        gl = gold.lower()
        low = conv_low[str(it["conv"])]
        for attempt in range(5):
            try:
                cands = parse(chat(PROMPT.format(q=it["question"], gold=gold, n=N_CAND)))
                keep = []
                for c in cands:
                    cl = c.lower().strip()
                    if len(cl) < 2 or cl == gl or cl in gl or gl in cl or cl in low:
                        continue
                    if cl in {x.lower() for x in keep}:
                        continue
                    keep.append(c.strip())
                if len(keep) >= 2:
                    with lk:
                        fh.write(json.dumps({"k": k, "negs": keep}) + "\n")
                        fh.flush()
                    return k, keep
                errs["too_few_after_screens"] += 1
            except Exception as e:
                errs[type(e).__name__ + ":" + str(e)[:50]] += 1
            time.sleep(min(30, 1.5 * (3 ** attempt)) * (0.5 + random.random()))
        return k, []

    res = dict(seen)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, v in ex.map(run, todo):
            res[k] = v
    fh.close()
    ok = sum(1 for v in res.values() if len(v) >= 2)
    json.dump({"negs": res}, open(args.out, "w"), indent=1)
    print(f"[e2negs] wrote {args.out}: {len(res)} items, {ok} with >=2 candidates")
    if errs:
        for k, v in errs.most_common(6):
            print(f"    {v:5d}  {k}")
    print("E2NEGS_DONE", flush=True)


if __name__ == "__main__":
    main()
