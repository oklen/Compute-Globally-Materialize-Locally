"""kv_hnegs.py -- LLM-written hard negatives for the HotpotQA bridge items (CPU + an OpenAI-compatible LLM endpoint (JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL, see kvmemory/llm_api.py)).

kv_cite / kv_ans / kv_region first borrowed negatives from other items' answers of the same
shape class.  The smoke showed exactly E3's defect C1: type-mismatched negatives ('137th' vs
'middleweight') leave the no-context prior at +35..+55 log-odds, so the served passages never
enter the decision.  Same fix as kv_e3negs: write the candidates per item -- same type, same
length, plausible without the passages -- from the question and the gold ONLY (the generator never
sees the passages).  The consumer still picks WHICH candidate by the no-context probe.

Also writes same-kind stand-ins for the BRIDGE entity (a person/org/work name), used by the
bridge test (d_B: b -> X) and the placebo.

    python -m kvmemory.kv_hnegs --limit 800 --out ./out/hnegs.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.kv_e3negs import chat as _chat, parse


def chat(prompt, timeout=120):
    """kv_e3negs.chat + exponential backoff on HTTP 429 (12 workers tripped the rate limit)."""
    import urllib.error
    for attempt in range(6):
        try:
            return _chat(prompt, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                time.sleep(2 ** attempt + random.random())
                continue
            raise
from kvmemory.kv_cite import load_items

P_ANS = """You are building a two-choice reading-comprehension test.
Question: {q}
True answer: {gold}
Write {n} WRONG answers to that question. Each one must be:
- the same kind of thing as the true answer (a person for a person, a year for a year, a place for a place, a title for a title, a number for a number), and about the same length;
- plausible to a reader who has NOT seen the source passages -- they must have no way to tell it from the true answer;
- NOT a paraphrase, synonym, abbreviation, translation or superset/subset of the true answer, and not absurd or comic.
Reply with a JSON array of {n} strings and nothing else."""

P_BRIDGE = """Here is the name of an entity as it appears as a Wikipedia article title: "{b}"
Write {n} OTHER names of the same kind (same type of entity, same length and style, realistic, not famous enough to be instantly recognised) that could replace it in a sentence without looking odd.
Reply with a JSON array of {n} strings and nothing else."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/hotpotqa/hotpot_dev_distractor.jsonl")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = load_items(args.data)
    random.Random(args.seed).shuffle(items)            # the SAME shuffle every consumer uses
    items = items[: args.limit]
    done = {}
    if os.path.exists(args.out):
        done = json.load(open(args.out)).get("negs", {})
        print("[hnegs] resume: %d done" % len(done), flush=True)

    def work(it):
        if it["id"] in done and done[it["id"]].get("ans"):
            return it["id"], done[it["id"]]      # only skip items ALREADY filled (not empty 429s)
        rec = {"v": it["v"], "b": it["b"], "ans": [], "bridge": []}
        for attempt in range(3):
            try:
                rec["ans"] = [x for x in parse(chat(P_ANS.format(q=it["q"], gold=it["v"], n=args.n)))
                              if x.lower() != it["v"].lower()][: args.n]
                if rec["ans"]:
                    break
            except Exception as e:
                rec["err_ans"] = str(e)[:120]
        for attempt in range(3):
            try:
                rec["bridge"] = [x for x in parse(chat(P_BRIDGE.format(b=it["b"], n=4))) if x.lower() != it["b"].lower()][:4]
                if rec["bridge"]:
                    break
            except Exception as e:
                rec["err_bridge"] = str(e)[:120]
        return it["id"], rec

    res = dict(done)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, (iid, rec) in enumerate(ex.map(work, items), 1):
            res[iid] = rec
            if k % 25 == 0:
                json.dump({"negs": res}, open(args.out, "w"), indent=1)
                ok = sum(1 for r in res.values() if r.get("ans"))
                print("[hnegs] %d/%d  with answer negs %d" % (k, len(items), ok), flush=True)
    json.dump({"negs": res}, open(args.out, "w"), indent=1)
    ok = sum(1 for r in res.values() if r.get("ans")); okb = sum(1 for r in res.values() if r.get("bridge"))
    print("[hnegs] done: %d items, %d with answer negs, %d with bridge names" % (len(res), ok, okb))
    print("HNEGS_DONE", flush=True)


if __name__ == "__main__":
    main()
