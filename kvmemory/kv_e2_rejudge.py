"""Cross-family re-judge of the E2 answers.

The pre-registered plan keeps this as the standing insurance on the primary estimand: kv_e2 judges with the
SAME local model it generated with (kvmemory.cross_judge.judge(llm, ...)), so a positive
harv - iso could in principle be judge variance rather than a real difference.  Every answer is
saved for exactly this reason (`ans_none`, `ans_text`, `ans_harv`, `ans_iso`, `ans_p_harv`,
`ans_p_iso`), so re-scoring costs API calls and no GPU.

The re-judge runs on a different model family from the Qwen3-8B reader (the reported run used a
commercial chat model with reasoning disabled) through an OpenAI-compatible LLM endpoint (JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL, see kvmemory/llm_api.py).

What matters is NOT that the two judges agree on absolute accuracy -- they will not, and that is
uninteresting.  What matters is whether the PAIRED DIFFERENCE survives.  A judge that is uniformly
stricter shifts both arms equally and cancels in harv - iso; only a judge that is differentially
strict between the two arms can move it.  So this prints both judges' deltas side by side, and
their per-item disagreement rate.

Reasoning MUST stay disabled on servers that have it (JUDGE_EXTRA_BODY, see llm_api): with it on, the
reported endpoint spent 80-90s reasoning and returned the trace instead of the verdict, which reads
as a network timeout rather than as an error.

    JUDGE_BASE_URL=https://your-endpoint/v1 JUDGE_API_KEY=... JUDGE_MODEL=your-model \\
    JUDGE_EXTRA_BODY='{"thinking": {"type": "disabled"}}' PYTHONPATH=. \\
        python -m kvmemory.kv_e2_rejudge --res ./out/e2_full.json --sets ./out/e2_sets.json \\
        --out ./out/e2_rejudged.json
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, hashlib, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_api import chat as _llm_chat

ARMS = ("none", "text", "harv", "iso", "p_harv", "p_iso")

PROMPT = """You are grading one short answer against the reference answer.

Question: {q}
Reference answer: {gold}
Candidate answer: {ans}

The candidate is CORRECT if it states the same fact as the reference, allowing paraphrase,
different formatting of dates/numbers, and extra harmless detail. It is INCORRECT if it states a
different fact, refuses, says it does not know, or is empty.

Reply with exactly one word: CORRECT or INCORRECT."""


def chat(prompt, timeout=120):
    """One chat call to the configured OpenAI-compatible endpoint (kvmemory.llm_api)."""
    return _llm_chat(prompt, timeout=timeout)


def verdict(q, gold, ans):
    if not (ans or "").strip():
        return 0                      # empty is incorrect without spending a call
    for attempt in range(5):
        try:
            t = chat(PROMPT.format(q=q, gold=gold, ans=ans)).strip().upper()
            if "INCORRECT" in t: return 0
            if "CORRECT" in t:   return 1
        except Exception:
            pass
        time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    return None                        # unresolved -- excluded, never silently counted as wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="./out/e2_full.json")
    ap.add_argument("--out", default="./out/e2_rejudged.json")
    ap.add_argument("--cache", default="./out/e2_rejudge_cache.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sets", default="", help="e2_sets.json; the question text lives here, not in the rows")
    a = ap.parse_args()

    R = json.load(open(a.res)); rows = R["rows"]

    # kv_e2 rows carry conv/qi/gold but NOT the question text -- it lives in e2_sets.json and has
    # to be joined on (conv, qi). Caught by a smoke test before this ever ran for real; without
    # it the whole re-judge would have died with KeyError: 'question' the moment E2 finished,
    # hours after the fact and with nobody watching.
    sets_path = a.sets or R.get("sets") or "./out/e2_sets.json"
    S = json.load(open(sets_path))
    qmap = {(str(i["conv"]), i["qi"]): i["question"] for i in S["items"]}
    missing = [(r["conv"], r["qi"]) for r in rows if (str(r["conv"]), r["qi"]) not in qmap]
    assert not missing, ("%d rows have no question in %s (e.g. %s) -- refusing to judge without "
                         "the question rather than silently grading on gold alone"
                         % (len(missing), sets_path, missing[:3]))
    print("joined %d rows to questions from %s" % (len(rows), sets_path))

    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    jobs = []
    for r in rows:
        q = qmap[(str(r["conv"]), r["qi"])]
        for arm in ARMS:
            ans = r.get("ans_%s" % arm)
            k = hashlib.sha1(("%s|%s|%s" % (q, r["gold"], ans)).encode()).hexdigest()
            if k not in cache: jobs.append((k, q, r["gold"], ans))
    print("re-judging %d answers (%d already cached)" % (len(jobs), len(rows) * len(ARMS) - len(jobs)))

    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(verdict, q, g, s): k for k, q, g, s in jobs}
        for f in cf.as_completed(futs):
            cache[futs[f]] = f.result()
            done += 1
            if done % 200 == 0:
                json.dump(cache, open(a.cache, "w"))
                print("  %d/%d" % (done, len(jobs)), flush=True)
    json.dump(cache, open(a.cache, "w"))

    unresolved = 0
    for r in rows:
        q = qmap[(str(r["conv"]), r["qi"])]
        for arm in ARMS:
            k = hashlib.sha1(("%s|%s|%s" % (q, r["gold"], r.get("ans_%s" % arm))).encode()).hexdigest()
            v = cache.get(k)
            if v is None: unresolved += 1
            r["j2_%s" % arm] = v
    json.dump({"sets": R.get("sets"), "arm": R.get("arm"), "k": R.get("k"), "rows": rows},
              open(a.out, "w"), indent=1)
    print("wrote %s  (unresolved verdicts: %d)" % (a.out, unresolved))

    # --- does the paired difference survive the judge swap?
    import numpy as np
    from kvmemory import kv_e2stats as K
    rng = np.random.default_rng(1234)
    for tag, jp in (("original judge (Qwen3-8B, same family as the reader)", "j"),
                    ("cross-family judge", "j2")):
        by = {}
        for r in rows:
            h, i = r.get("%s_harv" % jp), r.get("%s_iso" % jp)
            if h is None or i is None: continue
            by.setdefault(str(r["conv"]), []).append(int(h) - int(i))
        if by: K.report("PRIMARY harv - iso -- %s" % tag, by, rng)
    agree = tot = 0
    for r in rows:
        for arm in ARMS:
            x, y = r.get("j_%s" % arm), r.get("j2_%s" % arm)
            if x is None or y is None: continue
            tot += 1; agree += int(int(x) == int(y))
    if tot:
        print("\n  judge agreement %d/%d = %.3f  (absolute agreement is NOT the test; the test is "
              "whether harv - iso above is unchanged)" % (agree, tot, agree / tot))


if __name__ == "__main__":
    main()
