"""rA_loose.py -- E1's loose role labels: the LLM channel the strict pass never had.

WHY THIS IS NOT OPTIONAL.  rA_roles labels a turn V only if a normalised form of the gold answer
appears in it, and M only if the turn shares >= 2 content words with the question.  Both rules are
purely lexical, and both err in the SAME direction for pi:

  * a turn that states the answer in PARAPHRASE is scored U, so it never enters min-rank over V;
    601 questions have a V set that does not even contain the official gold evidence turn.
  * a turn that is plainly about the question's topic but happens to share one content word is
    scored U, so it never enters min-rank over M.

The first inflates pi (the value row we are supposed to find is invisible), the second deflates it.
The plan therefore asks for pi as an INTERVAL -- strict as one endpoint, an LLM-judged loose pass
as the other -- and says so explicitly: this quantity is definition-sensitive and a reader has to
see how much.  Reporting only the strict number is reporting one end of an interval as a point.

SCOPE.  Judging every turn of every conversation for every question is 353k (question, turn) pairs.
We judge a deterministic candidate set per question instead -- the union of

    M_strict                      the >= 2-shared-word screen
    V[N1]                         everything the lexical ladder already calls value-bearing
    gold_idx                      the official evidence turns
    top-20 remaining by overlap with the QUESTION      (catches the 1-shared-word misses)
    top-20 remaining by overlap with the GOLD ANSWER   (catches paraphrase with low question overlap)

capped at 80 per question.  pi is decided by the BEST-ranked member of each role, so a turn no
retriever ever surfaces cannot change it; what matters is that plausible candidates are judged.
The cap and the two top-20 lists are part of the definition and are reported with the number.

The judge sees the question, the gold answer and the turns.  It does NOT see the strict labels, so
it cannot simply ratify them.

    python -m kvmemory.rA_loose \\
        --labels ./out/rA_labels.json --data ./data/locomo10.json \\
        --workers 8 --out ./out/rA_loose.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_api import chat as _llm_chat
from kvmemory.locomo import load_locomo
from kvmemory.rA_roles import _utterance, toks

BATCH = 20
CAP = 80

PROMPT = """You are labelling turns of a long personal conversation for one question.

Question: {q}
The correct answer to that question is: {gold}

Label each numbered turn below with exactly one of:
  V  the turn STATES the answer -- it asserts the value, in any wording, including a paraphrase,
     a synonym, an equivalent date/number format, or a partial form that a reader would accept as
     the answer to this question.
  M  the turn is ABOUT this question's topic, people or events -- it would look relevant to
     somebody searching for the answer -- but it does NOT state the answer.
  U  neither: the turn is about something else.

Judge the turn on its own words. Do not label a turn V merely because it is near one that is.

{turns}

Reply with a JSON object mapping each turn number to "V", "M" or "U", and nothing else."""


def chat(prompt, timeout=120):
    """One chat call to the configured OpenAI-compatible endpoint (kvmemory.llm_api)."""
    return _llm_chat(prompt, timeout=timeout)


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    out = {}
    for k, v in d.items():
        try:
            i = int(str(k).strip())
        except ValueError:
            continue
        v = str(v).strip().upper()[:1]
        if v in ("V", "M", "U"):
            out[i] = v
    return out


def candidates(rec, texts, tok_sets):
    qt = set(toks(rec["question"]))
    gt = set(toks(rec["gold"]))
    base = set(rec["M_strict"]) | set(rec["V"]["N1"]) | set(rec["gold_idx"])
    rest = [i for i in range(len(texts)) if i not in base]
    by_q = sorted(rest, key=lambda i: (-len(qt & tok_sets[i]), i))
    by_q = [i for i in by_q if qt & tok_sets[i]][:20]
    rest2 = [i for i in rest if i not in set(by_q)]
    by_g = sorted(rest2, key=lambda i: (-len(gt & tok_sets[i]), i))
    by_g = [i for i in by_g if gt & tok_sets[i]][:20]
    return sorted(base | set(by_q) | set(by_g))[:CAP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=4,
                    help="The reported endpoint tolerated short bursts at 16 but not 4k sustained calls: the first "
                         "full pass at 8 lost 80 percent of its batches. The graph ingest shares "
                         "the same gateway, so keep this low.")
    ap.add_argument("--cache", default="",
                    help="JSONL of finished batches; written as we go and re-read on restart, so "
                         "a throttled run resumes instead of paying for the whole pass again")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(v, None)

    recs = json.load(open(args.labels))["records"]
    if args.limit:
        recs = recs[: args.limit]
    eps = {str(e.conv_id): e for e in load_locomo(args.data)}
    cache = {}
    for cid, ep in eps.items():
        t = [_utterance(s) for s in ep.segments]
        cache[cid] = (t, [set(toks(x)) for x in t])

    jobs = []
    for ri, rec in enumerate(recs):
        texts, tok_sets = cache[str(rec["conv"])]
        cand = candidates(rec, texts, tok_sets)
        for b in range(0, len(cand), BATCH):
            jobs.append((ri, cand[b:b + BATCH]))
    print(f"[loose] {len(recs)} questions -> {len(jobs)} batched judge calls "
          f"({sum(len(j[1]) for j in jobs)} turn judgements)", flush=True)

    # resume: a batch is identified by (record index, first turn index in the batch)
    cpath = args.cache or (args.out + ".jsonl")
    seen = {}
    if os.path.exists(cpath):
        for line in open(cpath):
            try:
                d = json.loads(line)
            except Exception:
                continue
            seen[(d["ri"], d["k"])] = {int(k): v for k, v in d["lab"].items()}
        print(f"[loose] resuming: {len(seen)} batches already cached", flush=True)
    todo = [j for j in jobs if (j[0], j[1][0]) not in seen]
    print(f"[loose] {len(todo)} batches to fetch", flush=True)

    errs = Counter()
    lk = threading.Lock()
    cf_out = open(cpath, "a")

    def run(job):
        ri, idxs = job
        rec = recs[ri]
        texts, _ = cache[str(rec["conv"])]
        block = "\n".join(f"[{i}] {texts[i]}" for i in idxs)
        prompt = PROMPT.format(q=rec["question"], gold=rec["gold"], turns=block)
        for attempt in range(5):
            try:
                got = parse(chat(prompt))
                if got:
                    lab = {i: got[i] for i in idxs if i in got}
                    with lk:
                        cf_out.write(json.dumps({"ri": ri, "k": idxs[0], "lab": lab}) + "\n")
                        cf_out.flush()
                    return ri, lab
                errs["empty_or_unparseable"] += 1
            except Exception as e:
                # count the failure TYPE.  The first pass swallowed these and reported a result
                # that looked like "the judge found almost nothing" instead of "80 percent of the
                # calls never landed".
                errs[type(e).__name__ + ":" + str(e)[:60]] += 1
            time.sleep(min(30, 1.5 * (3 ** attempt)) * (0.5 + random.random()))
        return ri, {}

    out = {ri: {} for ri in range(len(recs))}
    for (ri, k), lab in seen.items():
        out[ri].update(lab)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ri, lab in ex.map(run, todo):
            out[ri].update(lab)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} batches | errors so far: {sum(errs.values())}",
                      flush=True)
    cf_out.close()
    if errs:
        print("  failure histogram:")
        for k, v in errs.most_common(8):
            print(f"    {v:6d}  {k}")

    res, stats = [], Counter()
    for ri, rec in enumerate(recs):
        lab = out[ri]
        V_loose = sorted(i for i, v in lab.items() if v == "V")
        M_loose = sorted(i for i, v in lab.items() if v == "M")
        vs, ms = set(rec["V"]["N1"]), set(rec["M_strict"])
        stats["judged"] += len(lab)
        stats["V_new"] += len(set(V_loose) - vs)
        stats["V_lost"] += len(vs - set(V_loose) - set(M_loose) - {i for i in vs if i not in lab})
        stats["M_new"] += len(set(M_loose) - ms)
        stats["unjudged_rec"] += int(not lab)
        res.append({"conv": rec["conv"], "qi": rec["qi"], "cat": rec["cat"],
                    "n_judged": len(lab), "V_loose": V_loose, "M_loose": M_loose,
                    "nV_strict": len(vs), "nM_strict": len(ms),
                    "gold_idx_labelled_V": sorted(set(rec["gold_idx"]) & set(V_loose))})
    json.dump({"model": MODEL, "cap": CAP, "batch": BATCH, "records": res},
              open(args.out, "w"), indent=1)
    n = len(res)
    print(f"\n===== loose labels (n={n} questions) =====")
    print(f"  turn judgements: {stats['judged']}   questions with no usable judgement: "
          f"{stats['unjudged_rec']}")
    print(f"  V rows the strict ladder missed: {stats['V_new']}   "
          f"M rows it missed: {stats['M_new']}")
    gs = sum(1 for r in res if r["gold_idx_labelled_V"])
    print(f"  questions whose OFFICIAL gold evidence is now labelled V: {gs}/{n}")
    print(f"  mean |V| strict {sum(r['nV_strict'] for r in res)/n:.2f} -> loose "
          f"{sum(len(r['V_loose']) for r in res)/n:.2f} ; "
          f"|M| strict {sum(r['nM_strict'] for r in res)/n:.2f} -> loose "
          f"{sum(len(r['M_loose']) for r in res)/n:.2f}")
    print("RA_LOOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
