"""kv_e3negs.py -- generate the hard-negative candidate pool for the E3 control (CPU + an OpenAI-compatible LLM endpoint (JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL, see kvmemory/llm_api.py)).

Why this exists.  The frozen control (kv_harvest4) paired the gold against another QA's gold from
the same conversation.  That negative is usually the wrong TYPE for the question -- ask where
somebody moved from, offer "joining activist group, going to pride parades, ..." -- so the model
answers from the question alone and the served rows never enter the decision.  That is defect C1.

Borrowing a negative from the dataset's own gold pool (kv_e3ctrl's fallback) fixes the type but not
the plausibility: with only a few hundred golds the best available confusable for "Turtles." is
"Seraphim", and the no-context prior stays at p(gold)=1.00.  So the candidates are WRITTEN for each
item instead of borrowed: same type, same register, same conversational world, and individually
unfalsifiable without the conversation.

The generator sees ONLY the question and the gold.  It never sees the conversation, the carrier,
or which world (original / counterfactual) the carrier will be harvested from, so it cannot
influence the contrast it exists to make measurable.  kv_e3ctrl still picks WHICH candidate to use
by the no-context prior probe, so a candidate that happens to be implausible is simply not chosen.

    python -m kvmemory.kv_e3negs --ds all --out ./out/e3_negs.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_api import chat as _llm_chat
from kvmemory.kv_harvest3 import load_realtalk, load_locomo, select_qa
from kvmemory.kv_harvest4 import N_INJECT, hard_neg

N_CAND = 8

PROMPT = """You are building a two-choice memory test over a long personal conversation.

Question: {q}
True answer: {gold}

Write {n} WRONG answers to that question. Each one must be:
- the same kind of thing as the true answer (a place for a place, a date for a date, a count for a
  count, an activity for an activity), and about the same length and register;
- something that could plausibly have been true of this person -- a reader who has NOT seen the
  conversation must have no way to tell it from the true answer;
- NOT a paraphrase, synonym, translation, abbreviation or superset/subset of the true answer, and
  not obviously absurd, comic, or archaic.

Reply with a JSON array of {n} strings and nothing else."""


def qkey(ds, cid, q):
    return f"{ds}|{cid}|{hashlib.sha1(q.encode()).hexdigest()[:12]}"


def chat(prompt, timeout=120):
    """One chat call to the configured OpenAI-compatible endpoint (kvmemory.llm_api)."""
    return _llm_chat(prompt, timeout=timeout)


def parse(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    return [str(x).strip() for x in arr if isinstance(x, (str, int, float)) and str(x).strip()]


def eligible(ds_name, cv, n_inject=N_INJECT):
    """Byte-identical to kv_harvest4 phase B / kv_e3ctrl eligibility (the CAP is the only knob:
    N_INJECT=4 reproduces the published 62-item set, a larger cap buys statistical power)."""
    qas = select_qa(cv)
    turns = cv["turns"]
    items = []
    for qa in qas:
        gl = qa["gold"].strip().lower()
        if len(gl) < 3:
            continue
        if not any(gl in turns[i][2].lower() for i in qa["goldev"]):
            continue
        neg0 = hard_neg(qas, qa)
        if not neg0:
            continue
        if neg0.strip().lower() in gl or gl in neg0.strip().lower():
            continue
        items.append(qa)
    items.sort(key=lambda x: (-x["strong"], x["q"]))
    return items[:n_inject]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["realtalk", "locomo", "all"], default="all")
    ap.add_argument("--rt_path", default="./data/REALTALK")
    ap.add_argument("--lc_path", default="./data/locomo10.json")
    ap.add_argument("--n_inject", type=int, default=N_INJECT,
                    help="per-conversation cap; 4 = the published item set, 12 = 111 items")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="./out/e3_negs.json")
    args = ap.parse_args()
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(v, None)

    tasks = []
    if args.ds in ("locomo", "all"):
        tasks += [("locomo", c) for c in load_locomo(args.lc_path)]
    if args.ds in ("realtalk", "all"):
        tasks += [("realtalk", c) for c in load_realtalk(args.rt_path)]

    jobs = []
    for ds_name, cv in tasks:
        conv_low = "\n".join(t[2] for t in cv["turns"]).lower()
        for qa in eligible(ds_name, cv, args.n_inject):
            jobs.append((qkey(ds_name, cv["cid"], qa["q"]), qa["q"], qa["gold"].strip(), conv_low))
    print(f"[negs] {len(jobs)} eligible items over {len(tasks)} conversations", flush=True)

    def run(j):
        k, q, gold, conv_low = j
        gl = gold.lower()
        best = []
        for attempt in range(3):
            try:
                cands = parse(chat(PROMPT.format(q=q, gold=gold, n=N_CAND)))
            except Exception as e:
                cands = []
                if attempt == 2:
                    print(f"[negs] FAIL {k}: {e}", flush=True)
            keep = []
            for c in cands:
                cl = c.lower().strip()
                if len(cl) < 2 or cl == gl or cl in gl or gl in cl:
                    continue
                if cl in conv_low:          # already in the history -> not a clean negative
                    continue
                if cl in {x.lower() for x in keep}:
                    continue
                keep.append(c.strip())
            best = keep
            if len(best) >= 4:
                break
        return k, best

    res = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, v in ex.map(run, jobs):
            res[k] = v
    n_ok = sum(1 for v in res.values() if len(v) >= 4)
    json.dump({"model": MODEL, "n_cand": N_CAND, "negs": res}, open(args.out, "w"), indent=1)
    print(f"[negs] wrote {args.out}: {len(res)} items, {n_ok} with >=4 usable candidates")
    for k in sorted(res)[:5]:
        print(f"  {k}: {res[k]}")
    print("E3NEGS_DONE", flush=True)


if __name__ == "__main__":
    main()
