"""kv_e2sets.py -- E2 step 1: build the V-EXCLUDED serve sets (CPU/GPU, no generation).

E2 asks the composite question the whole §7 story needs:

    when a deployed retriever is FORCED to work without any row that states the value, are the rows
    it does pick worth more as spliced KV than as the same text re-encoded in isolation?

Two things have to be true of the serve set before that question means anything, and both are
easier to get wrong than right:

  (1) ZERO value rows served -- not "the gold evidence turn was not served".  `evidence_in_ctx` is
      a conjunction over the official gold ids and cannot express "no row anywhere in the served
      set states the answer".  We build the V set ourselves (rA_roles' normalisation ladder) and
      union it with the LLM paraphrase channel (rA_loose), because a row that states the value in
      paraphrase is a value row no matter what the substring test says.  Everything in that union
      is removed from the candidate pool BEFORE the retriever ranks, so the condition holds by
      construction rather than by luck -- which is also what makes the sample size usable.
  (2) A PLACEBO serve set.  If spliced KV beats isolated encoding on rows selected for a DIFFERENT
      question of the same conversation, the benefit is generic contextualisation, not anything
      about this question's omitted value.  The placebo is the same retriever, the same k, the same
      exclusion, just a different query -- and we additionally forbid it from containing any of
      THIS question's value rows.

Serving is at the retriever's own top-k with no hot window: kvmemory's router serves `hot_turns`
recent turns verbatim outside the routed budget, which would put un-retrieved rows into the context
and break condition (1).  E2 runs hot=0 and the serve set here is the whole story.

    python -m kvmemory.kv_e2sets --data ./data/locomo10.json \\
        --labels ./out/rA_labels.json --loose ./out/rA_loose.json \\
        --arm rerank --k 5 --out ./out/e2_sets.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.locomo import load_locomo
from kvmemory.rA_arms import build_arms, Ctx
from kvmemory.rA_roles import _utterance, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--loose", default="", help="rA_loose output; its V_loose rows are unioned "
                                                "into the exclusion (paraphrase channel)")
    ap.add_argument("--arm", default="rerank")
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--cache-dir", default="./out/e2_dense_cache")
    ap.add_argument("--rerank-depth", type=int, default=50)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-nv", type=int, default=1, help="require at least this many known value "
                                                          "rows -- without one there is nothing to "
                                                          "exclude and the condition is vacuous")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = json.load(open(args.labels))["records"]
    loose = {}
    if args.loose and os.path.exists(args.loose):
        for r in json.load(open(args.loose))["records"]:
            loose[(str(r["conv"]), r["qi"])] = (r.get("V_loose") or [], r.get("n_judged", 0))
        print(f"[e2sets] loose channel: {len(loose)} questions judged", flush=True)
    else:
        print("[e2sets] WARNING no loose file -- exclusion is lexical only, so a paraphrased "
              "value row can survive into the serve set", flush=True)

    eps = {str(e.conv_id): e for e in load_locomo(args.data)}
    arms = build_arms(args.arm, model_dir=args.model_dir, cache_dir=args.cache_dir,
                      rerank_depth=args.rerank_depth)
    # build_arms keys the registry by each arm's DISPLAY name ("rerank(rrf,d=50)"), not by the
    # string you asked for, so arms[args.arm] is a KeyError.  One arm requested -> one arm back.
    assert len(arms) == 1, f"--arm takes exactly one arm, got {list(arms)}"
    arm_name, arm = next(iter(arms.items()))
    print(f"[e2sets] arm = {arm_name}", flush=True)

    ctxs = {}
    for cid, ep in eps.items():
        txt = [_utterance(s) for s in ep.segments]
        ctxs[cid] = Ctx(conv=cid, tok=[list(toks(t)) for t in txt], txt=txt,
                        turn=[s.turn for s in ep.segments])

    by_conv = defaultdict(list)
    for r in recs:
        by_conv[str(r["conv"])].append(r)

    out, stats = [], Counter()
    for cid in sorted(by_conv):
        ctx = ctxs[cid]
        rows = by_conv[cid]
        # rank once per question, then mask -- masking before ranking and masking after are the
        # same top-k as long as the scorer is per-document, which every arm here is.
        sc = {}
        for r in rows:
            sc[r["qi"]] = arm.scores(r["question"], ctx)
        vsets = {}
        for r in rows:
            lv, nj = loose.get((cid, r["qi"]), ([], 0))
            vsets[r["qi"]] = set(r["V"]["N1"]) | set(lv)
        for j, r in enumerate(rows):
            V = vsets[r["qi"]]
            if len(V) < args.min_nv:
                stats["skip_no_value_row"] += 1
                continue
            s = sc[r["qi"]]
            cand = [i for i in range(len(ctx.txt)) if i not in V]
            top = sorted(cand, key=lambda i: (-s[i], -ctx.turn[i]))[: args.k]
            # placebo: another question of the same conversation, different gold, same exclusion
            pl = None
            for off in range(1, len(rows)):
                r2 = rows[(j + off) % len(rows)]
                if str(r2["gold"]).strip().lower() == str(r["gold"]).strip().lower():
                    continue
                s2 = sc[r2["qi"]]
                cand2 = [i for i in range(len(ctx.txt)) if i not in V and i not in vsets[r2["qi"]]]
                t2 = sorted(cand2, key=lambda i: (-s2[i], -ctx.turn[i]))[: args.k]
                if t2 and not (set(t2) & V):
                    pl = (r2["qi"], r2["question"], t2)
                    break
            if pl is None:
                stats["skip_no_placebo"] += 1
                continue
            assert not (set(top) & V), "value row leaked into the serve set"
            out.append({
                "conv": cid, "qi": r["qi"], "cat": r["cat"], "question": r["question"],
                "gold": r["gold"], "n_turns": r["n_turns"],
                "nV_strict": len(r["V"]["N1"]), "nV_used": len(V),
                "loose_judged": loose.get((cid, r["qi"]), ([], 0))[1],
                "serve": top, "placebo_qi": pl[0], "placebo_q": pl[1], "placebo": pl[2],
                "gold_idx": r["gold_idx"],
                "serve_has_goldev": sorted(set(top) & set(r["gold_idx"])),
            })
            stats["kept"] += 1
        print(f"[e2sets] conv {cid}: {sum(1 for o in out if o['conv'] == cid)} items", flush=True)

    json.dump({"arm": arm_name, "arm_flag": args.arm, "k": args.k, "items": out, "stats": dict(stats)},
              open(args.out, "w"), indent=1)
    n = len(out)
    print(f"\n===== E2 serve sets ({arm_name}, k={args.k}) =====")
    print(f"  items: {n}   skipped: no value row {stats['skip_no_value_row']}, "
          f"no placebo {stats['skip_no_placebo']}")
    print(f"  conversations: {len(set(o['conv'] for o in out))}")
    per = Counter(o["conv"] for o in out)
    print(f"  per conversation: {sorted(per.values())}")
    print(f"  items whose serve set still contains an official gold-evidence turn: "
          f"{sum(1 for o in out if o['serve_has_goldev'])} "
          f"(these are evidence turns that do NOT state the value -- kept, and flagged)")
    print(f"  mean |V| excluded: {sum(o['nV_used'] for o in out)/n:.2f} "
          f"(strict-only would have been {sum(o['nV_strict'] for o in out)/n:.2f})")
    print("E2SETS_DONE", flush=True)


if __name__ == "__main__":
    main()
