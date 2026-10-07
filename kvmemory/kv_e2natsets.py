"""kv_e2natsets.py -- the NATURAL-OCCURRENCE serve sets: same retriever, NO forced exclusion.

The plan asked for this alongside the forced condition and it was not run at first; the E2
write-up lists it as never-run condition 2.  The forced version answers "when the retriever CANNOT
reach a value row, is spliced KV worth more than the same text re-encoded".  Its weakness:
every arm sits near the floor, so `harv == self` is a parity statement about a task nobody
can do.  The natural version fixes exactly that:

    rank the FULL pool, take top-k, and let the chips fall.  Some items get a value row served,
    some do not.  On the ones that DO, accuracy is far above the floor -- and parity there is a
    parity statement at an operating point that actually works.

So the point of this run is NOT another shot at materialization (six failures in the forced condition).  It is
to turn claim A -- "reused KV is indistinguishable from re-encoding the same text" -- from a
floor-level result into a useful-operating-point result.

Everything else is byte-identical to kv_e2sets: same arm, same scores, same placebo rule, same
`_utterance` text.  The ONLY changes are the candidate pool and what gets recorded:

    cand = every turn                  (was: every turn not in V)
    nV_served recorded per item        (was: asserted to be 0)
    placebo also natural               (was: additionally forbidden this question's V rows)

--- kv_e2sets.py docstring follows --------------------------------------------------------------

E2 step 1: build the V-EXCLUDED serve sets (CPU/GPU, no generation).

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

    python -m kvmemory.kv_e2natsets --data ./data/locomo10.json \\
        --labels ./out/rA_labels.json --loose ./out/rA_loose.json \\
        --arm rerank --k 5 --forced ./out/e2_sets.json --out ./out/e2_natsets.json
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
                                                          "rows -- with none, nV_served is 0 by "
                                                          "definition and the item carries no "
                                                          "information about reachability")
    ap.add_argument("--forced", default="", help="the forced-condition sets (e2_sets.json). If "
                                                 "given, report how much the natural serve sets "
                                                 "actually differ -- if they barely differ, the "
                                                 "retriever never picks a value row anyway and "
                                                 "this whole condition is vacuous.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = json.load(open(args.labels))["records"]
    loose = {}
    if args.loose and os.path.exists(args.loose):
        for r in json.load(open(args.loose))["records"]:
            loose[(str(r["conv"]), r["qi"])] = (r.get("V_loose") or [], r.get("n_judged", 0))
        print(f"[e2natsets] loose channel: {len(loose)} questions judged", flush=True)
    else:
        print("[e2natsets] WARNING no loose file -- exclusion is lexical only, so a paraphrased "
              "value row can survive into the serve set", flush=True)

    eps = {str(e.conv_id): e for e in load_locomo(args.data)}
    arms = build_arms(args.arm, model_dir=args.model_dir, cache_dir=args.cache_dir,
                      rerank_depth=args.rerank_depth)
    # build_arms keys the registry by each arm's DISPLAY name ("rerank(rrf,d=50)"), not by the
    # string you asked for, so arms[args.arm] is a KeyError.  One arm requested -> one arm back.
    assert len(arms) == 1, f"--arm takes exactly one arm, got {list(arms)}"
    arm_name, arm = next(iter(arms.items()))
    print(f"[e2natsets] arm = {arm_name}", flush=True)

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
            # THE CHANGE: rank the whole pool.  Whether a value row gets served is now an
            # OUTCOME of the retriever, not something we imposed.
            cand = list(range(len(ctx.txt)))
            top = sorted(cand, key=lambda i: (-s[i], -ctx.turn[i]))[: args.k]
            # placebo: another question of the same conversation, different gold, same exclusion
            pl = None
            for off in range(1, len(rows)):
                r2 = rows[(j + off) % len(rows)]
                if str(r2["gold"]).strip().lower() == str(r["gold"]).strip().lower():
                    continue
                s2 = sc[r2["qi"]]
                t2 = sorted(range(len(ctx.txt)), key=lambda i: (-s2[i], -ctx.turn[i]))[: args.k]
                if t2:
                    pl = (r2["qi"], r2["question"], t2)
                    break
            if pl is None:
                stats["skip_no_placebo"] += 1
                continue
            nv_served = sorted(set(top) & V)
            stats["nV_served_%d" % min(len(nv_served), 3)] += 1
            out.append({
                "nV_served": len(nv_served), "V_served_idx": nv_served,
                "p_nV_served": len(set(pl[2]) & V),
                "conv": cid, "qi": r["qi"], "cat": r["cat"], "question": r["question"],
                "gold": r["gold"], "n_turns": r["n_turns"],
                "nV_strict": len(r["V"]["N1"]), "nV_used": len(V),
                "loose_judged": loose.get((cid, r["qi"]), ([], 0))[1],
                "serve": top, "placebo_qi": pl[0], "placebo_q": pl[1], "placebo": pl[2],
                "gold_idx": r["gold_idx"],
                "serve_has_goldev": sorted(set(top) & set(r["gold_idx"])),
            })
            stats["kept"] += 1
        print(f"[e2natsets] conv {cid}: {sum(1 for o in out if o['conv'] == cid)} items", flush=True)

    # GATE: the natural sets have to actually differ from the forced ones, otherwise the
    # retriever never reaches a value row and this condition measures nothing new.
    if args.forced and os.path.exists(args.forced):
        F = {(str(i["conv"]), i["qi"]): i for i in json.load(open(args.forced))["items"]}
        both = [o for o in out if (o["conv"], o["qi"]) in F]
        same = sum(1 for o in both if set(o["serve"]) == set(F[(o["conv"], o["qi"])]["serve"]))
        jac = [len(set(o["serve"]) & set(F[(o["conv"], o["qi"])]["serve"]))
               / len(set(o["serve"]) | set(F[(o["conv"], o["qi"])]["serve"])) for o in both]
        print("\n[natsets] vs forced sets: %d shared items, %d (%.1f%%) have an IDENTICAL serve "
              "set, mean Jaccard %.3f" % (len(both), same, 100 * same / max(len(both), 1),
                                          sum(jac) / max(len(jac), 1)))
        print("[natsets] items whose natural serve set contains >=1 value row: %d (%.1f%%)"
              % (sum(1 for o in out if o["nV_served"]),
                 100 * sum(1 for o in out if o["nV_served"]) / max(len(out), 1)))
        assert same < len(both), "natural == forced on EVERY item -- condition is vacuous"

    json.dump({"arm": arm_name, "arm_flag": args.arm, "k": args.k, "natural": True,
               "items": out, "stats": dict(stats)}, open(args.out, "w"), indent=1)
    n = len(out)
    print(f"\n===== E2 serve sets ({arm_name}, k={args.k}) =====")
    print(f"  items: {n}   skipped: no value row {stats['skip_no_value_row']}, "
          f"no placebo {stats['skip_no_placebo']}")
    print(f"  conversations: {len(set(o['conv'] for o in out))}")
    per = Counter(o["conv"] for o in out)
    print(f"  per conversation: {sorted(per.values())}")
    from collections import Counter as _C
    d = _C(o["nV_served"] for o in out)
    print(f"  nV_served distribution: {dict(sorted(d.items()))}")
    print(f"  naturally zero V ({sum(v for k, v in d.items() if k == 0)}) vs some V "
          f"({sum(v for k, v in d.items() if k > 0)})")
    print(f"  items whose serve set still contains an official gold-evidence turn: "
          f"{sum(1 for o in out if o['serve_has_goldev'])} "
          f"(these are evidence turns that do NOT state the value -- kept, and flagged)")
    print(f"  mean |V| excluded: {sum(o['nV_used'] for o in out)/n:.2f} "
          f"(strict-only would have been {sum(o['nV_strict'] for o in out)/n:.2f})")
    print("E2NATSETS_DONE", flush=True)


if __name__ == "__main__":
    main()
