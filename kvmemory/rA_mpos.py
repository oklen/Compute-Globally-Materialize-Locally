"""rA_mpos.py -- E1 follow-up: is the top-ranked MENTION row a plausible CARRIER, or just noise?

E1 (rA_rank.py) measures pi = P(min rank over M < min rank over V): a deployed index puts a
topic-matching turn above every value-bearing turn. That says the served set is dominated by
non-value rows. It does NOT say those rows carry anything.

  M  = "shares >=2 content words with the question, does not state the value"
     = { rows written AFTER the value was stated }  u  { rows written BEFORE it, or unrelated }

Only the first subset can possibly carry a materialized value: semantic materialization is a
WRITE-TIME commitment, so a turn written before the source event cannot have absorbed it. A turn
written after it, in the same conversation, referring to the same topic, is exactly the shape of
`root` in the synthetic template (references the source, does not restate the value).

So we split the top-ranked M row by its position relative to the earliest V row:

  DOWNSTREAM  turn(M*) > min turn(V)    carrier-shaped: materialization is POSSIBLE
  UPSTREAM    turn(M*) < min turn(V)    carrier-impossible: pure topical noise

This is necessary, NOT sufficient. Being downstream does not make a row a carrier -- proving that
needs a donor pair (E2). What this decides is whether E2's V-excluded serve condition is asking a
real question or an artificial one: if the rows a real retriever brings back instead of the value
are overwhelmingly UPSTREAM, there is nothing there to harvest and E2 is answering a question the
corpus never poses. If they are DOWNSTREAM, the setup is the one the corpus actually produces.

TWO SEPARATE QUESTIONS, and they must not be conflated:

  (a) POSITION  is the top-M row downstream of the value?
      Reported against the base rate `M_down_frac` -- the fraction of the WHOLE M set that is
      downstream. If observed ~= base, the retriever is not choosing downstream-ness; the corpus is.
      A high downstream rate with zero excess means the setup is real but the retriever is blind
      to it, which is a claim about the corpus, not about the retriever.

  (b) REFERENCE  does the top-M row talk about the same EVENT as the value row?
      `root` in the synthetic template *references the source*. The real-corpus proxy is content
      overlap with the value row EXCLUDING the question's own terms -- M rows share question terms
      by construction, so only the residual overlap carries information. Compared against the mean
      over all downstream M rows in the same item: excess > 0 means the retriever lands on rows
      that discuss the value's event; excess ~= 0 means it lands on topical noise that merely
      shares the question's words. ONLY (b) makes "the retriever brought back the root" a claim
      about references rather than about turn numbers.

Reported per category, because E1's headline is that multi-hop inverts.

    python -m kvmemory.rA_mpos --data ./data/locomo10.json --labels ./out/rA_labels.json --arm bm25
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.locomo import load_locomo
from kvmemory.components import _tok
from kvmemory.rA_rank import ranks_from, cluster_bootstrap, _ctx
from kvmemory.rA_arms import build_arms
from kvmemory.rA_roles import _utterance

CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--arm", default="bm25",
                    help="one arm name from rA_arms.build_arms (bm25/dense/rrf/rerank/...)")
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="out/rA_mpos.json")
    args = ap.parse_args()

    eps = load_locomo(args.data)
    seg_index = {ep.conv_id: ep.segments for ep in eps}
    recs = [r for r in json.load(open(args.labels))["records"]
            if r["nV"]["N1"] > 0 and r["nM"] > 0]
    arms = build_arms(args.arm, model_dir=args.model_dir, cache_dir=args.cache_dir)
    assert len(arms) == 1, f"--arm must name exactly one arm, got {list(arms)}"
    arm = next(iter(arms.values()))

    doc_cache, items = {}, []
    for r in recs:
        conv = r["conv"]
        ctx = _ctx(seg_index, conv, False, doc_cache)
        segs, turns, docs = seg_index[conv], ctx.turn, ctx.tok
        V, M = r["V"]["N1"], r["M_strict"]
        pos = ranks_from(arm.scores(r["question"], ctx), turns)
        # the top-ranked row of each role
        mstar = min(M, key=lambda i: pos[i])
        vstar = min(V, key=lambda i: pos[i])
        v_first = min(turns[i] for i in V)          # earliest turn that states the value
        beat = int(pos[mstar] < pos[vstar])
        # (b) reference probe. Question terms are shared by every M row by construction, so they
        # are subtracted out; what remains is content the row shares with the VALUE row itself.
        qset = set(_tok(r["question"]))
        vset = set(docs[vstar]) - qset
        def _ref(i):
            return len((set(docs[i]) - qset) & vset)
        down_M = [i for i in M if turns[i] > v_first]
        ref_star = _ref(mstar)
        ref_base = statistics.fmean(_ref(i) for i in down_M) if down_M else float("nan")
        items.append({
            "conv": conv, "qi": r["qi"], "cat": r["cat"], "beat": beat,
            "m_turn": turns[mstar], "v_first": v_first,
            "downstream": int(turns[mstar] > v_first),
            "gap": turns[mstar] - v_first,
            # how much of the served-instead-of-value mass is carrier-shaped: over the whole M set
            "M_down_frac": statistics.fmean(int(turns[i] > v_first) for i in M),
            "ref_star": ref_star, "ref_base": ref_base,
            "ref_excess": (ref_star - ref_base) if down_M else None,
            "m_text": _utterance(segs[mstar])[:180],
            "v_text": _utterance(segs[vstar])[:180],
            "question": r["question"], "gold": r["gold"],
        })

    print(f"arm = {arm.name} | items {len(items)}\n")

    def block(tag, sub):
        if not sub:
            return None
        won = [x for x in sub if x["beat"]]
        if not won:
            print(f"  {tag:26s} n={len(sub):4d}  (no items where M beats V)")
            return None
        d = statistics.fmean(x["downstream"] for x in won)
        lo, hi = cluster_bootstrap(won, lambda P: statistics.fmean(x["downstream"] for x in P)) \
            if len({x["conv"] for x in won}) > 1 else (float("nan"),) * 2
        gaps = [x["gap"] for x in won if x["downstream"]]
        rw = [x for x in won if x["ref_excess"] is not None]
        rex = statistics.fmean(x["ref_excess"] for x in rw) if rw else float("nan")
        rlo, rhi = cluster_bootstrap(rw, lambda P: statistics.fmean(x["ref_excess"] for x in P)) \
            if rw and len({x["conv"] for x in rw}) > 1 else (float("nan"),) * 2
        print(f"  {tag:26s} n={len(sub):4d} | M beats V on {len(won):4d}")
        print(f"  {'':26s}   (a) POSITION  downstream {d:.3f} [{lo:.3f},{hi:.3f}]  "
              f"base {statistics.fmean(x['M_down_frac'] for x in won):.3f}  "
              f"EXCESS {d - statistics.fmean(x['M_down_frac'] for x in won):+.3f}  "
              f"| median gap +{statistics.median(gaps) if gaps else float('nan'):.0f} turns")
        print(f"  {'':26s}   (b) REFERENCE shared-with-V terms (question terms removed): "
              f"top-M {statistics.fmean(x['ref_star'] for x in rw):.2f} vs "
              f"downstream-M mean {statistics.fmean(x['ref_base'] for x in rw):.2f}  "
              f"EXCESS {rex:+.2f} [{rlo:+.2f},{rhi:+.2f}]")
        return {"n": len(sub), "n_beat": len(won), "downstream": d, "ci": [lo, hi],
                "median_gap": statistics.median(gaps) if gaps else None,
                "M_down_frac": statistics.fmean(x["M_down_frac"] for x in won),
                "pos_excess": d - statistics.fmean(x["M_down_frac"] for x in won),
                "ref_star": statistics.fmean(x["ref_star"] for x in rw) if rw else None,
                "ref_base": statistics.fmean(x["ref_base"] for x in rw) if rw else None,
                "ref_excess": rex, "ref_excess_ci": [rlo, rhi]}

    res = {"overall": block("ALL", items)}
    print()
    for c in sorted(CATS):
        res[f"cat{c}"] = block(f"cat{c} {CATS[c]}", [x for x in items if x["cat"] == c])

    print("\nEXAMPLES -- multi-hop items where a downstream mention outranked every value row\n")
    ex = [x for x in items if x["cat"] == 1 and x["beat"] and x["downstream"]]
    for x in ex[:6]:
        print(f"  Q: {x['question']}")
        print(f"  gold: {x['gold']}")
        print(f"    V  (turn {x['v_first']}): {x['v_text']}")
        print(f"    M* (turn {x['m_turn']}, +{x['gap']}): {x['m_text']}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": res, "items": items}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
