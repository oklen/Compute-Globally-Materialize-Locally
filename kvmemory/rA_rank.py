"""rA_rank.py -- E1: measure R-A' on a real corpus.

    pi = P( min rank over M  <  min rank over V )

"a deployed index ranks a turn that matches the topic but does NOT state the value above the turn
that DOES state it." Roles come from rA_roles.py; this module only ranks and counts.

WHAT IS AND IS NOT CLAIMED. The synthetic result (paper section 7) is a *theorem* about the
term-frequency family: the carrier and the reference edge have identical term multisets w.r.t. the
query, so BM25 returns the identical float and the recency tiebreak decides. Real turns are not
term-identical, so nothing here is provable -- pi is a *rate*, an empirical quantity, and must be
reported as such.

INDEXING TEXT vs LABELLING TEXT (deliberate asymmetry).
  * labels  use the UTTERANCE ONLY (rA_roles._utterance) -- the "[timestamp] speaker: " prefix
    that locomo.py:54 prepends would otherwise make every date answer match its session timestamp
    and every question naming a speaker share >=2 tokens with all of that speaker's turns.
  * retrieval indexes the FULL seg.text, because that is what a deployed memory system stores.
    --index utterance re-runs on the stripped text as a sensitivity check; the gap is exactly how
    much of retrieval performance is the speaker/timestamp prefix.

THREE GATES, all reported before any pi:
  variance  per-item rank statistics must have non-zero variance and a non-constant top-1 role.
            On the synthetic template BM25 *and* dense both score 0 here -- the 200 items are
            near-isomorphic -- so this gate catches corpus homogeneity, not scorer weakness.
  planted   a row that trivially answers the question is inserted; every arm must rank it #1.
            An arm that cannot find a planted row is broken, not challenged.
  leak      indexing against question+gold must lift the best-V rank sharply (validated on the
            synthetic side, where the source jumps to rank 0 with r@1 = 1.00).

    python -m kvmemory.rA_rank --data ./data/locomo10.json --labels ./out/rA_labels.json \
        --arms lex,dense,rrf,rerank --model-dir ./models --out ./out/rA_rank.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.locomo import load_locomo
from kvmemory.components import _tok
from kvmemory.rA_roles import _utterance


def ranks_from(scores: list[float], turns: list[int]) -> list[int]:
    """rank[i] = 0-based position of candidate i. Ties broken toward recency, as every router in
    components.py does (the (score, turn) descending sort)."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], turns[i]), reverse=True)
    pos = [0] * len(scores)
    for r, i in enumerate(order):
        pos[i] = r
    return pos


# ---------------------------------------------------------------- measurement

def _ctx(seg_index, conv, use_utterance, cache):
    from kvmemory.rA_arms import Ctx
    key = (conv, use_utterance)
    if key not in cache:
        segs = seg_index[conv]
        texts = [(_utterance(s) if use_utterance else s.text) for s in segs]
        cache[key] = Ctx(conv=conv, tok=[_tok(t) for t in texts], txt=texts,
                         turn=[s.turn for s in segs])
    return cache[key]


def measure(arms, recs, seg_index, use_utterance: bool, query_of):
    """One pass per arm. Returns per-arm list of per-item dicts."""
    out = {a: [] for a in arms}
    doc_cache = {}
    for r in recs:
        conv = r["conv"]
        ctx = _ctx(seg_index, conv, use_utterance, doc_cache)
        turns = ctx.turn
        V, M = r["V"]["N1"], r["M_strict"]
        if not V or not M:
            continue
        q = query_of(r)
        for aname, arm in arms.items():
            pos = ranks_from(arm.scores(q, ctx), turns)
            rv, rm = min(pos[i] for i in V), min(pos[i] for i in M)
            top = pos.index(0)
            out[aname].append({
                "conv": conv, "qi": r["qi"], "cat": r["cat"],
                "nV": len(V), "nM": len(M), "short": r.get("short_answer", 0),
                # Under a uniform random permutation the smallest rank in M u V lands in M with
                # probability |M|/(|M|+|V|). Raw pi is NOT interpretable without this: with |M|
                # median 10 and |V| median 1, chance alone puts pi near .86. The primary
                # statistic is therefore delta = beat - chance; delta < 0 means the arm finds
                # value rows BETTER than a coin, delta > 0 means it actively prefers mentions.
                "chance": len(M) / (len(M) + len(V)),
                "rank_V": rv, "rank_M": rm, "beat": int(rm < rv),
                "top1_role": "V" if top in set(V) else ("M" if top in set(M) else "U"),
            })
    return out


def cluster_bootstrap(items, stat, b=4000, seed=1234):
    """Resample CONVERSATIONS, not items -- questions inside one conversation are far from
    independent (same people, same events, the same turns serving as evidence repeatedly)."""
    by = defaultdict(list)
    for it in items:
        by[it["conv"]].append(it)
    convs = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(b):
        pool = []
        for _ in range(len(convs)):
            pool += by[convs[rng.randrange(len(convs))]]
        if pool:
            vals.append(stat(pool))
    vals.sort()
    lo = vals[int(.025 * len(vals))]
    hi = vals[int(.975 * len(vals)) - 1]
    return lo, hi


def _delta(P):
    return statistics.fmean(x["beat"] - x["chance"] for x in P)


def report(tag, arm_items):
    pi = statistics.fmean(x["beat"] for x in arm_items)
    ch = statistics.fmean(x["chance"] for x in arm_items)
    dl = pi - ch
    dlo, dhi = cluster_bootstrap(arm_items, _delta)
    lo, hi = cluster_bootstrap(arm_items, lambda P: statistics.fmean(x["beat"] for x in P))
    rv = [x["rank_V"] for x in arm_items]
    rm = [x["rank_M"] for x in arm_items]
    top = Counter(x["top1_role"] for x in arm_items)
    gate_var = statistics.pvariance(rv) > 0 and len(top) > 1
    verdict = "prefers MENTIONS" if dlo > 0 else ("finds VALUES" if dhi < 0 else "n.s.")
    print(f"  {tag:22s} pi={pi:.3f}[{lo:.3f},{hi:.3f}]  chance={ch:.3f}  "
          f"DELTA={dl:+.3f}[{dlo:+.3f},{dhi:+.3f}] {verdict:16s}"
          f"| rV mean {statistics.fmean(rv):6.1f} med {statistics.median(rv):4.0f} "
          f"| rM mean {statistics.fmean(rm):5.1f} | top1 {dict(top)} "
          f"| VAR {'PASS' if gate_var else 'FAIL'}")
    return {"pi": pi, "ci": [lo, hi], "chance": ch, "delta": dl, "delta_ci": [dlo, dhi],
            "n": len(arm_items),
            "rank_V_mean": statistics.fmean(rv), "rank_M_mean": statistics.fmean(rm),
            "rank_V_var": statistics.pvariance(rv), "top1": dict(top),
            "var_gate": gate_var}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--index", choices=["full", "utterance"], default="full")
    ap.add_argument("--arms", default="lex",
                    help="comma list: lex,bm25,overlap,dense,rrf,rerank")
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--cache-dir", default=None, help="where dense doc embeddings are cached")
    ap.add_argument("--rerank-depth", type=int, default=50)
    ap.add_argument("--skip-planted", action="store_true",
                    help="the planted-row gate re-scores 200 pools; skip for the model arms once "
                         "the lexical arms have established the harness is sound")
    ap.add_argument("--out", default="out/rA_rank.json")
    args = ap.parse_args()

    from kvmemory.rA_arms import build_arms
    ARMS = build_arms(args.arms, model_dir=args.model_dir, cache_dir=args.cache_dir,
                      rerank_depth=args.rerank_depth)

    eps = load_locomo(args.data)
    seg_index = {ep.conv_id: ep.segments for ep in eps}
    L = json.load(open(args.labels))
    recs = [r for r in L["records"] if r["nV"]["N1"] > 0 and r["nM"] > 0]
    print(f"measurable questions {len(recs)} over {len({r['conv'] for r in recs})} conversations "
          f"| indexing = {args.index} text | arms = {list(ARMS)}\n")

    res = {"_arms": list(ARMS), "_index": args.index, "_rerank_depth": args.rerank_depth}

    print("PRIMARY -- bare question")
    prim = measure(ARMS, recs, seg_index, args.index == "utterance", lambda r: r["question"])
    res["primary"] = {a: report(a, it) for a, it in prim.items()}

    print("\nGATE: format leak (index against question + gold; best-V rank must drop sharply)")
    leak = measure(ARMS, recs, seg_index, args.index == "utterance",
                   lambda r: r["question"] + " " + r["gold"])
    res["leak"] = {}
    for a in ARMS:
        m0 = statistics.fmean(x["rank_V"] for x in prim[a])
        m1 = statistics.fmean(x["rank_V"] for x in leak[a])
        ok = m1 < m0 / 2
        print(f"  {a:22s} rank(best V) {m0:.1f} -> {m1:.1f}   {'PASS' if ok else 'FAIL'}")
        res["leak"][a] = {"bare": m0, "leaked": m1, "pass": ok}

    res["planted"] = {}
    if args.skip_planted:
        print("\nGATE: planted row -- SKIPPED (--skip-planted)")
    else:
        print("\nGATE: planted row (a row that trivially answers must rank #1 for every arm)")
        from kvmemory.rA_arms import Ctx
        n_plant = min(200, len(recs))
        for a, arm in ARMS.items():
            hits = 0
            for r in recs[:n_plant]:
                segs = seg_index[r["conv"]]
                texts = [(_utterance(s) if args.index == "utterance" else s.text) for s in segs]
                plant = f"{r['question']} {r['gold']} {r['gold']}"
                texts = texts + [plant]
                # the pool changed, so per-conversation caches inside an arm must not be reused:
                # give the planted pool its own conv key.
                c = Ctx(conv=f"{r['conv']}#plant{r['qi']}", tok=[_tok(t) for t in texts],
                        txt=texts, turn=[s.turn for s in segs] + [max(s.turn for s in segs) + 1])
                pos = ranks_from(arm.scores(r["question"], c), c.turn)
                hits += int(pos[-1] == 0)
            print(f"  {a:22s} planted row at rank 0 on {hits}/{n_plant}   "
                  f"{'PASS' if hits >= 0.95 * n_plant else 'FAIL'}")
            res["planted"][a] = hits

    print("\nSTRATIFIED by |V| (does pi fall as the value appears in more turns?)")
    res["by_nV"] = {}
    for a, items in prim.items():
        row = {}
        for lo, hi, lab in ((1, 1, "1"), (2, 3, "2-3"), (4, 99, "4+")):
            sub = [x for x in items if lo <= x["nV"] <= hi]
            if sub:
                row[lab] = (len(sub), statistics.fmean(x["beat"] for x in sub),
                            statistics.fmean(x["chance"] for x in sub))
        print(f"  {a:22s} " + "  ".join(
            f"|V|={k}: d={v[1]-v[2]:+.3f} (pi={v[1]:.3f} ch={v[2]:.3f} n={v[0]})"
            for k, v in row.items()))
        res["by_nV"][a] = row

    print("\nSTRATIFIED by category")
    for a, items in prim.items():
        row = {}
        for c in sorted({x["cat"] for x in items}):
            sub = [x for x in items if x["cat"] == c]
            dlo, dhi = cluster_bootstrap(sub, _delta) if len({s["conv"] for s in sub}) > 1 else (float("nan"),) * 2
            row[c] = (len(sub), statistics.fmean(x["beat"] for x in sub),
                      statistics.fmean(x["chance"] for x in sub), dlo, dhi)
        print(f"  {a:22s} " + "  ".join(
            f"cat{k}: d={v[1]-v[2]:+.3f}[{v[3]:+.3f},{v[4]:+.3f}] (n={v[0]})"
            for k, v in row.items()))
        res.setdefault("by_cat", {})[a] = row

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
