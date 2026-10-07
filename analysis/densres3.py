"""Position control: how much of the stated/unstated asymmetry in (7b) comes from where the value-bearing sentence sits inside tool_out?

Motivation (2026-08-27): the `recent` window of H2O/StreamingLLM keeps the **tail of the whole sequence**,
for decoding continuity, whereas our `recency` arm only ranks inside tool_out (the sequence tail, scratchpad/answer, is always kept).
The two are not the same thing. Worse: if stated values naturally sit later and unstated values earlier,
then the 'asymmetry' of `recency` is a tautology -- it is position by construction.

A known instrument bias: the candidate scan in kv_dens.py:475-486 (pass 1) breaks at the **first eligible paragraph**,
and qs["unstated"] = got[0], so the main unstated probe systematically takes the **earliest** usable paragraph.

Criterion: after stratifying by the value sentence's relative position, which arms' conclusions hold and which collapse.
"""
import json, random, statistics as st, sys

R = json.load(open(sys.argv[1]))["rows"]
B, BUDS = 4000, (10, 20, 40)
RANKERS = ("oracle", "attn_h2o", "attn_snap", "attn_traj", "bm25_now", "bm25_q1", "idf", "recency")


def boot(xs, f=st.fmean, seed=1234):
    rnd = random.Random(seed); n = len(xs)
    o = sorted(f([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return o[int(.025 * B)], o[int(.975 * B)]


def cell(xs):
    lo, hi = boot(xs)
    return "%+.3f%s" % (st.fmean(xs), "*" if lo * hi > 0 else " ")


def relpos(r, k):
    """Relative position of the midpoint of the sentence holding the asked value inside tool_out, 0 = earliest, 1 = latest. Averaged over sentences."""
    U = r.get(k + "_sentence")
    if not U:
        return None
    lo, hi = min(u["s"] for u in U), max(u["e"] for u in U)
    g = [u for u in U if u["has_gold"]]
    if not g or hi <= lo:
        return None
    return st.fmean([((u["s"] + u["e"]) / 2 - lo) / (hi - lo) for u in g])


def qualified(k):
    return [r for r in R if r.get(k + "_full") is not None and abs(r.get(k + "_none", 0)) < 3
            and abs(r[k + "_full"] - r[k + "_frame"]) > 2]


print("=" * 92)
print("A  relative position of the value sentence inside tool_out: stated vs unstated")
for k in ("stated", "unstated"):
    ps = [p for p in (relpos(r, k) for r in qualified(k)) if p is not None]
    q = [sum(1 for p in ps if lo <= p < hi) for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))]
    print("  %-9s n=%3d  median %.3f  mean %.3f  quartiles[earliest..latest] %s"
          % (k, len(ps), st.median(ps), st.fmean(ps), q))

pair = [r for r in R if r.get("stated_full") is not None and r.get("unstated_full") is not None
        and abs(r.get("stated_none", 0)) < 3 and abs(r.get("unstated_none", 0)) < 3
        and abs(r["stated_full"] - r["stated_frame"]) > 2
        and abs(r["unstated_full"] - r["unstated_frame"]) > 2]
d = [a - b for a, b in ((relpos(r, "stated"), relpos(r, "unstated")) for r in pair)
     if a is not None and b is not None]
lo, hi = boot(d)
print("  same-item pairs, stated position - unstated position: n=%d  mean %+.3f  CI [%+.3f,%+.3f]  positive %d/%d"
      % (len(d), st.fmean(d), lo, hi, sum(1 for x in d if x > 0), len(d)))
print("  => stated values sit systematically later and unstated values earlier. Any **purely positional** ranker therefore produces a spurious asymmetry.")

for k in ("unstated", "stated"):
    sat = qualified(k)
    early = [r for r in sat if (relpos(r, k) or 1) < .5]
    late = [r for r in sat if (relpos(r, k) or 0) >= .5]
    print("\n" + "=" * 92)
    print("B  %s layer, paired per item vs random, stratified by value-sentence position" % k)
    print("  %-10s %-8s | %-26s | %s" % ("", "", "value early n=%d" % len(early), "value late n=%d" % len(late)))
    print("  %-10s %-8s | %8s %8s %8s | %8s %8s %8s"
          % ("ranker", "full sample", "B=10%", "B=20%", "B=40%", "B=10%", "B=20%", "B=40%"))
    for rk in RANKERS:
        def col(X, b):
            return [(r["ret_%s_%s_%d" % (k, rk, b)] - r["ret_%s_random_%d" % (k, b)])
                    / (r[k + "_full"] - r[k + "_frame"]) for r in X]
        if len(early) < 8 or len(late) < 8:
            print("  %-10s too few items per stratum" % rk); continue
        print("  %-10s %-8s | %8s %8s %8s | %8s %8s %8s"
              % (rk, cell(col(sat, 10)), *[cell(col(early, b)) for b in BUDS],
                 *[cell(col(late, b)) for b in BUDS]))

print("\n" + "=" * 92)
print("Reading:")
print("  holds (same sign and significance in both strata) => the arm's conclusion is not made by position.")
print("  collapses (significant in one stratum, reversed or null in the other) => the arm actually measures position.")
