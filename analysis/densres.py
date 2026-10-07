"""densres.py -- read out kv_dens.py (paper section 9).

Five questions, in the order the proposal registered them:

Q1  density profile   how much of the load do the densest x% of tool_out tokens carry
Q2  proxy vs causal   retention curves: at budget B, how much of tool_out's contribution does
                      each ranker recover?  Recovery = (lo_B - lo_frame) / (lo_full - lo_frame),
                      so 1.0 = keeping everything, 0.0 = keeping none of tool_out.
Q3  stratification    stated vs unstated (and prior-saturated items excluded)
Q4  cross-question    Spearman rho between two same-type follow-ups' density vectors.  This is the
                      one that decides whether ANY query-free evictor can work.
Q5  additivity        sum_u evict(u) vs evict(whole tool_out).  If the sum is far below the whole,
                      units are REDUNDANT and greedy per-unit eviction understates the loss.

Plus two free checks: measured density vs HotpotQA's own supporting_facts labels, and the
concentration of load in sentences that literally contain the probed value.

Items are independent HotpotQA questions (not turns of one conversation), so the resampling unit is
the item and a plain percentile bootstrap is calibrated here -- unlike the 10-cluster LoCoMo runs
where it was not (measured: at G=10 a percentile cluster bootstrap rejects a true null 11% of the time at a nominal 5%).
"""
import json, math, random, sys, statistics as st
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "dens.json"
SAT = 3.0          # |lo_none| above this = the model already knows it; evict is ~0 by construction
MINDEN = 2.0       # a retention RATIO needs a real denominator: items where tool_out
                   # contributes < 2 nats are dropped from Q2 (and counted)
B = 4000
RANKERS = ("oracle", "attn_h2o", "attn_snap", "attn_traj", "bm25_now", "bm25_q1", "idf", "recency", "random")
QF = {"attn_traj", "bm25_q1", "idf", "recency", "random"}
BUDS = (5, 10, 20, 40)
D = json.load(open(PATH))
R = D["rows"]
print("n=%d  skip=%s" % (len(R), D.get("skip")))


def boot(xs, f=st.fmean, seed=1234):
    if len(xs) < 4:
        return (None, None)
    rnd = random.Random(seed)
    n, out = len(xs), []
    for _ in range(B):
        out.append(f([xs[rnd.randrange(n)] for _ in range(n)]))
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def line(name, xs, extra=""):
    if not xs:
        print("  %-22s n=0" % name); return
    lo, hi = boot(xs)
    ci = "[%+.3f,%+.3f]" % (lo, hi) if lo is not None else "[n<4]"
    print("  %-22s n=%-4d mean %+8.3f  median %+8.3f  %s %s"
          % (name, len(xs), st.fmean(xs), st.median(xs), ci, extra))


for KIND in ("stated", "unstated"):
    rows = [r for r in R if (KIND + "_full") in r]
    live = [r for r in rows if abs(r.get(KIND + "_none", 0.0)) < SAT]
    print("\n" + "=" * 92)
    print("%s  n=%d  (prior unsaturated |none|<%.0f: n=%d)" % (KIND, len(rows), SAT, len(live)))
    if not rows:
        continue
    print("  levels: full %+.2f  frame(drop the whole tool_out) %+.2f  none %+.2f"
          % (st.fmean(r[KIND + "_full"] for r in rows),
             st.fmean(r[KIND + "_frame"] for r in rows),
             st.fmean(r.get(KIND + "_none", 0.0) for r in rows)))

    # ---------- Q5 additivity + Q1 profile ----------
    for g in ("passage", "sentence", "block"):
        us = [(r, r.get("%s_%s" % (KIND, g))) for r in live]
        us = [(r, u) for r, u in us if u]
        if not us:
            continue
        whole = [r[KIND + "_full"] - r[KIND + "_frame"] for r, _ in us]
        summed = [sum(x["evict"] for x in u) for _, u in us]
        rat = [s / w for (s, w) in zip(summed, whole) if abs(w) > 1.0]
        print("\n  --- %s granularity (%.1f units/item, %.0f tok/unit) ---"
              % (g, st.fmean(len(u) for _, u in us),
                 st.fmean(st.fmean(x["n"] for x in u) for _, u in us)))
        print("    Q5 additivity Σevict(u) / evict(whole bucket): median %.2f   (<<1 = redundancy; greedy per-unit eviction underestimates the loss)"
              % (st.median(rat) if rat else float("nan")))
        # Q1 Lorenz: order units by evict desc, cumulative evict share at cumulative token share
        prof = defaultdict(list)
        for _, u in us:
            tot_e = sum(max(0.0, x["evict"]) for x in u)
            tot_t = sum(x["n"] for x in u)
            if tot_e <= 0 or tot_t <= 0:
                continue
            o = sorted(u, key=lambda x: -x["evict"] / x["n"])
            ce = ct = 0.0
            marks = {}
            for x in o:
                ce += max(0.0, x["evict"]); ct += x["n"]
                for b in BUDS:
                    if b not in marks and ct / tot_t >= b / 100.0:
                        marks[b] = ce / tot_e
            for b in BUDS:
                prof[b].append(marks.get(b, 1.0))
        if prof:
            print("    Q1 density profile: " + "  ".join(
                "densest %d%% of tokens carry %.0f%%" % (b, 100 * st.median(prof[b])) for b in BUDS))
        # free checks (sentence only)
        if g == "sentence":
            hg = [x["evict"] for _, u in us for x in u if x["has_gold"]]
            ot = [x["evict"] for _, u in us for x in u if not x["has_gold"]]
            dh = [x["evict"] / x["n"] for _, u in us for x in u if x["has_gold"]]
            do = [x["evict"] / x["n"] for _, u in us for x in u if not x["has_gold"]]
            print("    sentence with the asked value vs rest:  evict %+.2f (n=%d) vs %+.2f (n=%d)   density/100tok %+.2f vs %+.2f"
                  % (st.fmean(hg) if hg else 0, len(hg), st.fmean(ot) if ot else 0, len(ot),
                     100 * st.fmean(dh) if dh else 0, 100 * st.fmean(do) if do else 0))
            sp = [x["evict"] for _, u in us for x in u if x.get("is_sup")]
            ns = [x["evict"] for _, u in us for x in u if x.get("is_sup") == 0]
            if sp and ns and KIND == "stated":
                # AUC of "official supporting fact" as a predictor of measured evict
                pair = tot = 0
                for a in sp:
                    for b2 in ns:
                        tot += 1; pair += 1 if a > b2 else (0.5 if a == b2 else 0)
                print("    official supporting_facts sentences vs rest: evict %+.2f (n=%d) vs %+.2f (n=%d)  AUC %.3f"
                      % (st.fmean(sp), len(sp), st.fmean(ns), len(ns), pair / max(1, tot)))

    # ---------- Q2 retention curves ----------
    print("\n  --- Q2 retention curve (recovery = (lo_B - frame)/(full - frame); 1.0 = keep all of tool_out) ---")
    print("    Note: `oracle` is the greedy order of **leave-one-out density**, not a true ceiling -- if Q5 shows redundancy across units (Σ << whole bucket),")
    print("          leave-one-out systematically underestimates each unit's importance, so another order can beat it (recovery>1 can occur).")
    print("    %-11s %-9s %s" % ("ranker", "ruler", "  ".join("B=%d%%" % b for b in BUDS)))
    dropped = sum(1 for r in live if abs(r[KIND + "_full"] - r[KIND + "_frame"]) <= MINDEN)
    print("    (%d/%d items excluded from the ratio because tool_out contributes < %.0f nats -- normal for the stated layer: the answer carries the value itself)"
          % (dropped, len(live), MINDEN))
    base = {}
    for rk in RANKERS:
        cells, keep = [], {}
        for b in BUDS:
            xs = []
            for r in live:
                k = "ret_%s_%s_%d" % (KIND, rk, b)
                den = r[KIND + "_full"] - r[KIND + "_frame"]
                if k in r and abs(den) > MINDEN:
                    xs.append((r[k] - r[KIND + "_frame"]) / den)
            keep[b] = xs
            cells.append("%.3f" % st.fmean(xs) if xs else "  na ")
        base[rk] = keep
        print("    %-11s %-9s %s   n=%d" % (rk, "query-free" if rk in QF else "query-aware",
                                            "  ".join("%6s" % c for c in cells),
                                            len(keep[BUDS[0]])))
    fr = ["B=%d%% %s" % (b, ("%.0f%%" % (100 * sum(1 for x in base["oracle"][b] if x >= 1.0)
                                                / len(base["oracle"][b]))) if base["oracle"][b] else "na")
          for b in BUDS]
    print("    share where pruning is free (oracle recovery >= 1.0: keeping the densest B%% is no worse than keeping all): " + "  ".join(fr))
    # paired vs random, at each budget
    print("    paired per item vs random:")
    for rk in RANKERS:
        if rk == "random":
            continue
        cells = []
        for b in BUDS:
            pa = []
            for r in live:
                ka, kb = "ret_%s_%s_%d" % (KIND, rk, b), "ret_%s_random_%d" % (KIND, b)
                den = r[KIND + "_full"] - r[KIND + "_frame"]
                if ka in r and kb in r and abs(den) > MINDEN:
                    pa.append((r[ka] - r[kb]) / den)
            if pa:
                lo, hi = boot(pa)
                cells.append("%+.3f%s" % (st.fmean(pa), "*" if lo is not None and lo * hi > 0 else " "))
            else:
                cells.append("  na  ")
        print("    %-11s %-9s %s" % (rk, "", "  ".join("%7s" % c for c in cells)))

# ---------- Q4 cross-question stability ----------
print("\n" + "=" * 92)
print("Q4 stability across follow-ups (Spearman rho of sentence-level density vectors)")
for k, lab in (("rho_unstated_unstated2", "two unstated follow-ups (same type, different targets)"),
               ("rho_stated_unstated", "stated vs unstated (across types)")):
    xs = [r[k] for r in R if r.get(k) is not None]
    if xs:
        pos = sum(1 for x in xs if x > 0)
        line(lab, xs, "positive %d/%d" % (pos, len(xs)))
    else:
        print("  %-22s n=0" % lab)
print("  Reading: rho ≈ 0 or negative => whether a unit is load-bearing is QUERY-SPECIFIC => no evictor that has not seen the next question")
print("        can produce this order; high rho => a question-independent information density exists, and query-free eviction is feasible.")

na = sum(1 for r in R if not r.get("attn_traj_ok"))
print("\nitems without attn_traj: %d/%d" % (na, len(R)))
