import os
import json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from math import comb
from kvmemory import kv_e2stats as K

NEW = json.load(open("e3_rank12.json"))["rows"]
OLD = {(r["ds"], r["conv"], r["q"]): r for r in
       json.load(open("e3_rep12.json"))["rows"]}
rng = np.random.default_rng(1234)

# 1. the one mismatched column -- direction tells us whether it is the documented counter fix
d = [(r["cf_gold_residual"], OLD[(r["ds"], r["conv"], r["q"])]["cf_gold_residual"])
     for r in NEW if (r["ds"], r["conv"], r["q"]) in OLD]
worse = [(a, b) for a, b in d if a > b]
print("cf_gold_residual: %d differ; new>old in %d of them; examples %s"
      % (sum(1 for a, b in d if a != b), len(worse), [x for x in d if x[0] != x[1]][:5]))
print("clusters G = %d" % len({r["conv"] for r in NEW}))

def prim(rows, qf, pf):
    g = {}
    for r in rows:
        if qf in r and pf in r:
            g.setdefault(str(r["conv"]), []).append(r[qf] - r[pf])
    flat = [x for v in g.values() for x in v]
    rr = K.report("x", g, rng, quiet=True)
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    m = pos + neg; k = min(pos, neg)
    pv = 2.0 * sum(comb(m, i) for i in range(k + 1)) / (2.0 ** m) if m else 1.0
    return len(flat), rr, pos, neg, min(pv, 1.0)

print("\n--- is the log-odds result just the rescued items? ---")
railed = [r for r in NEW if r.get("q_both_railed")]
unr = [r for r in NEW if not r.get("q_both_railed")]
for tag, rows in (("ALL", NEW), ("drop both-railed (n=%d)" % len(unr), unr)):
    n, rr, pos, neg, pv = prim(rows, "q_dlo", "p_dlo")
    print("  %-28s n=%3d  point=%+.4f  WCR p=%.4f  CI [%+.4f,%+.4f]  sign %d+/%d- p=%.4f"
          % (tag, n, rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1], pos, neg, pv))

print("\n--- leave-one-conversation-out, log-odds PRIMARY ---")
convs = sorted({str(r["conv"]) for r in NEW})
vals = []
for c in convs:
    sub = [r for r in NEW if str(r["conv"]) != c]
    n, rr, *_ = prim(sub, "q_dlo", "p_dlo")
    vals.append(rr["point"])
print("  range [%+.4f, %+.4f] over %d conversations; all positive: %s"
      % (min(vals), max(vals), len(vals), all(v > 0 for v in vals)))

print("\n--- do the two scales agree per item? ---")
both = [(r["q_dp"] - r["p_dp"], r["q_dlo"] - r["p_dlo"]) for r in NEW
        if "q_dp" in r and "p_dp" in r and "q_dlo" in r and "p_dlo" in r]
agree = sum(1 for a, b in both if (a > 0) == (b > 0) or a == 0 or b == 0)
flip = [(a, b) for a, b in both if a * b < 0]
print("  n=%d  same sign or tied: %d   OPPOSITE sign: %d" % (len(both), agree, len(flip)))
print("  the log-odds scale is NOT a more powerful version of the same test -- it reweights:")
print("  it magnifies moves near the rails and shrinks moves near p=.5.")
