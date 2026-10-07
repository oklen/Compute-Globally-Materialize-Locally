import os
import json, math, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kvmemory import kv_e2stats as K

NEW = json.load(open("e3_rank12.json"))["rows"]
OLD = json.load(open("e3_rep12.json"))["rows"]
print("n new=%d  old=%d" % (len(NEW), len(OLD)))

key = lambda r: (r["ds"], r["conv"], r["q"])
o = {key(r): r for r in OLD}
pairs = [(r, o[key(r)]) for r in NEW if key(r) in o]
print("matched items: %d" % len(pairs))

# ---------- 1. determinism: the published columns must be bit-for-bit ----------
cols = [c for c in OLD[0] if c.endswith("_p_gold")] + ["neg", "gold_is_a", "sel_p_gold",
                                                       "cf_gold_residual"]
diff = {}
for n, ol in pairs:
    for c in cols:
        if c in n and c in ol and n[c] != ol[c]:
            diff.setdefault(c, 0)
            diff[c] += 1
print("REPRODUCTION: %s" % ("bit-for-bit on all %d columns" % len(cols) if not diff
                            else "MISMATCH " + str(diff)))

# ---------- 2. what the log-odds recovers ----------
rail = [r for r in NEW if r.get("q_both_railed")]
z_dp = [r for r in NEW if r.get("q_dp") == 0.0]
rescued = [r for r in z_dp if abs(r.get("q_dlo", 0.0)) > 1e-6]
print("\nboth-railed items: %d   exactly-zero q_dp: %d   of those, nonzero q_dlo: %d"
      % (len(rail), len(z_dp), len(rescued)))
if rescued:
    v = sorted(abs(r["q_dlo"]) for r in rescued)
    print("   |q_dlo| on rescued: min=%.3f median=%.3f max=%.3f"
          % (v[0], v[len(v)//2], v[-1]))

# ---------- 3. PRIMARY under both scales ----------
def by_conv(field_hi, field_lo):
    g = {}
    for r in NEW:
        if field_hi not in r or field_lo not in r:
            continue
        g.setdefault(str(r["conv"]), []).append(r[field_hi] - r[field_lo])
    return g

rng = np.random.default_rng(1234)
print("\n%-34s %10s %10s %10s %22s" % ("estimand", "point", "SE", "WCR p", "WCR 95% CI"))
for name, hi, lo in (("carrier  orig-cf   (p_gold)", "q_orig_p_gold", "q_cf_p_gold"),
                     ("placebo  orig-cf   (p_gold)", "p_orig_p_gold", "p_cf_p_gold"),
                     ("carrier  orig-cf   (log-odds)", "q_orig_lo", "q_cf_lo"),
                     ("placebo  orig-cf   (log-odds)", "p_orig_lo", "p_cf_lo")):
    g = by_conv(hi, lo)
    r = K.report(name, g, rng, quiet=True)
    print("%-34s %+10.4f %10.4f %10.4f  [%+.4f,%+.4f]"
          % (name, r["point"], r["se"], r["p_wcr"], r["ci_wcr"][0], r["ci_wcr"][1]))

# PRIMARY = carrier - placebo, paired within item
for tag, qf, pf in (("p_gold", "q_dp", "p_dp"), ("log-odds", "q_dlo", "p_dlo")):
    g = {}
    for r in NEW:
        if qf in r and pf in r:
            g.setdefault(str(r["conv"]), []).append(r[qf] - r[pf])
    n = sum(len(v) for v in g.values())
    rr = K.report("PRIMARY " + tag, g, rng, quiet=True)
    flat = [x for v in g.values() for x in v]
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    ties = sum(1 for x in flat if x == 0)
    # exact two-sided sign test
    from math import comb
    m = pos + neg
    k = min(pos, neg)
    pv = 2.0 * sum(comb(m, i) for i in range(k + 1)) / (2.0 ** m) if m else 1.0
    med = sorted(flat)[len(flat)//2]
    tr = sorted(flat)[5:-5]
    print("\nPRIMARY (carrier - placebo) on %s: n=%d  point=%+.4f  WCR p=%.4f  CI [%+.4f,%+.4f]"
          % (tag, n, rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1]))
    print("   median=%+.4f  trimmed(5 each end)=%+.4f  sign test %d+/%d- (%d ties) p=%.4f"
          % (med, sum(tr)/len(tr), pos, neg, ties, min(pv, 1.0)))
