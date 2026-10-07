import os
import json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from math import comb
from kvmemory import kv_e2stats as K

D = json.load(open("e2_k.json"))
R, KS = D["rows"], D["ks"]
print("n=%d  conversations=%d  ks=%s" % (len(R), len({r["conv"] for r in R}), KS))
rng = np.random.default_rng(1234)

def stat(rows, f, label, show=True):
    g = {}
    for r in rows:
        v = f(r)
        if v is not None:
            g.setdefault(str(r["conv"]), []).append(v)
    flat = [x for v in g.values() for x in v]
    if len(g) < 2 or not flat:
        return None
    rr = K.report(label, g, rng, quiet=True)
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    m, k = pos + neg, min(pos, neg)
    pv = 2.0 * sum(comb(m, i) for i in range(k + 1)) / (2.0 ** m) if m else 1.0
    if show:
        print("    %-40s n=%4d %+8.4f p=%.4f CI[%+.4f,%+.4f] med=%+.3f %d+/%d- sp=%.4f"
              % (label, len(flat), rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1],
                 sorted(flat)[len(flat)//2], pos, neg, min(pv, 1.0)))
    rr["sign_p"] = min(pv, 1.0)
    return rr

def dif(k, hi, lo):
    a, b = "k%d_%s_lo" % (k, hi), "k%d_%s_lo" % (k, lo)
    return lambda r: (r[a] - r[b]) if a in r and b in r else None

def adj(k, hi, lo):
    a, b = "k%d_%s_lo" % (k, hi), "k%d_%s_lo" % (k, lo)
    pa, pb = "k%d_p_%s_lo" % (k, hi), "k%d_p_%s_lo" % (k, lo)
    def f(r):
        if all(x in r for x in (a, b, pa, pb)):
            return (r[a] - r[b]) - (r[pa] - r[pb])
        return None
    return f

print("\n########## NULL-BY-CONSTRUCTION CHECK ##########")
z = [abs(r["k1_self_lo"] - r["k1_iso_lo"]) for r in R if "k1_self_lo" in r]
zp = [abs(r["k1_p_self_lo"] - r["k1_p_iso_lo"]) for r in R if "k1_p_self_lo" in r]
print("  k=1, self and iso are the SAME computation -> difference must be identically 0")
print("  max |self-iso| = %.8f over %d items ; placebo %.8f over %d  -> %s"
      % (max(z), len(z), max(zp), len(zp),
         "PASS" if max(z) == 0 and max(zp) == 0 else "FAIL"))

print("\n########## THE POINT THAT MATTERS: k=1 ##########")
print("  At k=1 no rows can contextualise each other, so harv - iso is the PURE")
print("  'this one row was encoded inside the whole conversation vs encoded alone'.")
stat(R, dif(1, "harv", "iso"), "k=1  harv - iso   (conversation channel)")
stat(R, dif(1, "p_harv", "p_iso"), "k=1  placebo")
stat(R, adj(1, "harv", "iso"), "k=1  harv - iso  MINUS placebo")

print("\n########## THE SWEEP ##########")
for k in KS:
    print("  --- k = %d ---" % k)
    stat(R, dif(k, "harv", "iso"), "harv - iso    total")
    stat(R, dif(k, "self", "iso"), "self - iso    rows see each other")
    stat(R, dif(k, "harv", "self"), "harv - self   rest of the conversation")
    stat(R, adj(k, "harv", "iso"), "harv - iso    MINUS placebo")
    stat(R, adj(k, "self", "iso"), "self - iso    MINUS placebo")

print("\n########## does the joint-encoding gain scale with the budget? ##########")
print("  %-6s %12s %12s %12s" % ("k", "harv-iso", "self-iso", "harv-self"))
for k in KS:
    row = []
    for hi, lo in (("harv", "iso"), ("self", "iso"), ("harv", "self")):
        r = stat(R, dif(k, hi, lo), "x", show=False)
        row.append("%+.4f" % r["point"] if r else "  n/a ")
    print("  k=%-4d %12s %12s %12s" % (k, *row))
