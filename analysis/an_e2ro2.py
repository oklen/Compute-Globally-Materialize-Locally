import os
import json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from math import comb
from kvmemory import kv_e2stats as K
R = json.load(open("e2_ro.json"))["rows"]
HAVE = [r for r in R if r.get("r_harv_lo") is not None]
rng = np.random.default_rng(1234)

def stat(rows, hi, lo, label):
    g = {}
    for r in rows:
        if r.get(hi) is None or r.get(lo) is None: continue
        g.setdefault(str(r["conv"]), []).append(r[hi] - r[lo])
    flat = [x for v in g.values() for x in v]
    if len(g) < 2: return None
    rr = K.report(label, g, rng, quiet=True)
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    m, k = pos+neg, min(pos, neg)
    pv = 2.0*sum(comb(m, i) for i in range(k+1))/(2.0**m) if m else 1.0
    s = sorted(flat); tr = s[len(s)//20: len(s)-len(s)//20]
    print("    %-36s n=%4d %+8.4f p=%.4f CI[%+.4f,%+.4f] med=%+.3f trim5%%=%+.3f %d+/%d- sp=%.4f"
          % (label, len(flat), rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1],
             s[len(s)//2], sum(tr)/len(tr), pos, neg, min(pv,1.0)))
    return rr

print("=== CONTROL: is the sign flip the SCALE or the SUBSET? ===")
print("  the forced-choice readout exists only where the item had a usable negative:")
print("  %d of %d items (%.0f%%).  Compare judged accuracy on BOTH sets.\n" % (len(HAVE), len(R), 100.0*len(HAVE)/len(R)))
print("  judged accuracy, ALL 1269 items:")
stat(R, "j_harv", "j_iso", "harv - iso")
stat(R, "j_p_harv", "j_p_iso", "p_harv - p_iso (placebo)")
print("  judged accuracy, ONLY the %d with a readout:" % len(HAVE))
stat(HAVE, "j_harv", "j_iso", "harv - iso")
stat(HAVE, "j_p_harv", "j_p_iso", "p_harv - p_iso (placebo)")
print("  log-odds, same %d items:" % len(HAVE))
stat(HAVE, "r_harv_lo", "r_iso_lo", "harv - iso")
stat(HAVE, "r_p_harv_lo", "r_p_iso_lo", "p_harv - p_iso (placebo)")

print("\n=== state-specific part: (real - placebo), paired within item ===")
for tag, hi, lo, phi, plo in (("harv - iso", "r_harv_lo", "r_iso_lo", "r_p_harv_lo", "r_p_iso_lo"),
                              ("self - iso", "r_self_lo", "r_iso_lo", "r_p_self_lo", "r_p_iso_lo"),
                              ("harv - self", "r_harv_lo", "r_self_lo", "r_p_harv_lo", "r_p_self_lo")):
    g = {}
    for r in HAVE:
        if all(r.get(k) is not None for k in (hi, lo, phi, plo)):
            g.setdefault(str(r["conv"]), []).append((r[hi]-r[lo]) - (r[phi]-r[plo]))
    flat = [x for v in g.values() for x in v]
    rr = K.report(tag, g, rng, quiet=True)
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    m, k = pos+neg, min(pos, neg)
    pv = 2.0*sum(comb(m, i) for i in range(k+1))/(2.0**m) if m else 1.0
    print("    %-36s n=%4d %+8.4f p=%.4f CI[%+.4f,%+.4f] %d+/%d- sp=%.4f"
          % (tag+" minus its placebo", len(flat), rr["point"], rr["p_wcr"],
             rr["ci_wcr"][0], rr["ci_wcr"][1], pos, neg, min(pv,1.0)))

print("\n=== leave-one-conversation-out, log-odds harv - iso ===")
convs = sorted({str(r["conv"]) for r in HAVE})
vals = []
for c in convs:
    sub = [r for r in HAVE if str(r["conv"]) != c]
    g = {}
    for r in sub: g.setdefault(str(r["conv"]), []).append(r["r_harv_lo"] - r["r_iso_lo"])
    vals.append(K.report("x", g, rng, quiet=True)["point"])
print("    range [%+.4f, %+.4f]  all positive: %s" % (min(vals), max(vals), all(v > 0 for v in vals)))
