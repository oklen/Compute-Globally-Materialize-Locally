import os
import json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from math import comb
from kvmemory import kv_e2stats as K

R = json.load(open("e2_ro.json"))["rows"]
print("n=%d  conversations=%d" % (len(R), len({r["conv"] for r in R})))
have = [r for r in R if r.get("r_harv_lo") is not None]
print("with forced-choice readout: %d (%.0f%%)" % (len(have), 100.0*len(have)/len(R)))
rng = np.random.default_rng(1234)

def stat(rows, hi, lo, label, show=True):
    g = {}
    for r in rows:
        if r.get(hi) is None or r.get(lo) is None:
            continue
        g.setdefault(str(r["conv"]), []).append(r[hi] - r[lo])
    flat = [x for v in g.values() for x in v]
    if len(g) < 2 or not flat:
        return None
    rr = K.report(label, g, rng, quiet=True)
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    m, k = pos + neg, min(pos, neg)
    pv = 2.0 * sum(comb(m, i) for i in range(k + 1)) / (2.0 ** m) if m else 1.0
    med = sorted(flat)[len(flat) // 2]
    if show:
        print("    %-38s n=%4d  %+8.4f  p=%.4f  CI[%+.4f,%+.4f]  med=%+.3f  %d+/%d- signp=%.4f"
              % (label, len(flat), rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1],
                 med, pos, neg, min(pv, 1.0)))
    return rr

for tag, sub in (("ALL", R),
                 ("cat1 multi-hop", [r for r in R if str(r.get("cat")) == "1"]),
                 ("cat4 single-hop", [r for r in R if str(r.get("cat")) == "4"])):
    print("\n=============== %s  (n=%d) ===============" % (tag, len(sub)))
    print("  --- JUDGED ACCURACY (one bit per item) ---")
    stat(sub, "j_harv", "j_iso", "harv - iso      published PRIMARY")
    stat(sub, "j_p_harv", "j_p_iso", "p_harv - p_iso  placebo")
    stat(sub, "j_self", "j_iso", "self - iso      encode-regime channel")
    stat(sub, "j_harv", "j_self", "harv - self     CONVERSATION channel")
    print("  --- LOG-ODDS (same probe, unbounded scale) ---")
    stat(sub, "r_harv_lo", "r_iso_lo", "harv - iso      published PRIMARY")
    stat(sub, "r_p_harv_lo", "r_p_iso_lo", "p_harv - p_iso  placebo")
    stat(sub, "r_self_lo", "r_iso_lo", "self - iso      encode-regime channel")
    stat(sub, "r_p_self_lo", "r_p_iso_lo", "p_self - p_iso  placebo")
    stat(sub, "r_harv_lo", "r_self_lo", "harv - self     CONVERSATION channel")
    stat(sub, "r_p_harv_lo", "r_p_self_lo", "p_harv - p_self placebo")

print("\n=============== arm levels (log-odds, mean) ===============")
import statistics as st
for a in ("none", "iso", "self", "harv", "p_iso", "p_self", "p_harv"):
    v = [r["r_%s_lo" % a] for r in R if r.get("r_%s_lo" % a) is not None]
    w = [r["r_%s_p" % a] for r in R if r.get("r_%s_p" % a) is not None]
    print("  %-8s log-odds %+7.4f    p_gold %.4f    n=%d" % (a, st.fmean(v), st.fmean(w), len(v)))
