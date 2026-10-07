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
    rr = K.report(label, g, rng, quiet=True)
    pos = sum(1 for x in flat if x > 0); neg = sum(1 for x in flat if x < 0)
    m, k = pos+neg, min(pos, neg)
    pv = 2.0*sum(comb(m, i) for i in range(k+1))/(2.0**m) if m else 1.0
    print("    %-30s n=%4d %+8.4f p=%.4f CI[%+.4f,%+.4f] %d+/%d- sp=%.4f"
          % (label, len(flat), rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1],
             pos, neg, min(pv,1.0)))

print("The two readouts are DIFFERENT TASKS on the same served rows:")
print("  judged   = free 24-token generation, scored by a judge   (recall)")
print("  log-odds = 2-option forced choice, gold vs a hard negative (recognition)")
print("Compare them on the SAME 599 items so the comparison is apples to apples.\n")
print("  === judged accuracy, the 599 items with a readout ===")
for hi, lo, lab in (("j_harv","j_iso","harv - iso"), ("j_self","j_iso","self - iso"),
                    ("j_harv","j_self","harv - self"),
                    ("j_p_self","j_p_iso","p_self - p_iso (placebo)")):
    stat(HAVE, hi, lo, lab)
print("  === log-odds, the same 599 items ===")
for hi, lo, lab in (("r_harv_lo","r_iso_lo","harv - iso"), ("r_self_lo","r_iso_lo","self - iso"),
                    ("r_harv_lo","r_self_lo","harv - self"),
                    ("r_p_self_lo","r_p_iso_lo","p_self - p_iso (placebo)")):
    stat(HAVE, hi, lo, lab)

print("\n  === do they agree per item? (self - iso) ===")
both = [(r["j_self"]-r["j_iso"], r["r_self_lo"]-r["r_iso_lo"]) for r in HAVE]
opp = sum(1 for a,b in both if a*b < 0)
tie = sum(1 for a,b in both if a == 0)
print("    n=%d  judged is 0 on %d (%.0f%%)  opposite sign on %d"
      % (len(both), tie, 100.0*tie/len(both), opp))
gen = [b for a,b in both if a == 0]
print("    on the %d items the judge cannot separate, log-odds mean = %+.4f"
      % (len(gen), sum(gen)/len(gen)))
