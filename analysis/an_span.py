import os
import json, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kvmemory import kv_e2stats as K
R = json.load(open("e2_ro.json"))["rows"]
S = {(str(x["conv"]), int(x["qi"])): x for x in json.load(open("e2_sets.json"))["items"]}
HAVE = [r for r in R if r.get("r_self_lo") is not None]
rng = np.random.default_rng(1234)

for r in HAVE:
    it = S.get((str(r["conv"]), int(r["qi"])))
    srv = sorted(it["serve"])
    r["_span"] = srv[-1] - srv[0]
    r["_gapmed"] = st.median([b - a for a, b in zip(srv, srv[1:])])
    r["_nadj"] = sum(1 for a, b in zip(srv, srv[1:]) if b - a == 1)
    r["_nt"] = it["n_turns"]
    r["_d"] = r["r_self_lo"] - r["r_iso_lo"]
    r["_dp"] = (r["r_self_lo"] - r["r_iso_lo"]) - (r["r_p_self_lo"] - r["r_p_iso_lo"])

def band(rows, key, edges, label):
    print("  %s" % label)
    prev = -1
    for e in edges + [10**9]:
        sub = [r for r in rows if prev < r[key] <= e]
        if len(sub) < 40:
            prev = e; continue
        g, gp = {}, {}
        for r in sub:
            g.setdefault(str(r["conv"]), []).append(r["_d"])
            gp.setdefault(str(r["conv"]), []).append(r["_dp"])
        rr = K.report("x", g, rng, quiet=True)
        rp = K.report("x", gp, rng, quiet=True) if len(gp) > 1 else None
        print("    %-22s n=%4d  self-iso %+7.4f (p=%.3f)   minus placebo %+7.4f (p=%.3f)"
              % ("%s in (%d,%s]" % (key, prev, e if e < 10**8 else "inf"), len(sub),
                 rr["point"], rr["p_wcr"],
                 rp["point"] if rp else float("nan"), rp["p_wcr"] if rp else float("nan")))
        prev = e

print("Does the joint-encoding gain depend on how the served rows sit in the conversation?")
print("n=%d items with a readout\n" % len(HAVE))
band(HAVE, "_span", [50, 150, 300], "by SPAN (last served row index - first)")
print()
band(HAVE, "_nadj", [0, 1], "by number of ADJACENT served pairs")
print()
sp = [r["_span"] for r in HAVE]
print("  span distribution: min=%d p25=%d med=%d p75=%d max=%d (conversation length median %d)"
      % (min(sp), sorted(sp)[len(sp)//4], sorted(sp)[len(sp)//2],
         sorted(sp)[3*len(sp)//4], max(sp), st.median([r["_nt"] for r in HAVE])))
c = np.corrcoef([r["_span"] for r in HAVE], [r["_d"] for r in HAVE])[0, 1]
print("  Pearson r(span, self-iso) = %+.4f   (n=%d)" % (c, len(HAVE)))
