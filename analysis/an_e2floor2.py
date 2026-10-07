import os
import re, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kvmemory import kv_e2stats as K
PAT = re.compile(r"^\[e2\] (\S+) q(\d+) n(\d) t(\d) i(\d) s(\d) h(\d) \| pi(\d) ps(\d) ph(\d) \|")
rows, seen = [], set()
for line in open("e2floor_full.log", encoding="utf-8", errors="replace"):
    m = PAT.match(line)
    if not m: continue
    c, qi, n, t, i, s, h, pi, ps, ph = m.groups()
    if (c, qi) in seen: continue
    seen.add((c, qi))
    rows.append(dict(conv=c, qi=int(qi), j_none=int(n), j_iso=int(i), j_self=int(s),
                     j_harv=int(h), j_p_iso=int(pi), j_p_self=int(ps), j_p_harv=int(ph)))
rng = np.random.default_rng(1234)
def rep(rs, hi, lo, label):
    g = {}
    for r in rs: g.setdefault(r["conv"], []).append(r[hi] - r[lo])
    rr = K.report(label, g, rng, quiet=True)
    print("    %-40s n=%4d %+0.4f  p=%.4f  CI [%+.4f,%+.4f]"
          % (label, sum(len(v) for v in g.values()), rr["point"], rr["p_wcr"],
             rr["ci_wcr"][0], rr["ci_wcr"][1]))
    return rr["point"]

print("ROBUSTNESS: conversation 9 is truncated (57 of its 130 items) because the worker was")
print("reclaimed mid-run.  Drop it entirely and see whether the split survives.\n")
for tag, rs in (("all 10 conversations (conv9 partial)", rows),
                ("drop conv 9 -> 9 clean conversations", [r for r in rows if r["conv"] != "9"])):
    print("  %s  n=%d, G=%d" % (tag, len(rs), len({r["conv"] for r in rs})))
    a = rep(rs, "j_harv", "j_iso", "harv - iso   (published, = sum of below)")
    b = rep(rs, "j_self", "j_iso", "self - iso   path, not about the source")
    c = rep(rs, "j_harv", "j_self", "harv - self  THE conversation channel")
    print("    identity check: %+.4f + %+.4f = %+.4f   vs   %+.4f\n" % (b, c, b + c, a))

# floor-vs-none: is `self` even doing anything at all relative to serving nothing?
print("  sanity: does the served text help at all?")
rep(rows, "j_iso", "j_none", "iso - none")
rep(rows, "j_harv", "j_none", "harv - none")
