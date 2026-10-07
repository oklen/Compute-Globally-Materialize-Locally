import os
import re, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kvmemory import kv_e2stats as K

PAT = re.compile(r"^\[e2\] (\S+) q(\d+) n(\d) t(\d) i(\d) s(\d) h(\d) \| pi(\d) ps(\d) ph(\d) \|")
rows = []
for line in open("e2floor_full.log", encoding="utf-8", errors="replace"):
    m = PAT.match(line)
    if m:
        c, qi, n, t, i, s, h, pi, ps, ph = m.groups()
        rows.append(dict(conv=c, qi=int(qi), j_none=int(n), j_text=int(t), j_iso=int(i),
                         j_self=int(s), j_harv=int(h), j_p_iso=int(pi), j_p_self=int(ps),
                         j_p_harv=int(ph)))
# de-dup: the smoke run wrote 2 rows of conv 0 into a different file, but guard anyway
seen, uniq = set(), []
for r in rows:
    k = (r["conv"], r["qi"])
    if k not in seen:
        seen.add(k); uniq.append(r)
rows = uniq
print("recovered %d items from the log, %d conversations" % (len(rows), len({r["conv"] for r in rows})))
per = {}
for r in rows:
    per[r["conv"]] = per.get(r["conv"], 0) + 1
print("per conversation:", dict(sorted(per.items(), key=lambda x: int(x[0]))))

# attach category from the published E2 output
CAT = {}
for r in json.load(open("e2_rejudged.json"))["rows"]:
    CAT[(str(r["conv"]), int(r["qi"]))] = str(r.get("cat"))
for r in rows:
    r["cat"] = CAT.get((r["conv"], r["qi"]))
print("category attached for %d/%d" % (sum(1 for r in rows if r["cat"]), len(rows)))

rng = np.random.default_rng(1234)
def rep(rows, hi, lo, label):
    g = {}
    for r in rows:
        g.setdefault(r["conv"], []).append(r[hi] - r[lo])
    n = sum(len(v) for v in g.values())
    rr = K.report(label, g, rng, quiet=True)
    print("  %-42s n=%4d  %+0.4f   WCR p=%.4f   CI [%+.4f,%+.4f]"
          % (label, n, rr["point"], rr["p_wcr"], rr["ci_wcr"][0], rr["ci_wcr"][1]))
    return rr

for tag, sub in (("ALL", rows),
                 ("cat1 multi-hop", [r for r in rows if r["cat"] == "1"]),
                 ("cat4 single-hop", [r for r in rows if r["cat"] == "4"])):
    if len(sub) < 60:
        continue
    print("\n================ %s  (n=%d) ================" % (tag, len(sub)))
    accs = {a: sum(r["j_" + a] for r in sub) / len(sub)
            for a in ("none", "text", "iso", "self", "harv", "p_iso", "p_self", "p_harv")}
    print("  ACC " + "  ".join("%s=%.3f" % (k, v) for k, v in accs.items()))
    print("  --- published PRIMARY (sums the two channels) ---")
    rep(sub, "j_harv", "j_iso", "harv - iso                 (published)")
    rep(sub, "j_p_harv", "j_p_iso", "p_harv - p_iso             (placebo)")
    print("  --- the split ---")
    rep(sub, "j_self", "j_iso", "self - iso    path + rows-see-each-other")
    rep(sub, "j_p_self", "j_p_iso", "p_self - p_iso             (placebo)")
    rep(sub, "j_harv", "j_self", "harv - self   THE conversation channel")
    rep(sub, "j_p_harv", "j_p_self", "p_harv - p_self            (placebo)")
