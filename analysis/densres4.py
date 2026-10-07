"""Q2c: why do the query-free **lexical** rankers (bm25_q1 / idf) do worse than random?

In the main table bm25_q1 is -0.110* and idf -0.101*, and the position stratification (densres3.py) shows both **hold in both strata**,
so position does not cause it. Here the mechanisms are ruled out one by one.
"""
import json, re, statistics as st, collections, sys

R = json.load(open(sys.argv[1]))["rows"]
STOP = set("the a an of in on at to for and or is are was were be been which who what "
           "that this these those with by from as it its his her their he she they".split())
cw = lambda t: {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2}
K = "unstated"
sat = [r for r in R if r.get(K + "_full") is not None and abs(r.get(K + "_none", 0)) < 3
       and abs(r[K + "_full"] - r[K + "_frame"]) > 2]
rec = lambda r, rk, b: ((r["ret_%s_%s_%d" % (K, rk, b)] - r[K + "_frame"])
                        / (r[K + "_full"] - r[K + "_frame"]))
print("n=%d\n" % len(sat) + "=" * 88)

print("H1  do they spend the budget on the wrong passage? -- no")
rk_g, nseg, top1 = [], [], 0
for r in sat:
    P = r[K + "_passage"]
    g = [i for i, p in enumerate(P) if p["has_gold"]]
    if len(P) < 2 or not g:
        continue
    q = cw(r["q1"])
    order = sorted(range(len(P)), key=lambda i: -len(q & cw(P[i]["title"] or "")))
    rk_g.append(order.index(g[0]) + 1); nseg.append(len(P)); top1 += (order[0] == g[0])
print("  probed passage ranked first by word overlap with q1: %d/%d = %.1f%% (random expectation %.1f%%)"
      % (top1, len(rk_g), 100 * top1 / len(rk_g), 100 / st.fmean(nseg)))
print("  => at the passage level they **point the right way**, more than twice as well as random. The direction is not the problem.\n" + "=" * 88)

print("H2  'random gets partial credit, concentrated rankers are all-or-nothing'? -- no, all three are all-or-nothing")
print("  %-10s %-5s %7s %7s %7s %7s" % ("ranker", "B", "mean", "median", "<0.1", "≥0.8"))
for b in (10, 20):
    for rk in ("random", "bm25_q1", "idf", "attn_h2o"):
        x = [rec(r, rk, b) for r in sat]
        print("  %-10s %-5s %7.3f %7.3f %6.0f%% %6.0f%%"
              % (rk, "%d%%" % b, st.fmean(x), st.median(x),
                 100 * sum(v < .1 for v in x) / len(x), 100 * sum(v >= .8 for v in x) / len(x)))
print("  => all medians are 0.01-0.09. The gap is **only in hit rate**: at B=10% random hits 12%, the two lexical arms only 5%.")
print("     So what needs explaining is why the hit rate is under half of uniform sampling.\n" + "=" * 88)

print("H3  is the target systematically missed at block level? -- consistent with the data, but **not directly verified** (dens.json does not store block text)")
c = collections.Counter(u["sent_id"] for r in sat for u in r[K + "_sentence"] if u["has_gold"])
tot = sum(c.values())
print("  sent_id distribution of the sentence holding the asked value: " + "  ".join("%s:%.0f%%" % (k, 100 * c[k] / tot)
                                                for k in sorted(c, key=lambda x: (x is None, x))[:4]))
ln = [u["n"] for r in sat for u in r[K + "_sentence"] if u["has_gold"]]
print("  median value-sentence length %d tok ≈ %.1f 16-token blocks" % (st.median(ln), st.median(ln) / 16))
print("""  By construction kv_dens.py:479 `YEAR.findall(seg.split(". ")[0])` requires the year to be in the **first sentence** => 75% at sent_id=0,
     i.e. the Wikipedia-style definitional sentence "NAME (born DATE) is a ...". That sentence spans 2-3 16-token blocks,
     and **the entity-name block is not the date block**. bm25_q1 matches the entity name in the question and idf prefers rare proper nouns --
     both are pulled to the **name block**, systematically one block off the target.
     Verifying this directly needs the block text, which dens.json does not store => rerun, or re-parse traj.json.""")
print("=" * 88)
print("Conclusion: the two lexical arms losing to random is a **real effect** (it holds in both position strata); the mechanism **points to a block-level mismatch** but is not directly verified.")
