"""X2 supplementary reading: score meaning pairs by global polarity (each object's A set is mostly normal / positive, its B set mostly abnormal / negative).

Scoring by each object's own state set is too strict: the harvest arm often answers another object's state word (PASSED, DOWN, ...), which that set leaves undecidable.
Here all objects' A words form a positive list and all B words a negative list (majority vote; LOCKED/UNLOCKED is ambiguous, so database is dropped),
an answer takes the polarity of the list it falls in, and pairs are then counted by direction. Usage: python3 x2_polarity.py qwen3.json [...]
"""
import collections
import json
import math
import sys


def bp(k, n):
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


for path in sys.argv[1:]:
    d = json.load(open(path))
    rows = d["rows"]
    votes = collections.Counter()
    for r in rows:
        if r["obj"] == "database":
            continue
        for w in r["A"]:
            votes[(w, "+")] += 1
        for w in r["B"]:
            votes[(w, "-")] += 1
    pos = {w for (w, s) in votes if s == "+" and votes[(w, "+")] > votes.get((w, "-"), 0)}
    neg = {w for (w, s) in votes if s == "-" and votes[(w, "-")] > votes.get((w, "+"), 0)}

    def pol(w):
        return "+" if w in pos else ("-" if w in neg else None)

    print(f"== {d['tag']}")
    for arm in ("harv", "full"):
        f = a = same = oth = 0
        for r in rows:
            if r["kind"] != "meaning" or r["obj"] == "database":
                continue
            p1, p2 = pol(r[f"w_{arm}1"]), pol(r[f"w_{arm}2"])
            t1 = "+" if r["d1"] in r["A"] else "-"
            t2 = "+" if r["d2"] in r["A"] else "-"
            if p1 is None or p2 is None:
                oth += 1
            elif p1 == p2:
                same += 1
            elif (p1, p2) == (t1, t2):
                f += 1
            else:
                a += 1
        print(f"  {arm}: meaning pairs (global polarity) follow:anti = {f}:{a}  p={bp(f, f + a):.1e} | same polarity {same} | undecidable {oth}")
    c = collections.Counter((r["w_harv1"], r["w_harv2"]) for r in rows if r["kind"] == "meaning")
    print("  harv answers in meaning pairs (top 10):", c.most_common(10))
