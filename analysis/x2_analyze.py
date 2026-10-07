"""X2 readout: paired direction counts (follow:anti) for meaning pairs and wording pairs, and the rate at which the donor word recurs.

Usage: python3 x2_analyze.py qwen3.json [ministral3.json gemma4.json ...]
Meaning pairs (the two donor words belong to opposite states): among pairs whose two answers differ in state, count 'each follows its own donor state' against 'reversed'.
Wording pairs (two words for the same state): among pairs whose two answers use different words, count 'each reproduces its own donor word' against 'crossed'.
full is the source-present ceiling (the wording should follow there); the two iso answers are identical and can never separate.
"""
import json
import math
import sys


def binom_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def state(word, A, B):
    return "A" if word in A else ("B" if word in B else None)


for path in sys.argv[1:]:
    d = json.load(open(path))
    rows = d["rows"]
    print(f"== {d['tag']}  ({len(rows)} pairs: {sum(r['kind']=='meaning' for r in rows)} meaning, {sum(r['kind']=='wording' for r in rows)} wording)")
    for arm in ("harv", "full"):
        # meaning pairs
        fol = anti = same = other = 0
        for r in rows:
            if r["kind"] != "meaning":
                continue
            A, B = set(r["A"]), set(r["B"])
            s1, s2 = state(r[f"w_{arm}1"], A, B), state(r[f"w_{arm}2"], A, B)
            t1, t2 = state(r["d1"], A, B), state(r["d2"], A, B)
            if s1 is None or s2 is None:
                other += 1
            elif s1 == s2:
                same += 1
            elif (s1, s2) == (t1, t2):
                fol += 1
            else:
                anti += 1
        # wording pairs
        wf = wa = wsame = wother = 0
        verb = tot = 0
        for r in rows:
            o1, o2 = r[f"w_{arm}1"], r[f"w_{arm}2"]
            if r["kind"] == "meaning":
                verb += (o1 == r["d1"]) + (o2 == r["d2"]); tot += 2
                continue
            if o1 == o2:
                wsame += 1
            elif (o1, o2) == (r["d1"], r["d2"]):
                wf += 1
            elif (o1, o2) == (r["d2"], r["d1"]):
                wa += 1
            else:
                wother += 1
        print(f"  {arm:4s} meaning follow:anti = {fol}:{anti} (p={binom_p(fol, fol + anti):.1e}), same state {same}, non-state answer {other}"
              f" | wording follow:anti = {wf}:{wa} (p={binom_p(wf, wf + wa):.2g}), same word {wsame}, other {wother}"
              f" | donor word recurs in meaning pairs {verb}/{tot}")
    iso_words = {}
    for r in rows:
        iso_words[r["w_iso"]] = iso_words.get(r["w_iso"], 0) + 1
    print("  iso default words (top 6):", sorted(iso_words.items(), key=lambda x: -x[1])[:6])
