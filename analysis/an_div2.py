import os
import json, glob, sys, re, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.locomo import load_locomo
from kvmemory.rA_roles import _utterance

rows = []
for p in sorted(glob.glob("e3div_*/e3div.*.json")):
    rows += json.load(open(p))["rows"]
seen, R = set(), []
for r in rows:
    k = (r["ds"], r["conv"], r["q"])
    if k not in seen:
        seen.add(k); R.append(r)
TX = {}
for cv in load_locomo("../data/locomo10.json"):
    TX[str(cv.cid)] = [_utterance(s) for s in cv.segments]
R = [r for r in R if str(r["conv"]) in TX]
print("locomo items usable for the text join: %d" % len(R))

STOP = set("the a an and or but if then of to in on at for with is are was were be been am i you "
           "he she it we they me my your his her our their that this these those do did does not "
           "no yes so as by from about him them us what when where who how why have has had will "
           "would can could just really very much more most some any all".split())
def words(s):
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2}

print("\n[5] SELECTIVITY -- the decisive control.")
print("    A diffuse perturbation moves every downstream turn by about the same amount.")
print("    Genuine absorption should move the turns that TALK ABOUT the fact more.\n")
buckets = {}
for r in R:
    txt = TX[str(r["conv"])]
    qw = words(r["q"]) | words(r["gold"])
    for t in r["turns"]:
        if not t["downstream"] or t["own_gold"] or t["j"] >= len(txt):
            continue
        ov = len(words(txt[t["j"]]) & qw)
        b = min(ov, 4)
        buckets.setdefault(b, []).append(t["rel"])
base = None
for b in sorted(buckets):
    v = buckets[b]
    if len(v) < 30: continue
    m = st.fmean(v)
    if base is None: base = m
    print("    shares %s content word(s) with question+gold: n=%6d  mean=%.4f  median=%.4f  x%.2f"
          % ("%d" % b if b < 4 else "4+", len(v), m, sorted(v)[len(v)//2], m / base))

# strongest possible version: turns that literally name the answer's topic vs turns that do not
hit = [x for b, v in buckets.items() if b >= 2 for x in v]
miss = [x for b, v in buckets.items() if b < 2 for x in v]
if hit and miss:
    print("\n    >=2 shared: n=%6d mean=%.4f      <2 shared: n=%6d mean=%.4f      ratio %.3f"
          % (len(hit), st.fmean(hit), len(miss), st.fmean(miss), st.fmean(hit)/st.fmean(miss)))

# and the same split restricted to FAR turns, where the local decay cannot explain anything
far_hit, far_miss = [], []
for r in R:
    fg = r["first_gold"]
    if fg is None: continue
    txt = TX[str(r["conv"])]
    qw = words(r["q"]) | words(r["gold"])
    for t in r["turns"]:
        if not t["downstream"] or t["own_gold"] or t["j"] >= len(txt) or (t["j"] - fg) <= 50:
            continue
        ov = len(words(txt[t["j"]]) & qw)
        (far_hit if ov >= 2 else far_miss).append(t["rel"])
if far_hit and far_miss:
    print("    beyond 50 turns from the change:  >=2 shared n=%5d mean=%.4f | <2 shared n=%5d "
          "mean=%.4f | ratio %.3f"
          % (len(far_hit), st.fmean(far_hit), len(far_miss), st.fmean(far_miss),
             st.fmean(far_hit)/st.fmean(far_miss)))
