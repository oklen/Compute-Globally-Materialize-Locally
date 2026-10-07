import os
import json, glob, sys, re, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = []
for p in sorted(glob.glob("e3div_*/e3div.*.json")):
    rows += json.load(open(p))["rows"]
seen, R = set(), []
for r in rows:
    k = (r["ds"], r["conv"], r["q"])
    if k not in seen:
        seen.add(k); R.append(r)
print("items=%d  conversations=%d  datasets=%s"
      % (len(R), len({r["conv"] for r in R}), sorted({r["ds"] for r in R})))

# ---------- 1. the floor ----------
pre = [t["rel"] for r in R for t in r["turns"] if not t["downstream"] and not t["own_gold"]]
print("\n[1] FLOOR  turns before the first gold occurrence have identical inputs in both worlds")
print("    n=%d   max=%.3e   -> %s" % (len(pre), max(pre), "PASS (deterministic)" if max(pre) == 0 else "FAIL"))

# ---------- 2. how much survived the position guard ----------
kept = sum(len(r["turns"]) for r in R)
shift = sum(r["n_shift"] for r in R)
print("\n[2] POSITION GUARD  kept %d turns, dropped %d shifted (%.0f%%)"
      % (kept, shift, 100.0 * shift / max(kept + shift, 1)))
print("    length of the two worlds matched outright on %d/%d items"
      % (sum(r["len_match"] for r in R), len(R)))

# ---------- 3. the number ----------
post = [t["rel"] for r in R for t in r["turns"] if t["downstream"] and not t["own_gold"]]
post.sort()
print("\n[3] DOWNSTREAM, own text and position IDENTICAL in both worlds")
print("    n=%d  mean=%.4f  median=%.4f  p90=%.4f  max=%.4f  zero=%d(%.1f%%)"
      % (len(post), st.fmean(post), post[len(post)//2], post[int(.9*len(post))], post[-1],
         sum(1 for x in post if x == 0), 100.0*sum(1 for x in post if x == 0)/len(post)))

# ---------- 4. distance decay: diffuse perturbation or targeted absorption? ----------
print("\n[4] DECAY WITH DISTANCE from the first changed turn")
print("    a diffuse perturbation decays smoothly; an absorbed fact should not have to")
band = {}
for r in R:
    fg = r["first_gold"]
    if fg is None: continue
    for t in r["turns"]:
        if not t["downstream"] or t["own_gold"]: continue
        d = t["j"] - fg
        b = 1 if d <= 2 else 5 if d <= 5 else 10 if d <= 10 else 25 if d <= 25 else \
            50 if d <= 50 else 100 if d <= 100 else 999
        band.setdefault(b, []).append(t["rel"])
for b in sorted(band):
    v = band[b]
    if len(v) < 30: continue
    print("    dist<=%-4s n=%5d  mean=%.4f  median=%.4f" % (b if b < 999 else ">100", len(v),
                                                            st.fmean(v), sorted(v)[len(v)//2]))

# ---------- 5. selectivity: is the drift on topically-related turns? ----------
from kvmemory.locomo import load_locomo
TX = {}
for cv in load_locomo("../data/locomo10.json"):
    TX[str(cv["cid"])] = [t[2] for t in cv["turns"]]
STOP = set("the a an and or but if then of to in on at for with is are was were be been i you he "
           "she it we they my your his her our their that this these those do did does not no yes "
           "so as by from about me him them us what when where who how why".split())
def words(s):
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2}
print("\n[5] SELECTIVITY  does the drift sit on turns that talk about the question?")
rel_hit, rel_miss = [], []
for r in R:
    if str(r["conv"]) not in TX: continue
    txt = TX[str(r["conv"])]
    qw = words(r["q"]) | words(r["gold"])
    for t in r["turns"]:
        if not t["downstream"] or t["own_gold"]: continue
        if t["j"] >= len(txt): continue
        ov = len(words(txt[t["j"]]) & qw)
        (rel_hit if ov >= 2 else rel_miss).append(t["rel"])
if rel_hit and rel_miss:
    print("    shares >=2 content words with the question/gold: n=%5d mean=%.4f median=%.4f"
          % (len(rel_hit), st.fmean(rel_hit), sorted(rel_hit)[len(rel_hit)//2]))
    print("    shares  <2                                     : n=%5d mean=%.4f median=%.4f"
          % (len(rel_miss), st.fmean(rel_miss), sorted(rel_miss)[len(rel_miss)//2]))
    print("    ratio = %.3f  (1.0 = no selectivity at all)" % (st.fmean(rel_hit)/st.fmean(rel_miss)))
