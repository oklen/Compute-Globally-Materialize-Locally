import os
import json, re, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the SAME loaders the measurement used, so the text is byte-identical to what was encoded
from kvmemory.kv_harvest3 import load_realtalk, load_locomo

R = json.load(open("e3_div3.json"))["rows"]
TX = {}
for ds, loader in (("realtalk", lambda: load_realtalk("../data/REALTALK")),
                   ("locomo", lambda: load_locomo("../data/locomo10.json"))):
    for cv in loader():
        TX[(ds, str(cv["cid"]))] = [t[2] for t in cv["turns"]]
print("loaded texts for %d conversations (%s)"
      % (len(TX), {d: sum(1 for k in TX if k[0] == d) for d in ("realtalk", "locomo")}))
print("div rows: %d items; ds values %s" % (len(R), {r.get("ds") for r in R}))

STOP = set("the a an and or but if then of to in on at for with is are was were be been am i you "
           "he she it we they me my your his her our their that this these those do did does not "
           "no yes so as by from about him them us what when where who how why have has had will "
           "would can could just really very much more most some any all".split())
def words(s):
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2}

OUT, miss = [], []
for r in R:
    key = (r.get("ds"), str(r["conv"]))
    if key not in TX:
        # the div rows store conv as "L<id>" / "R<id>" in some passes; try the stripped form
        alt = (r.get("ds"), str(r["conv"])[1:])
        key = alt if alt in TX else None
    if key is None:
        miss.append((r.get("ds"), r["conv"])); continue
    txt = TX[key]; fg = r["first_gold"]
    qw = words(r["q"]) | words(r["gold"])
    for t in r["turns"]:
        if not t["downstream"] or t["own_gold"] or t["j"] >= len(txt):
            continue
        OUT.append({"ds": r.get("ds"), "conv": "%s:%s" % (r.get("ds"), r["conv"]), "q": r["q"],
                    "j": t["j"], "dist": (t["j"] - fg) if fg is not None else -1,
                    "ov": len(words(txt[t["j"]]) & qw), "ntok": t["ntok"],
                    "rel": t["rel"], "rel_last": t["rel_last"], "rel_rest": t["rel_rest"],
                    "rel_maxc": t["rel_maxc"], "n_last": t["n_last"], "d_last": t["d_last"],
                    "n_rest": t["n_rest"], "d_rest": t["d_rest"]})
print("joined %d downstream turns; items joined %d/%d; unjoined %s"
      % (len(OUT), len(R) - len(miss), len(R), sorted(set(miss))[:6]))
print("conversations in join: %d ; questions: %d"
      % (len({o["conv"] for o in OUT}), len({(o["conv"], o["q"]) for o in OUT})))
print("by ds: %s" % {d: len({o["conv"] for o in OUT if o["ds"] == d}) for d in ("realtalk", "locomo")})
json.dump(OUT, open("div3_join2.json", "w"))
