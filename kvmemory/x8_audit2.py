#!/usr/bin/env python3
"""x8_audit2.py -- X8 step-0 v2: STRICT parallel audit of REALTALK + LOCOMO (cats 1-3).

Frozen rules (per user's rulings 2026-07-18):
  source  = official evidence dia_ids; anchor session s_first = session of earliest evidence
            turn, s_last = session of the LAST evidence turn.
  carrier = a turn in a session > s_last (strict; "after ALL evidence") that
            (i) shares >= 2 content words with the QUESTION (toks: [a-z0-9_]+, len>2,
                stopword-filtered),
            (ii) does NOT contain the gold answer string (case-insensitive substring),
            (iii) is not itself an evidence turn.
            after-first counts (session > s_first) also reported for comparability with
            the user's loose screen.
  window-strict = additionally the +/-1-turn window is answer-free.
  strong  = carrier shares with the question a token of document frequency <= 2 across
            that conversation's sessions (rare-entity stratum).
  REALTALK extra: dyad graph from filenames; participant connected components reported
            (10 participants x 2 chats each => ~3 components; clustering plan must use
            components, not chats, as the top stratum).

    python3 kvmemory/x8_audit2.py --realtalk ./data/REALTALK --locomo ./data/locomo10.json
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

_TOKR = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(("the a an to of and in is are was were on at for with as by it this that "
                   "be been you your what which who how when where did does do i we they he "
                   "she from about their his her have has had will would can could there "
                   "not no yes but or if so its it's im i'm".split()))


def toks(s):
    return [t for t in _TOKR.findall(s.lower()) if t not in _STOP and len(t) > 2]


def audit(convs, tag, cats_keep):
    """convs: list of dicts with keys: cid, turns=[(sess, dia_id, text)], qa=list"""
    res = []
    for cv in convs:
        turns = cv["turns"]
        dia2sess = {d: s for s, d, _ in turns if d}
        sess_ids = sorted(set(s for s, _, _ in turns))
        sess_text = defaultdict(list)
        for s, _, b in turns:
            sess_text[s].append(b.lower())
        df = Counter()
        for s in sess_ids:
            for w in set(toks(" ".join(sess_text[s]))):
                df[w] += 1
        for qa in cv["qa"]:
            cat = qa.get("category", 0)
            gold = str(qa.get("answer", "")).strip()
            if cat not in cats_keep or not gold:
                continue
            gl = gold.lower()
            ev = [e for e in (qa.get("evidence") or []) if e in dia2sess]
            rec = {"tag": tag, "cid": cv["cid"], "cat": cat, "n_ev": len(ev)}
            if not ev:
                rec["status"] = "no_evidence"
                res.append(rec)
                continue
            s_first = min(dia2sess[e] for e in ev)
            s_last = max(dia2sess[e] for e in ev)
            rec["ev_sessions"] = len(set(dia2sess[e] for e in ev))
            qtok = set(toks(qa["question"]))
            evset = set(ev)
            cand_first, cand_last, strict_win, strong = [], [], [], []
            for idx, (s, d, b) in enumerate(turns):
                if d in evset:
                    continue
                bl = b.lower()
                share = qtok & set(toks(bl))
                if len(share) < 2 or gl in bl:
                    continue
                entry = (s, d)
                if s > s_first:
                    cand_first.append(entry)
                if s > s_last:
                    cand_last.append(entry)
                    wl = " ".join(t[2].lower() for t in turns[max(0, idx-1):idx+2])
                    if gl not in wl:
                        strict_win.append(entry)
                    if any(df.get(w, 99) <= 2 for w in share):
                        strong.append(entry)
            rec.update({"n_after_first": len(cand_first), "n_after_last": len(cand_last),
                        "n_window_strict": len(strict_win), "n_strong": len(strong),
                        "status": "ok"})
            res.append(rec)
    return res


def report(res, tag, ncomp_info=""):
    ok = [r for r in res if r["status"] == "ok"]
    print(f"\n===== {tag} ===== {ncomp_info}")
    print(f"QA considered={len(res)}  evidence-locatable={len(ok)}  "
          f"no_evidence={sum(1 for r in res if r['status']=='no_evidence')}")
    for name, key in (("after-FIRST-evidence (loose, user-comparable)", "n_after_first"),
                      ("after-LAST-evidence (STRICT headline)", "n_after_last"),
                      ("  + window-strict", "n_window_strict"),
                      ("  + strong-entity stratum", "n_strong")):
        q = [r for r in ok if r.get(key, 0) >= 1]
        bycat = Counter(r["cat"] for r in q)
        print(f"  {name:44s}: {len(q):4d} qualifying  by-cat={dict(sorted(bycat.items()))}")
    q = [r for r in ok if r.get("n_after_last", 0) >= 1]
    bycid = Counter(r["cid"] for r in q)
    print(f"  strict qualifying by conversation: {dict(sorted(bycid.items()))}")
    sing = [r for r in q if r.get("ev_sessions", 9) == 1]
    print(f"  strict ∩ single-evidence-session: {len(sing)}")
    hist = Counter(min(r["n_after_last"], 8) for r in ok)
    print(f"  carrier-count hist (8=8+): {dict(sorted(hist.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtalk", default="./data/REALTALK")
    ap.add_argument("--locomo", default="./data/locomo10.json")
    args = ap.parse_args()

    # ---- REALTALK ----
    convs = []
    edges = []
    for f in sorted(glob.glob(os.path.join(args.realtalk, "Chat_*.json"))):
        base = os.path.basename(f)[:-5]
        parts = base.split("_")
        pa, pb = parts[2], parts[3]
        edges.append((pa, pb))
        d = json.load(open(f))
        turns = []
        for k in sorted((k for k in d if re.fullmatch(r"session_\d+", k)),
                        key=lambda k: int(k.split("_")[1])):
            sn = int(k.split("_")[1])
            for t in d[k]:
                txt = (t.get("clean_text") or "").strip()
                turns.append((sn, t.get("dia_id", ""), f"{t.get('speaker','')}: {txt}"))
        convs.append({"cid": base.replace("Chat_", "C"), "turns": turns, "qa": d.get("qa", [])})
    # participant connected components
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    comps = defaultdict(set)
    for a, b in edges:
        comps[find(a)].update([a, b])
    comp_desc = f"| participants={len(parent)} dyads={len(edges)} components={len(comps)} " \
                f"sizes={[len(v) for v in comps.values()]}"
    rt = audit(convs, "REALTALK", cats_keep={1, 2, 3})
    report(rt, "REALTALK (cats 1-3)", comp_desc)

    # ---- LOCOMO cats 1-3 ----
    data = json.load(open(args.locomo))
    convs = []
    for i, el in enumerate(data):
        conv = el["conversation"]
        turns = []
        for k in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                        key=lambda k: int(k.split("_")[1])):
            sn = int(k.split("_")[1])
            for t in conv[k]:
                utt = (t.get("text") or "").strip()
                cap = t.get("blip_caption")
                if cap:
                    utt += f"  [shares an image: {cap}]"
                turns.append((sn, t.get("dia_id", ""), f"{t.get('speaker','')}: {utt}"))
        convs.append({"cid": f"L{i}", "turns": turns, "qa": el.get("qa", [])})
    lc = audit(convs, "LOCOMO", cats_keep={1, 2, 3})
    report(lc, "LOCOMO (cats 1-3)")

    json.dump({"realtalk": rt, "locomo": lc},
              open("./out/x8_audit2_rows.json", "w"))
    print("\nrows -> ./out/x8_audit2_rows.json")
    print("X8AUDIT2_DONE")


if __name__ == "__main__":
    main()
