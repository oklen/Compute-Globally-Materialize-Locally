"""rA_roles.py -- E1 step 1: label every turn of a LoCoMo conversation V / M / U for one question.

WHY. §7's retrieval half is measured on our own register template, where the carrier and the
reference edge are term-identical w.r.t. the query, so ANY bag-of-words scorer ties (proved for
the synthetic template: BM25 gives both rows the identical float). That is a theorem about the
synthetic family, not a rate in the wild. On a real corpus we cannot identify a *carrier* at all
-- distinguishing "carries the materialized value" from "points at the row that does" needs a
donor pair, which needs editing the corpus. What IS measurable is the weaker claim:

    R-A'  a deployed index ranks a turn that matches the topic but does NOT state the value
          above the turn that DOES state it.

That is exactly the failure MemReranker reports from the retrieval side ("semantic similarity does
not equate to containing the answer") and nobody has explained mechanistically.

ROLES, per (conversation, question), over every turn of that conversation:
  V  value-bearing -- the turn states the answer
  M  mention-only  -- shares >=2 content words with the question but does NOT state the answer
  U  unrelated

The M rule and the tokenizer are lifted verbatim from kv_harvest3.select_qa (the "carrier
candidate" screen), so the two experiments partition turns the same way.

NORMALISATION LADDER. `answer in text` is not enough: LoCoMo answers are often composite and
morphologically varied ("dinosaurs, nature"; "counseling or mental health for Transgender people"
where the turn says "trans people"). Four levels, each reported separately -- a big jump between
levels means the conclusion is normalisation-sensitive and the reader must see that:
  N0 raw substring
  N1 SQuAD normalize_answer (lowercase, strip punctuation/articles, fold whitespace)
  N2 N1 + split the answer on , / " or " / " and " and match ANY component
  N3 N2 + digit<->word and a crude plural/suffix strip

HARD GATE (A-1). LoCoMo `dia_id` is CONVERSATION-scoped, not global: 4,849 ids repeat across the
10 conversations. A global dia_id index silently resolves every gold evidence to the wrong turn
and the output still looks sane. We build the map per episode and assert it.

    python -m kvmemory.rA_roles --data ./data/locomo10.json --out ./out/rA_labels.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.locomo import load_locomo
from kvmemory.kv_harvest3 import toks

# ---------------------------------------------------------------- normalisation

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")
_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
           "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
           "11": "eleven", "12": "twelve"}
_WORDNUM = {v: k for k, v in _DIGITS.items()}
_SPLIT = re.compile(r",|\bor\b|\band\b|/")


def normalize_answer(s: str) -> str:
    """SQuAD official normalisation."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _stem(w: str) -> str:
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _n3(s: str) -> str:
    out = []
    for w in normalize_answer(s).split():
        w = _DIGITS.get(w, w)
        w = _WORDNUM.get(w, w)
        out.append(_stem(w))
    return " ".join(out)


def answer_forms(gold: str) -> dict:
    """The four normalisation levels; N2/N3 return a LIST of components (match ANY)."""
    n1 = normalize_answer(gold)
    comps = [c.strip() for c in _SPLIT.split(gold) if c.strip()]
    return {
        "N0": [gold],
        "N1": [n1] if n1 else [],
        "N2": [c for c in (normalize_answer(x) for x in comps) if c],
        "N3": [c for c in (_n3(x) for x in comps) if c],
    }


def _text_at(level: str, txt: str) -> str:
    if level == "N0":
        return txt
    if level == "N3":
        return _n3(txt)
    return normalize_answer(txt)


def _contains(pat: str, txt: str) -> bool:
    """Word-boundary containment. Plain substring is unsafe: the answer "2" matches inside
    "2023", "20", "12" -- one short numeric answer then labels a dozen unrelated turns V."""
    if not pat:
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(pat) + r"(?![A-Za-z0-9])", txt) is not None


# ---------------------------------------------------------------- labelling

_PREFIX = re.compile(r"^\s*(\[[^\]]*\]\s*)?[^:]{0,40}?:\s*")


def _utterance(seg) -> str:
    """The turn text minus the '[timestamp] speaker: ' prefix that locomo.py:54 prepends."""
    ts = (seg.meta or {}).get("timestamp") or ""
    sp = (seg.meta or {}).get("speaker") or ""
    t = seg.text
    if ts and t.startswith(f"[{ts}] "):
        t = t[len(ts) + 3:]
    if sp and t.startswith(f"{sp}: "):
        return t[len(sp) + 2:]
    return _PREFIX.sub("", t, count=1)


def label_episode(ep, min_share: int = 2) -> tuple[list, dict]:
    """Return (records, stats). One record per qualifying question."""
    segs = ep.segments
    # --- A-1 hard gate: the dia_id map is built from THIS episode only, and must be injective
    dia2idx = {}
    for i, s in enumerate(segs):
        if s.seg_id in dia2idx:
            raise AssertionError(
                f"duplicate dia_id {s.seg_id!r} WITHIN conversation {ep.conv_id} "
                f"(turns {dia2idx[s.seg_id]} and {i}) -- role labels would collapse")
        dia2idx[s.seg_id] = i

    # --- strip the "[timestamp] speaker: " prefix before matching.
    # locomo.py:54 builds text as f"[{ts}] {speaker}: {utt}", so the SESSION DATE lives in every
    # turn's text. Matching a date answer ("June 2023") against that hits the timestamp of every
    # turn in the session -- six consecutive false V rows, silently. Match the utterance only.
    texts = [_utterance(s) for s in segs]
    tok_sets = [set(toks(t)) for t in texts]
    pre = {lvl: [_text_at(lvl, t) for t in texts] for lvl in ("N0", "N1", "N2", "N3")}
    # N2 shares N1's text form; only the answer side is split
    pre["N2"] = pre["N1"]

    recs, stats = [], Counter()
    for qi, qa in enumerate(ep.qa):
        cat = qa.get("category")
        gold = str(qa.get("answer", "") or "").strip()
        if cat == 5 or not gold:
            stats["skip_no_gold_or_adversarial"] += 1
            continue
        ev_raw = qa.get("evidence") or []
        ev = [e for e in ev_raw if e in dia2idx]
        stats["ev_ids_unresolved"] += len(ev_raw) - len(ev)
        if not ev:
            stats["skip_no_resolvable_evidence"] += 1
            continue

        forms = answer_forms(gold)
        V = {}
        for lvl in ("N0", "N1", "N2", "N3"):
            pats = forms[lvl]
            V[lvl] = sorted(i for i, t in enumerate(pre[lvl])
                            if pats and any(_contains(p, t) for p in pats))

        qtok = set(toks(qa["question"]))
        vset_n1 = set(V["N1"])
        M = sorted(i for i in range(len(segs))
                   if i not in vset_n1 and len(qtok & tok_sets[i]) >= min_share)

        gold_idx = sorted(dia2idx[e] for e in ev)
        recs.append({
            "conv": ep.conv_id, "qi": qi, "cat": cat,
            "question": qa["question"], "gold": gold,
            "evidence": ev, "gold_idx": gold_idx,
            "n_turns": len(segs),
            "V": V, "nV": {k: len(v) for k, v in V.items()},
            "M_strict": M, "nM": len(M),
            # does the labelled V set actually cover the official gold evidence?
            "gold_in_V_N1": sorted(set(gold_idx) & vset_n1),
            # very short normalised answers ("2", "no") are ambiguous even under word-boundary
            # matching -- keep them separable rather than silently mixed in.
            "short_answer": int(len(normalize_answer(gold)) <= 2),
        })
        stats["kept"] += 1
    return recs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-share", type=int, default=2)
    ap.add_argument("--out", default="out/rA_labels.json")
    args = ap.parse_args()

    eps = load_locomo(args.data)
    all_recs, agg = [], Counter()
    seen_ids = set()
    dup_across = 0
    for ep in eps:
        ids = {s.seg_id for s in ep.segments}
        dup_across += len(seen_ids & ids)
        seen_ids |= ids
        recs, st = label_episode(ep, args.min_share)
        all_recs += recs
        agg.update(st)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n_conv": len(eps), "records": all_recs,
               "stats": dict(agg), "dup_dia_ids_across_conversations": dup_across},
              open(args.out, "w"), indent=1)

    print(f"conversations {len(eps)} | questions kept {agg['kept']}")
    print(f"A-1: dia_ids that repeat ACROSS conversations = {dup_across} "
          f"(a global index would mis-resolve them; we index per episode)")
    print(f"A-1: gold evidence ids that do NOT resolve = {agg['ev_ids_unresolved']}")
    print()
    for lvl in ("N0", "N1", "N2", "N3"):
        nv = [r["nV"][lvl] for r in all_recs]
        cov = sum(1 for r in all_recs if r["nV"][lvl] > 0)
        print(f"  {lvl}: questions with >=1 V row {cov}/{len(all_recs)} "
              f"({100*cov/max(1,len(all_recs)):.0f}%) | mean |V| {sum(nv)/max(1,len(nv)):.2f}")
    nm = [r["nM"] for r in all_recs]
    print(f"  M-strict: mean |M| {sum(nm)/max(1,len(nm)):.1f} | "
          f"questions with >=1 M {sum(1 for x in nm if x)}/{len(all_recs)}")
    both = sum(1 for r in all_recs if r["nV"]["N1"] > 0 and r["nM"] > 0)
    print(f"\n  MEASURABLE for pi (>=1 V and >=1 M, N1): {both}/{len(all_recs)}")
    dist = Counter(min(r["nV"]["N1"], 5) for r in all_recs if r["nV"]["N1"] > 0)
    print("  |V| distribution (N1, 5+ collapsed):",
          {k: dist[k] for k in sorted(dist)})
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
