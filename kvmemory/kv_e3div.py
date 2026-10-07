"""kv_e3div.py -- variant of kv_e3ctrl.py that adds the per-turn KV drift between the original and
counterfactual worlds (read by analysis/an_div*.py).

Parent module's description:
kv_e3ctrl.py -- E3: repair the X8 positive control so that `harv - iso` becomes interpretable.

The frozen control (kv_harvest4.py, phase B, Table tab:x8ctrl) failed for two DIFFERENT reasons,
and neither is a polish item: without a control that fires, a `harv > iso` on real dialogue cannot
be told apart from "the splice/recognition path works at all".

  C1  OPTION-PRIOR SATURATION (Qwen3).  The two-choice menu paired the gold against the gold of
      another QA in the SAME conversation, which is usually the wrong TYPE for the question (ask
      for a city, offer a date).  The model answered from the question alone: under the
      COUNTERFACTUAL carrier -- where the evidence turns have been rewritten to say the negative --
      it still picked gold ~80%.  The served rows never entered the decision, so the arm measured
      nothing.
  C2  UNSCORABLE FLIPPED ARM (Gemma-4).  The readout was an 8-token free generation scanned for a
      standalone A or B.  33/62 counterfactual readouts contained neither, so the follow:anti
      contrast was 3:0 -- too few items to adjudicate state transfer vs. readout activation.
  C3  INCOMPLETE COUNTERFACTUAL (found by this run's smoke, not previously recorded).  The frozen
      flip rewrote gold -> negative only inside the OFFICIAL EVIDENCE turns.  Every other mention
      of the value stayed, so the counterfactual conversation still supported gold and a carrier
      that faithfully absorbed its history had no reason to flip.  Part of "the counterfactual
      carrier still selects gold" was therefore the counterfactual, not the carrier.  We report
      gold_hits_nonev (mentions the frozen flip left behind) so the size of this is on the record.

Repairs (pre-registered; applied identically to EVERY arm; kv_harvest4.py is left byte-identical
so the published numbers stay reproducible):

  R1  PRIOR-BALANCED NEGATIVE.  Candidate pool = every other QA gold in the dataset that is
      shape-compatible with this gold (same class of {num, year, date, one capitalised token,
      short capitalised phrase, phrase}; length ratio within 2x) and that does NOT occur verbatim
      anywhere in the conversation (a negative already present in the history would be preferred
      or rejected for reasons that have nothing to do with the carrier).  Each surviving candidate
      is scored by the NO-CONTEXT arm only -- header + question + menu, nothing served -- and we
      keep the one minimising |p(gold letter) - 0.5|.
      The selector is blind to the treatment: it never sees a carrier, original or counterfactual,
      so it cannot move the orig-vs-cf contrast it is supposed to make measurable.  Menu order is
      the same sha1(question) rule as the frozen run, so the gold LETTER does not depend on which
      negative is chosen.  Item eligibility is byte-identical to kv_harvest4 (gold verbatim in >=1
      evidence turn; >=1 valid same-conversation negative exists), so the item set stays the
      reported n=62 and the before/after comparison is on the identical items.

  R2  FORCED-CHOICE LOGIT READOUT.  kv_fail2.answer_with_probs -- the same probe kv_bankprobe uses
      -- ends the prompt with "Answer:" so the next-token distribution IS the answer distribution,
      and compares the mass on {" A","A"} against {" B","B"}.  Scorable by construction.  The OLD
      free-generation readout is run as well on every arm, so the scorable-rate gate can be read on
      the old instrument and the new one side by side rather than asserted.

  R3  COMPLETE COUNTERFACTUAL.  The flip is applied to the WHOLE conversation, and the result is
      asserted gold-free.  The two worlds then differ in exactly one thing -- the value -- which is
      what makes "did the row follow its donor" a well-posed question.  --cf_scope evidence
      reproduces the frozen behaviour for the ablation.

  R4  PLACEBO CARRIER.  The smoke run showed the injected line's own TEXT is not inert: with
      nothing served the model is at p(gold)=.50, and the isolated injected line alone takes it to
      ~1.0 -- an assertion that "the answer was determined earlier" makes the model commit to its
      top prior.  So orig-vs-iso is confounded and cannot be the primary here.  orig-vs-cf is not:
      the two share byte-identical carrier text at byte-identical positions and differ only in the
      prefill they were harvested from.  To separate an ADDRESSED absorption from diffuse
      world-level contamination we run a second carrier that names a DIFFERENT question of the same
      conversation.  Both carriers live in the same two worlds, so the placebo's orig-cf gap is the
      diffuse baseline the real carrier has to beat.

Arms per item (each read out both ways), carrier c in {q = names this question, p = placebo}:
  none    header + question + menu, nothing served      instrument: measures the option prior
  goldev  the official evidence turns as visible text    invariant: is the TASK still the task
  c_iso   the injected carrier encoded ALONE at its original positions
  c_orig  the injected carrier's rows harvested from the ORIGINAL full-conversation prefill
  c_cf    the same rows harvested from the COUNTERFACTUAL prefill
PRIMARY: (q_orig - q_cf) - (p_orig - p_cf) on p(gold), plus follow:anti on the hard picks.

Gates (all reported; a gate that fails is reported as failed, not worked around):
  G1  instrument, C1:  none-arm gold rate within [.40,.60] and mean p(gold) within [.35,.65]
  G1' as the plan words it: cf-arm gold rate near chance (a WORKING path pushes it BELOW chance --
      both readings are served by reporting the number)
  G2  instrument, C2:  forced-choice scorable rate = 1.00, against the free-generation rate
  I1  invariant: goldev in [.85,.96] -- if it moved, the readout change made the task easier/harder
  I2  invariant: none at chance
  G3  instrument, C3: the counterfactual conversation contains zero verbatim occurrences of gold
Result: follow = orig picks gold AND cf picks neg;  anti = the mirror.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_e3ctrl --ds all --out ./out/e3.json
"""
from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest3 import load_realtalk, load_locomo, select_qa
from kvmemory.kv_harvest4 import N_INJECT, pick_ab, recog_q, hard_neg
from kvmemory.kv_e3negs import qkey

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
POOL_SCORED = 32          # candidates actually probed per item (deterministic prefix of the pool)


def shape_class(s):
    """Coarse answer TYPE.  A negative of the wrong type is answerable from the question alone,
    which is exactly the saturation C1 describes."""
    t = s.strip()
    if not t:
        return "phrase"
    if re.fullmatch(r"[\d,\.\s]+", t):
        return "num"
    if re.search(r"\b(19|20)\d{2}\b", t):
        return "year"
    low = t.lower()
    if any(m in low for m in MONTHS) or re.search(r"\b\d{1,2}/\d{1,2}", t):
        return "date"
    w = t.split()
    if len(w) == 1:
        return "name1" if t[:1].isupper() else "word1"
    if len(w) <= 3 and all(x[:1].isupper() for x in w if x):
        return "nameN"
    return "phrase"


def neg_pool(gold, conv_text_low, all_golds):
    """Shape-compatible, length-compatible, absent from this conversation.  Deterministic order."""
    gl = gold.strip().lower()
    kls = shape_class(gold)
    lg = max(1, len(gold.strip()))
    out = []
    for cand in all_golds:
        c = cand.strip()
        cl = c.lower()
        if len(c) < 3 or cl == gl or cl in gl or gl in cl:
            continue
        if shape_class(c) != kls:
            continue
        r = len(c) / lg
        if r < 0.5 or r > 2.0:
            continue
        if cl in conv_text_low:
            continue
        out.append(c)
    out.sort(key=lambda c: (abs(len(c) - lg), c))
    return out[:POOL_SCORED]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["realtalk", "locomo", "all"], required=True)
    ap.add_argument("--rt_path", default="./data/REALTALK")
    ap.add_argument("--lc_path", default="./data/locomo10.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_qa", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N items")
    ap.add_argument("--n_inject", type=int, default=N_INJECT,
                    help="per-conversation cap on probe items.  4 reproduces the published 62-item "
                         "set and is the primary reading; 12 takes it to 111 for power.")
    ap.add_argument("--negs", default="./out/e3_negs.json",
                    help="R1 candidate pool written by kv_e3negs (per-item hard negatives). "
                         "Missing file or missing item -> fall back to the dataset gold pool.")
    ap.add_argument("--cf_scope", choices=["conv", "evidence"], default="conv",
                    help="R3: 'conv' flips gold->neg through the WHOLE conversation (the two "
                         "worlds then differ only in the value); 'evidence' reproduces the frozen "
                         "evidence-turns-only flip for the ablation")
    ap.add_argument("--carriers", default="q,p",
                    help="q = the injected line names THIS question (frozen wording); "
                         "p = placebo, it names a different question of the same conversation")
    ap.add_argument("--frozen_neg", action="store_true",
                    help="ablation: keep kv_harvest4's same-conversation negative (R1 off), so the "
                         "R1 contribution can be read off the identical items")
    ap.add_argument("--out", default="./out/e3.json")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    HPOS = list(range(H))

    NEGS = {}
    if args.negs and os.path.exists(args.negs):
        NEGS = json.load(open(args.negs)).get("negs", {})
        print(f"[e3] loaded {len(NEGS)} generated negative sets from {args.negs}", flush=True)
    else:
        print(f"[e3] WARNING no --negs file at {args.negs}: the option prior will be balanced "
              f"only over borrowed dataset golds, which the smoke run showed is not enough",
              flush=True)

    if args.ds == "all":
        tasks = ([("realtalk", c) for c in load_realtalk(args.rt_path)]
                 + [("locomo", c) for c in load_locomo(args.lc_path)])
    else:
        loader = load_realtalk(args.rt_path) if args.ds == "realtalk" else load_locomo(args.lc_path)
        tasks = [(args.ds, c) for c in loader]
    tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]

    # dataset-wide gold pool for R1, built ONCE from the same frozen selection
    qas_of = {}
    golds_of = {"realtalk": set(), "locomo": set()}
    for ds_name, cv in tasks:
        q = select_qa(cv)
        if args.max_qa:
            q = q[: args.max_qa]
        qas_of[(ds_name, cv["cid"])] = q
        for x in q:
            golds_of[ds_name].add(x["gold"].strip())
    for k in golds_of:
        golds_of[k] = sorted(golds_of[k])

    def probe(cache, pos, qids):
        """R2 forced choice.  Returns (letter, p_A, p_B, raw_continuation)."""
        raw, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=8)
        pa, pb = cand.get("A", 0.0), cand.get("B", 0.0)
        return ("A" if pa >= pb else "B"), pa, pb, raw

    def free(cache, pos, qids):
        """The frozen readout, kept for the C2 gate: 8-token greedy scanned for a standalone A/B."""
        o, _, _ = llm._greedy_pos(cache, pos, qids, 8)
        return pick_ab(o), o

    def rows_cache(full_kv, spans, idxs):
        crows = [pp for i in idxs for pp in range(*spans[i])]
        rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
        sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
        return sub, crows

    def prefill(turn_list):
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids)
                for _, _, t in turn_list]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
        return kv, spans, eids

    rows = []
    for ds_name, cv in tasks:
        qas = qas_of[(ds_name, cv["cid"])]
        if not qas:
            continue
        turns = cv["turns"]
        conv_low = "\n".join(t[2] for t in turns).lower()

        # ---- eligibility: byte-identical to kv_harvest4 phase B ----
        probe_items = []
        for qa in qas:
            gl = qa["gold"].strip().lower()
            if len(gl) < 3:
                continue
            if not any(gl in turns[i][2].lower() for i in qa["goldev"]):
                continue
            neg0 = hard_neg(qas, qa)
            if not neg0:
                continue
            if neg0.strip().lower() in gl or gl in neg0.strip().lower():
                continue
            probe_items.append((qa, neg0.strip()))
        probe_items.sort(key=lambda x: (-x[0]["strong"], x[0]["q"]))
        probe_items = probe_items[:args.n_inject]
        if not probe_items:
            continue
        print(f"[e3] {ds_name} {cv['cid']} turns={len(turns)} items={len(probe_items)}", flush=True)

        qlist = [x[0]["q"] for x in probe_items]
        for ii, (qa, neg0) in enumerate(probe_items):
            gold = qa["gold"].strip()
            gl = gold.lower()
            hash_a = int(hashlib.sha1(("inj" + qa["q"]).encode()).hexdigest(), 16) % 2 == 0

            def menu(neg, gold_is_a, body="", forced=True):
                a, b = (gold, neg) if gold_is_a else (neg, gold)
                t = body + recog_q(qa["q"], a, b) + tail
                return llm._ids(t + "Answer:" if forced else t)

            # ---- R1: negative AND menu order chosen by the NO-CONTEXT prior only ----
            sel = []
            if args.frozen_neg:
                neg, gold_is_a = neg0, hash_a
            else:
                gen = [c for c in NEGS.get(qkey(ds_name, cv["cid"], qa["q"]), [])
                       if c.lower() not in conv_low and c.lower() != gl
                       and c.lower() not in gl and gl not in c.lower()]
                pool, seen = [], set()
                for c in gen + neg_pool(gold, conv_low, golds_of[ds_name]):
                    if c.lower() not in seen:
                        seen.add(c.lower())
                        pool.append(c)
                for cand in pool[:POOL_SCORED]:
                    for ga in (True, False):
                        c, p = assemble(llm, [(H_kv, HPOS)])
                        _, pa, pb, _ = probe(c, p, menu(cand, ga))
                        del c
                        tot = pa + pb
                        pg = (pa if ga else pb) / tot if tot > 0 else 0.5
                        sel.append({"neg": cand, "a": int(ga), "p_gold": round(pg, 4),
                                    "mass": round(tot, 4)})
                sel.sort(key=lambda d: (abs(d["p_gold"] - 0.5), d["neg"], -d["a"]))
                if sel:
                    neg, gold_is_a = sel[0]["neg"], bool(sel[0]["a"])
                else:
                    neg, gold_is_a = neg0, hash_a
            gold_letter, neg_letter = ("A", "B") if gold_is_a else ("B", "A")
            torch.cuda.empty_cache()

            evset = set(qa["goldev"])
            hits_nonev = sum(1 for j, t in enumerate(turns) if j not in evset and gl in t[2].lower())
            rec = {"ds": ds_name, "conv": cv["cid"], "cat": qa["cat"], "strong": qa["strong"],
                   "q": qa["q"], "gold": gold, "neg": neg, "neg_frozen": neg0,
                   "gold_letter": gold_letter, "gold_is_a": int(gold_is_a),
                   "shape": shape_class(gold), "n_probed": len(sel),
                   "sel_p_gold": sel[0]["p_gold"] if sel else None,
                   "gold_hits_nonev": hits_nonev, "cf_scope": args.cf_scope,
                   "neg_generated": int(bool(sel) and any(
                       d["neg"].lower() == neg.lower()
                       for d in sel[:1]) and neg.lower() in {
                       c.lower() for c in NEGS.get(qkey(ds_name, cv["cid"], qa["q"]), [])})}

            qids = menu(neg, gold_is_a)
            qids_free = menu(neg, gold_is_a, forced=False)

            # ---- text arms: none (prior instrument) and goldev (task invariant) ----
            gev_txt = "".join(turns[i][2] + "\n" for i in qa["goldev"])
            for arm, body in (("none", ""), ("goldev", gev_txt)):
                c, p = assemble(llm, [(H_kv, HPOS)])
                lt, pa, pb, raw = probe(c, p, menu(neg, gold_is_a, body))
                del c
                c, p = assemble(llm, [(H_kv, HPOS)])
                fl, fo = free(c, p, menu(neg, gold_is_a, body, forced=False))
                del c
                tot = pa + pb
                rec[arm + "_pick"] = lt
                rec[arm + "_p_gold"] = round((pa if gold_is_a else pb) / tot if tot > 0 else .5, 4)
                rec[arm + "_mass"] = round(tot, 5)
                rec[arm + "_free"] = fl
                rec[arm + "_raw"] = raw[:80]
                rec[arm + "_free_raw"] = fo[:80]

            # ---- the injected carriers, in both worlds ----
            last_sn = turns[-1][0]
            q_other = qlist[(ii + 1) % len(qlist)] if len(qlist) > 1 else None

            def inj_line(question):
                return (f"[session {last_sn+1} follow-up] moderator: for the record, the "
                        f"answer to \"{question}\" was determined earlier in this chat and "
                        "noted accordingly")

            carriers = {}
            for cname in args.carriers.split(","):
                cname = cname.strip()
                if cname == "q":
                    carriers["q"] = inj_line(qa["q"])
                elif cname == "p" and q_other:
                    carriers["p"] = inj_line(q_other)
            rec["placebo_q"] = q_other

            # word-boundary aware: the frozen flip used a bare re.escape, so a short gold also
            # rewrote the inside of longer words ('Rob' matched 9 places of which 1 was the word).
            # Only 6/62 items were affected and excluding them moved nothing, but the flip should
            # not put mangled tokens into the counterfactual world.
            _pb = re.escape(gold)
            if gold[:1].isalnum():
                _pb = r"\b" + _pb
            if gold[-1:].isalnum():
                _pb = _pb + r"\b"
            pat = re.compile(_pb, re.I)
            worlds = {}
            for tag in ("orig", "cf"):
                tl = []
                for j, (sn, d, txt) in enumerate(turns):
                    if tag == "cf" and (args.cf_scope == "conv" or j in evset):
                        txt = pat.sub(neg, txt)
                    tl.append((sn, d, txt))
                worlds[tag] = tl
            # count with the SAME boundary rule the flip uses.  A bare substring count reported 50
            # surviving "mentions" at n=111 -- every one of them a sub-word coincidence
            # (`yesterday` for gold "Yes", `catching` for "Cat", `networking` for "two",
            # `works` for "work").  The gate was measuring something the flip never claimed to
            # remove, and would have failed a run that was in fact clean.
            rec["cf_gold_residual"] = sum(len(pat.findall(t[2])) for t in worlds["cf"])
            rec["cf_gold_substr"] = sum(1 for t in worlds["cf"] if gl in t[2].lower())
            # ================= DIVERGENCE MEASUREMENT (kv_e3div) =================
            # Behaviourally, materialization is null: E2's k=1 point -- one retrieved row
            # encoded inside the whole conversation versus the same row encoded alone --
            # is -0.165 log-odds (p=.21), and harv - self is ~0 at every budget.  That is a
            # null about what the READER does with the row.  This asks the mechanical question
            # underneath it: does a natural turn's KV move at all when an upstream fact changes?
            #
            # The two worlds differ only in the value of `gold`.  For a turn whose own text is
            # untouched, everything about its forward pass is identical except the context it
            # attends to.  So any nonzero divergence IS absorption, by construction -- no reader,
            # no judge, no readout in the way.
            #
            # Built-in floor: a turn that ends BEFORE the first occurrence of gold has literally
            # identical inputs in both worlds (same tokens, same positions, same causal prefix),
            # so its divergence must be exactly 0.  Anything else means the measurement is
            # picking up nondeterminism rather than absorption, and nothing below is usable.
            def _flat_kv(kv, a, b):
                return [(K[:, :, a:b, :].float(), V[:, :, a:b, :].float()) for K, V in kv]

            kv_o, spans_o, _ = prefill(worlds["orig"])
            kv_c, spans_c, _ = prefill(worlds["cf"])
            first_gold = None
            for j, (sn, d, txt) in enumerate(worlds["orig"]):
                if pat.search(txt):
                    first_gold = j
                    break
            per_turn = []
            n_shift = 0
            for j in range(len(worlds["orig"])):
                ao, bo = spans_o[j]
                ac, bc = spans_c[j]
                # POSITION GUARD.  gold -> neg changes token counts ("Adoption agencies" ->
                # "Nonprofit food bank networks"), and one longer turn shifts every later turn's
                # POSITION IDS.  Those turns then differ through RoPE alone, which would be read
                # as absorption.  The zero-floor gate cannot catch this: the floor lives before
                # any edit, where no shift has happened yet.  So require the span to be
                # identical, not merely the same length.
                if (ao, bo) != (ac, bc):
                    n_shift += 1
                    continue
                own = bool(pat.search(worlds["orig"][j][2]))
                num = den = 0.0
                for (Ko, Vo), (Kc, Vc) in zip(_flat_kv(kv_o, ao - H, bo - H),
                                              _flat_kv(kv_c, ac - H, bc - H)):
                    num += float(((Ko - Kc) ** 2).sum() + ((Vo - Vc) ** 2).sum())
                    den += float((Ko ** 2).sum() + (Vo ** 2).sum())
                per_turn.append({"j": j, "own_gold": int(own),
                                 "downstream": int(first_gold is not None and j > first_gold),
                                 "rel": round(math.sqrt(num / den) if den > 0 else 0.0, 8)})
            del kv_o, kv_c
            torch.cuda.empty_cache()
            rec["n_shift"] = n_shift
            rec["len_match"] = int(spans_o[-1][1] == spans_c[-1][1])
            rec["first_gold"] = first_gold
            rec["n_turns"] = len(worlds["orig"])
            rec["turns"] = per_turn
            pre = [t["rel"] for t in per_turn if not t["downstream"] and not t["own_gold"]]
            post = [t["rel"] for t in per_turn if t["downstream"] and not t["own_gold"]]
            own = [t["rel"] for t in per_turn if t["own_gold"]]
            rec["div_pre_max"] = max(pre) if pre else None
            rec["div_post_mean"] = sum(post) / len(post) if post else None
            rec["div_post_max"] = max(post) if post else None
            rec["div_own_mean"] = sum(own) / len(own) if own else None
            rows.append(rec)
            print("[e3div] %s q=%.30s | turns=%d firstgold=%s shifted=%d | PRE max=%.3e (floor, must be 0) "
                  "| POST n=%d mean=%.3e max=%.3e | OWN n=%d mean=%.3e"
                  % (cv["cid"], qa["q"], len(per_turn), first_gold, n_shift,
                     rec["div_pre_max"] if rec["div_pre_max"] is not None else float("nan"),
                     len(post), rec["div_post_mean"] or 0.0, rec["div_post_max"] or 0.0,
                     len(own), rec["div_own_mean"] or 0.0), flush=True)
            if args.limit and len(rows) >= args.limit:
                break
            continue

            for cname, ctxt in sorted(carriers.items()):
                for tag in ("orig", "cf"):
                    tl = worlds[tag] + [(last_sn + 1, "INJ", ctxt)]
                    kv_v, spans_v, eids_v = prefill(tl)
                    inj_idx = len(tl) - 1
                    sub, crows = rows_cache(kv_v, spans_v, [inj_idx])
                    key = f"{cname}_{tag}"
                    c, p = assemble(llm, [(H_kv, HPOS), (sub, crows)])
                    lt, pa, pb, raw = probe(c, p, qids)
                    del c
                    c, p = assemble(llm, [(H_kv, HPOS), (sub, crows)])
                    fl, fo = free(c, p, qids_free)
                    del c, sub
                    tot = pa + pb
                    rec[key + "_pick"] = lt
                    rec[key + "_p_gold"] = round((pa if gold_is_a else pb) / tot if tot > 0 else .5, 4)
                    rec[key + "_mass"] = round(tot, 5)
                    rec[key + "_free"] = fl
                    rec[key + "_raw"] = raw[:80]
                    rec[key + "_free_raw"] = fo[:80]
                    if tag == "orig":
                        aa, bb = spans_v[inj_idx]
                        kv_i = encode_block(llm, eids_v[inj_idx], list(range(aa, bb)))
                        blk = [(H_kv, HPOS), (kv_i, list(range(aa, bb)))]
                        c, p = assemble(llm, blk)
                        lt, pa, pb, raw = probe(c, p, qids)
                        del c
                        c, p = assemble(llm, blk)
                        fl, fo = free(c, p, qids_free)
                        del c, kv_i
                        tot = pa + pb
                        rec[cname + "_iso_pick"] = lt
                        rec[cname + "_iso_p_gold"] = round((pa if gold_is_a else pb) / tot
                                                           if tot > 0 else .5, 4)
                        rec[cname + "_iso_free"] = fl
                    del kv_v
                    torch.cuda.empty_cache()
                rec[cname + "_follow"] = int(rec[cname + "_orig_pick"] == gold_letter
                                             and rec[cname + "_cf_pick"] == neg_letter)
                rec[cname + "_anti"] = int(rec[cname + "_orig_pick"] == neg_letter
                                           and rec[cname + "_cf_pick"] == gold_letter)
                rec[cname + "_dp"] = round(rec[cname + "_orig_p_gold"]
                                           - rec[cname + "_cf_p_gold"], 4)
            rows.append(rec)
            print(f"[e3] {cv['cid']} none={rec['none_p_gold']:.2f} "
                  + " ".join(f"{c}:iso={rec[c+'_iso_p_gold']:.2f}/o={rec[c+'_orig_p_gold']:.2f}"
                             f"/cf={rec[c+'_cf_p_gold']:.2f}" for c in sorted(carriers))
                  + f" resid={rec['cf_gold_residual']} nonev={hits_nonev} "
                  f"gold={gold[:20]!r} neg={neg[:20]!r}", flush=True)
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    json.dump({"rows": rows, "model": os.environ.get("SPRAG_MODEL_PATH", ""),
               "frozen_neg": args.frozen_neg, "cf_scope": args.cf_scope,
               "carriers": args.carriers}, open(args.out, "w"), indent=1)

    n = len(rows)
    if n:
        def rate(a, key="_pick"):
            return sum(int(r[a + key] == r["gold_letter"]) for r in rows) / n

        def mean(k):
            return sum(r[k] for r in rows) / n

        cs = sorted({c for r in rows for c in ("q", "p") if c + "_orig_pick" in r})
        print(f"\n===== E3 control (n={n}, cf_scope={args.cf_scope}, "
              f"frozen_neg={args.frozen_neg}) =====")
        for a in ["none", "goldev"] + [f"{c}_{t}" for c in cs for t in ("iso", "orig", "cf")]:
            # guard per ROW, not on rows[0]: a conversation with a single eligible item has no
            # placebo carrier, and keying off the first row killed the printer after a full run.
            sub = [r for r in rows if a + "_pick" in r]
            if not sub:
                continue
            gr = sum(int(r[a + "_pick"] == r["gold_letter"]) for r in sub) / len(sub)
            pg = sum(r[a + "_p_gold"] for r in sub) / len(sub)
            sc = sum(1 for r in sub if r.get(a + "_free") is not None) / len(sub)
            print(f"  {a:8s} n={len(sub):3d} gold={gr:.3f}  mean p_gold={pg:.3f}  "
                  f"free-readout scorable={sc:.3f}")
        for c in cs:
            sub = [r for r in rows if c + "_dp" in r]
            if not sub:
                continue
            print(f"  carrier {c}: n={len(sub)} follow={sum(r[c + '_follow'] for r in sub)} "
                  f"anti={sum(r[c + '_anti'] for r in sub)} "
                  f"mean(orig-cf) p_gold={sum(r[c + '_dp'] for r in sub) / len(sub):+.4f}")
        both = [r for r in rows if "q_dp" in r and "p_dp" in r]
        if both:
            print(f"  PRIMARY (q_dp - p_dp) n={len(both)}: "
                  f"{sum(r['q_dp'] - r['p_dp'] for r in both) / len(both):+.4f}")
        print(f"  G1 prior:    none gold={rate('none'):.3f} (gate .40-.60), "
              f"mean p_gold={mean('none_p_gold'):.3f} (gate .35-.65)")
        if "q_cf_free" in rows[0]:
            print(f"  G2 scorable: forced=1.000 vs free="
                  f"{sum(1 for r in rows if r['q_cf_free'] is not None) / n:.3f} on q_cf")
        print(f"  G3 cf clean: residual gold mentions="
              f"{sum(r['cf_gold_residual'] for r in rows)} (gate 0); "
              f"frozen flip would have left {sum(r['gold_hits_nonev'] for r in rows)}")
        print(f"  I1 goldev:   {rate('goldev'):.3f} (gate .85-.96)")
    print("E3_DONE", flush=True)


if __name__ == "__main__":
    main()
