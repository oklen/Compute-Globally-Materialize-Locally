"""kv_dens.py -- inside the tool_out bucket: which tokens carry the load, per token?

Paper section 9.  Region-level (kv_region.py) said `tool_out` is the only load-bearing
region -- but it is 66% of the stream, so "evict everything else" only frees 34%.  Here the bucket
is opened at three granularities and the estimator becomes DENSITY, not total:

    evict(u) = lo(full) - lo(full \\ u)       nats
    dens(u)  = evict(u) / |u|                 nats per token

Granularities (all tiled, non-overlapping, together covering every tool_out token):
    passage   the `[title] body` spans inside each RESULT block   (~4 / traj, ~119 tok)
    sentence  HotpotQA's own sentence split of each passage       (~25 / traj, ~20-30 tok)
    block     fixed 16-token blocks                               (~30 / traj) -- the granularity
              real systems evict at (MemDecay pages, H2O/SnapKV token selection)

RETENTION CURVES (the headline).  For a ranker pi and budget B, keep every non-tool_out token plus
the top-B% of tool_out tokens by pi, and read.  `oracle` (measured density) is the ceiling, `random`
the floor.  Query-aware rankers see the follow-up; query-free ones do not -- only the latter can run
in a deployed evictor that does not know the next question.

    oracle     measured dens(u)                                    ceiling
    attn_h2o   attention from ALL question positions (accumulated)  query-aware
    attn_snap  attention from the last 8 question positions         query-aware  (SnapKV window)
    attn_traj  attention accumulated DURING the trajectory itself   query-FREE   (the deployable one)
    bm25_now   BM25 of the block vs the follow-up                   query-aware
    bm25_q1    BM25 of the block vs the ORIGINAL question           query-free
    idf        IDF mass of the block                                query-free
    recency    position                                             query-free
    random     -                                                    floor

FOLLOW-UPS.  Same construction as kv_region.py: `stated` (the trajectory's own answer value) and
`unstated` (a year in a retrieved passage that no generated region mentions).  A SECOND unstated
follow-up (a different passage) is built when available: the Spearman rho between two same-type
density vectors is the whole query-free question -- if a unit's load is question-specific, no
evictor that has not seen the next question can rank it.

GATES.  (a) ceiling self-check: if dropping a single sentence moves the read-out on <10% of units,
the per-unit density read-out is floored and only the retention curves are reportable -- the smoke
run prints this fraction.  (b) placebo unit: a sentence of the source passage that does NOT contain
the probed value must read ~0.  (c) tiling assertion: sum |u| == |tool_out| at every granularity.
(d) additivity: sum_u evict(u) vs evict(whole tool_out) -- reported, never assumed.

Attention is taken from a SEPARATE eager forward; the KV used for every read-out still comes from
the normal SDPA path, so the read-outs stay bit-comparable with kv_region.py (changing the
compute path changes the numbers).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
        python -m kvmemory.kv_dens --traj ./out/traj.json --negs ./out/hnegs.json --out ./out/dens.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest4 import recog_q
from kvmemory.kv_e3ctrl import shape_class
from kvmemory.kv_cite import wb, strip_paren
from kvmemory.kv_ans import log_odds, YEAR

BUDGETS = (0.05, 0.10, 0.20, 0.40)
RANKERS = ("oracle", "attn_h2o", "attn_snap", "attn_traj", "bm25_now", "bm25_q1", "idf", "recency", "random")
QUERY_FREE = {"attn_traj", "bm25_q1", "idf", "recency", "random"}
WORD = re.compile(r"[a-z0-9]+")


def load_items_sent(path):
    """kv_cite.load_items + the per-title sentence list and the supporting-fact sentence ids.

    The sentence split is HotpotQA's own (context.sentences), and paras are built exactly as
    kv_cite does -- " ".join(sentences) -- so a passage's sentences tile its text with single
    spaces between them.  That is what lets a char range be mapped back to a sentence."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "bridge":
            continue
        ts, ss = r["context"]["title"], r["context"]["sentences"]
        sup = list(dict.fromkeys(r["supporting_facts"]["title"]))
        if len(sup) != 2 or any(t not in ts for t in sup):
            continue
        paras = {t: " ".join(ss[ts.index(t)]) for t in ts}
        pa = wb(r["answer"])
        A = [t for t in sup if pa.search(paras[t])]
        if len(A) != 1:
            continue
        tA = A[0]
        tB = [t for t in sup if t != tA][0]
        b = strip_paren(tA)
        if len(b) < 3 or not wb(b).search(paras[tB]):
            continue
        if not any(pa.search(s) for s in ss[ts.index(tA)]):
            continue
        supset = set(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]))
        out.append(dict(id=r["id"], q=r["question"], v=r["answer"], tA=tA, tB=tB, b=b,
                        titles=ts, paras=paras, sents={t: list(ss[ts.index(t)]) for t in ts},
                        sup=supset))
    return out


def tile_units(text, offsets, ntok, sents_by_title):
    """Split one tool_out region into passage / sentence / block units.

    `text` is the region text WITHOUT the trailing newline; `offsets` are char spans for the
    tokenisation of text + "\\n" (that is what kv_traj fed to the model).  Every granularity tiles
    [0, ntok) exactly -- asserted by the caller."""
    BIG = len(text) + 10          # the last unit absorbs the trailing "\n" token

    def tok_range(a, b):
        """tokens are assigned to a unit by their START char, so consecutive char ranges give
        consecutive, non-overlapping token ranges -- that is what makes the tiling exact."""
        lo = next((i for i, (cs, ce) in enumerate(offsets) if cs >= a), ntok)
        hi = next((i for i, (cs, ce) in enumerate(offsets) if cs >= b), ntok)
        return lo, max(hi, lo)

    marks = [(m.start(), m.group(1)) for m in re.finditer(r"\[([^\]]+)\] ", text)]
    if not marks:
        return {"passage": [dict(kind="passage", title=None, s=0, e=ntok, text=text)],
                "sentence": [dict(kind="sentence", title=None, sent_id=0, s=0, e=ntok, text=text)],
                "block": [dict(kind="block", s=a, e=min(a + 16, ntok),
                               text=text[offsets[a][0]:offsets[min(a + 16, ntok) - 1][1]])
                          for a in range(0, ntok, 16)]}
    passages, sentences = [], []
    for i, (cs, title) in enumerate(marks):
        c0 = 0 if i == 0 else cs                       # the "RESULT: " prefix rides with passage 0
        c1 = marks[i + 1][0] if i + 1 < len(marks) else BIG
        a, b = tok_range(c0, c1)
        if b > a:
            passages.append(dict(kind="passage", title=title, s=a, e=b, text=text[c0:min(c1, len(text))]))
        # sentence starts: the body begins right after "[title] "; sentences are joined by " ",
        # so start_j = body0 + sum(len(s_0..s_{j-1})) + j.  Spans are then made contiguous so no
        # token (least of all the joining space) falls between two units.
        body0 = text.index("] ", cs) + 2
        starts, cur = [], body0
        for j, sent in enumerate(sents_by_title.get(title) or []):
            if cur >= (c1 if c1 < BIG else len(text)):
                break
            starts.append((j, cur))
            cur += len(sent) + 1
        if not starts:
            starts = [(0, c0)]
        for k, (j, x0) in enumerate(starts):
            lo_c = c0 if k == 0 else x0                # "[title] " rides with sentence 0
            hi_c = starts[k + 1][1] if k + 1 < len(starts) else c1
            a2, b2 = tok_range(lo_c, hi_c)
            if b2 > a2:
                sentences.append(dict(kind="sentence", title=title, sent_id=j, s=a2, e=b2,
                                      text=text[lo_c:min(hi_c, len(text))]))
    blocks = [dict(kind="block", s=a, e=min(a + 16, ntok),
                   text=text[offsets[a][0]:offsets[min(a + 16, ntok) - 1][1]])
              for a in range(0, ntok, 16)]
    return {"passage": passages, "sentence": sentences, "block": blocks}


def _bm25(unit_texts, query, corpus):
    qt = set(WORD.findall(query.lower()))
    N = max(1, len(corpus))
    df = Counter()
    docs = [WORD.findall(p.lower()) for p in corpus]
    for d in docs:
        for w in set(d):
            df[w] += 1
    avg = max(1.0, sum(len(d) for d in docs) / N)
    out = []
    for u in unit_texts:
        d = WORD.findall(u.lower())
        tf = Counter(d)
        s = 0.0
        for w in qt:
            if w in tf:
                idf = math.log(1 + (N - df[w] + .5) / (df[w] + .5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        out.append(s)
    return out


def _idf(unit_texts, corpus):
    N = max(1, len(corpus))
    df = Counter()
    for p in corpus:
        for w in set(WORD.findall(p.lower())):
            df[w] += 1
    return [sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) for w in set(WORD.findall(u.lower())))
            for u in unit_texts]


def spearman(a, b):
    n = len(a)
    if n < 4:
        return None
    def rank(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return round(num / (da * db), 4) if da > 0 and db > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--data", default="./data/hotpotqa/hotpot_dev_distractor.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--done_from", default="")
    ap.add_argument("--negs", default="./out/hnegs.json")
    ap.add_argument("--max_attn_tok", type=int, default=2048)  # 36L x 32H x n^2 bf16; longest traj is 1865
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    HN = json.load(open(args.negs))["negs"] if args.negs and os.path.exists(args.negs) else {}
    print("[dens] LLM negatives for %d items" % len(HN), flush=True)
    T = json.load(open(args.traj))
    trajs = [t for t in T["trajs"] if t.get("answer") and t.get("src")]
    print("[dens] %d trajectories with an answer and the gold retrieved" % len(trajs), flush=True)
    if args.limit:
        trajs = trajs[: args.limit]
    items = {it["id"]: it for it in load_items_sent(args.data)}
    by_shape = {}
    for it in items.values():
        by_shape.setdefault(shape_class(it["v"]), []).append(it["v"])
    DONE = set()
    for g in filter(None, args.done_from.split(":")):
        for f in glob.glob(g):
            for line in open(f, errors="ignore"):
                m = re.match(r"^\[dens\] (\S+) \|", line)
                if m:
                    DONE.add(m.group(1))
    if DONE:
        print("[dens] resume: %d already done" % len(DONE), flush=True)

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    ids = lambda s: list(llm.tok(s, add_special_tokens=False).input_ids)
    # the whole unit split rides on char offsets; a slow tokenizer cannot produce them, and the
    # failure would only surface hundreds of items in.  Check once, loudly.
    _probe = llm.tok("RESULT: [T] a b c.\n", add_special_tokens=False, return_offsets_mapping=True)
    assert len(_probe["offset_mapping"]) == len(_probe["input_ids"]) and _probe["offset_mapping"][0][0] == 0, \
        "tokenizer gives no usable offset_mapping (need a fast tokenizer)"
    print("[dens] offset_mapping OK", flush=True)
    # SDPA silently returns an EMPTY attentions tuple for output_attentions=True, so the two
    # attention rankers need eager for their forward only.  The KV every read-out uses still comes
    # from the normal SDPA path (changing the compute path changes the numbers) -- the
    # eager forwards below produce attention SCORES only, never cache rows.
    _CORE = getattr(llm.model, "model", llm.model)
    _CORE = getattr(_CORE, "language_model", None) or _CORE
    _CFGS = [c for c in (llm.model.config, getattr(llm.model.config, "text_config", None),
                         getattr(_CORE, "config", None)) if c is not None]
    _ORIG = [getattr(c, "_attn_implementation", None) for c in _CFGS]

    def _eager(on):
        for c, o in zip(_CFGS, _ORIG):
            if o is not None:
                c._attn_implementation = "eager" if on else o

    def _attn_ok():
        _eager(True)
        try:
            t = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=llm.device)
            with torch.no_grad():
                o = _CORE(input_ids=t, use_cache=False, output_attentions=True)
            return bool(o.attentions) and o.attentions[0] is not None
        except Exception as e:                                # noqa: BLE001
            print("[dens] eager probe failed: %r" % (e,), flush=True)
            return False
        finally:
            _eager(False)

    ATTN_OK = _attn_ok()
    print("[dens] eager attention available: %s  (attn_h2o/attn_snap/attn_traj %s)"
          % (ATTN_OK, "on" if ATTN_OK else "SKIPPED"), flush=True)

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        return cand.get("A", 0.0), cand.get("B", 0.0)

    def menu_for(q, gold, cands):
        """forced-choice menu; the negative and the A/B order are picked by the NO-CONTEXT prior
        (|lo| minimal), so the selector is blind to the treatment -- same as kv_region/kv_ans."""
        best = None
        hids = ids(head)
        for neg in cands[: args.pool]:
            for ga in (True, False):
                a, b = (gold, neg) if ga else (neg, gold)
                kv = encode_block(llm, hids, list(range(len(hids))))
                c, pp = assemble(llm, [(kv, list(range(len(hids))))])
                pa, pb = probe(c, pp, llm._ids(recog_q(q, a, b) + tail + "Answer:"))
                del c
                tot = pa + pb
                pg = (pa if ga else pb) / tot if tot > 0 else 0.5
                if best is None or abs(pg - .5) < abs(best[0] - .5):
                    best = (pg, neg, ga, log_odds(pa, pb, ga))
        if best is None:
            return None
        pg, neg, ga, lo_none = best
        a, b = (gold, neg) if ga else (neg, gold)
        return dict(neg=neg, ga=ga, lo_none=round(lo_none, 4),
                    q2ids=torch.tensor([ids("<|im_end|>\n" + head + recog_q(q, a, b) + tail + "Answer:")],
                                       dtype=torch.long, device=llm.device))

    def serve_idx(full_kv, keep):
        """assemble exactly the cache rows at `keep` (ascending original positions)."""
        rt = torch.tensor(keep, dtype=torch.long)
        return assemble(llm, [([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv], keep)])

    def read_keep(full_kv, keep, m):
        c, pp = serve_idx(full_kv, keep)
        pa, pb = probe(c, pp, m["q2ids"])
        del c
        return round(log_odds(pa, pb, m["ga"]), 4)

    def attn_from_query(full_kv, keep, m, n_keep):
        """attention each kept cache row receives from the follow-up's tokens.
        h2o = summed over ALL query positions (accumulated importance), snap = last 8 only."""
        if not ATTN_OK:
            return None, None
        c, pp = serve_idx(full_kv, keep)
        L = m["q2ids"].shape[1]
        nxt = int(pp.max().item()) + 1
        _eager(True)
        try:
            with torch.no_grad():
                out = _CORE(input_ids=m["q2ids"], past_key_values=c, use_cache=True,
                            position_ids=torch.arange(nxt, nxt + L, device=llm.device).unsqueeze(0),
                            cache_position=torch.arange(n_keep, n_keep + L, device=llm.device),
                            attention_mask=torch.ones(1, n_keep + L, dtype=torch.long, device=llm.device),
                            output_attentions=True)
            A = out.attentions
            if not A or A[0] is None:
                return None, None
            h2o = torch.zeros(n_keep, dtype=torch.float32, device=llm.device)
            snap = torch.zeros(n_keep, dtype=torch.float32, device=llm.device)
            for a in A:                                   # (1, heads, L, n_keep + L)
                x = a[0, :, :, :n_keep].float().mean(0)   # mean over heads -> (L, n_keep)
                h2o += x.sum(0)
                snap += x[-8:].sum(0)
            del A, out
            return h2o.cpu().tolist(), snap.cpu().tolist()
        except Exception as e:                            # noqa: BLE001 -- ranker is optional
            print("[dens]   attn_from_query failed: %r" % (e,), flush=True)
            return None, None
        finally:
            _eager(False)
            del c
            torch.cuda.empty_cache()

    def attn_during_traj(stream, gen_pos):
        """attention the agent's OWN generation paid to each stream position -- query-free."""
        if not ATTN_OK or len(stream) > args.max_attn_tok or not gen_pos:
            return None
        _eager(True)
        try:
            idt = torch.tensor([stream], dtype=torch.long, device=llm.device)
            with torch.no_grad():
                out = _CORE(input_ids=idt, use_cache=False, output_attentions=True)
            A = out.attentions
            if not A or A[0] is None:
                return None
            gp = torch.tensor(gen_pos, dtype=torch.long, device=llm.device)
            acc = torch.zeros(len(stream), dtype=torch.float32, device=llm.device)
            for a in A:
                acc += a[0].float().mean(0).index_select(0, gp).sum(0)
            del A, out
            return acc.cpu().tolist()
        except Exception as e:                            # noqa: BLE001
            print("[dens]   attn_during_traj failed: %r" % (e,), flush=True)
            return None
        finally:
            _eager(False)
            torch.cuda.empty_cache()


    GENR = ("plan", "tool_in", "scratchpad", "answer")
    rows, skip = [], Counter()
    for t in trajs:
        if t["id"] in DONE:
            continue
        it = items.get(t["id"])
        if it is None:
            skip["no_item"] += 1; continue
        q1, v = t["q"], t["v"]
        stream, regs = t["stream"], t["regions"]
        N = len(stream)
        gen_text = " ".join(r["text"] for r in regs if r["name"] in ("plan", "scratchpad", "answer"))
        alltxt = " ".join(r["text"] for r in regs)

        # ---- units, in GLOBAL token coordinates ----
        units, TO, bad = {"passage": [], "sentence": [], "block": []}, [], ""
        for ri, r in enumerate(regs):
            if r["name"] != "tool_out":
                continue
            ntok = r["e"] - r["s"]
            enc = llm.tok(r["text"] + "\n", add_special_tokens=False, return_offsets_mapping=True)
            if len(enc["input_ids"]) != ntok:
                bad = "tok_mismatch %d!=%d" % (len(enc["input_ids"]), ntok); break
            u = tile_units(r["text"], list(enc["offset_mapping"]), ntok, it["sents"])
            for g in units:
                for x in u[g]:
                    units[g].append(dict(x, ri=ri, s=x["s"] + r["s"], e=x["e"] + r["s"]))
            TO.extend(range(r["s"], r["e"]))
        if bad:
            skip[bad.split()[0]] += 1; print("[dens] %s | SKIP %s" % (t["id"], bad), flush=True); continue
        if not TO:
            skip["no_tool_out"] += 1; continue
        for g in units:                                     # gate (c): exact tiling
            cov = sum(x["e"] - x["s"] for x in units[g])
            assert cov == len(TO), (t["id"], g, cov, len(TO))

        # ---- follow-ups ----
        cands_v = [x for x in by_shape.get(shape_class(v), []) if x.lower() != v.lower()
                   and not wb(x).search(alltxt) and not wb(v).search(x)]
        rnd.shuffle(cands_v)
        llm_negs = [x for x in HN.get(t["id"], {}).get("ans", []) if x.lower() != v.lower()
                    and not wb(x).search(alltxt) and not wb(v).search(x)]
        cands_v = list(dict.fromkeys(llm_negs + cands_v))
        qs = {}
        m1 = menu_for(q1, v, cands_v)
        if m1 is not None:
            qs["stated"] = (q1, v, m1, int(bool(wb(v).search(gen_text))))
        # `unstated` follows kv_region.py's rule EXACTLY (first qualifying passage, first region) so
        # that *_full and *_frame are bit-comparable with the region run.  A SECOND unstated
        # follow-up, on a different title, is then sought for the cross-question rho.
        # PASS 1 reproduces kv_region.py's rule exactly (first qualifying passage of each region,
        # year taken from that passage's FIRST sentence) so that `unstated` -- and with it *_full
        # and *_frame -- stay bit-comparable with the region run.
        # PASS 2 then sweeps EVERY remaining passage under a cleaner rule (exactly one year in the
        # whole passage, so the question is unambiguous) to find a SECOND follow-up.  Without it
        # only ~12% of items get one, and the cross-question rho -- the only result here that can
        # change deployment practice -- would run on a twelfth of the sample.
        cand_u, seen_t = [], set()

        def _cand(title, seg, first_sentence_only):
            if title in seen_t:
                return None
            ys = YEAR.findall(seg.split(". ")[0] if first_sentence_only else seg)
            if len(ys) != 1:
                return None
            f = ys[0]
            if wb(f).search(gen_text) or wb(f).search(q1) or f == v:
                return None
            negs = [str(int(f) + d) for d in (-7, -3, 3, 7, -12, 12, -20, 20)]
            negs = [y for y in negs if not wb(y).search(alltxt)]
            if not negs:
                return None
            seen_t.add(title)
            return (title,
                    "In the search results shown earlier, which year is mentioned in the "
                    "passage about %s?" % strip_paren(title), f, negs)

        for first_only in (True, False):
            for r in [x for x in regs if x["name"] == "tool_out"]:
                for mm in re.finditer(r"\[([^\]]+)\] ", r["text"]):
                    seg = r["text"][mm.end(): mm.end() + 400]
                    ys = YEAR.findall(seg.split(". ")[0])
                    # kv_region's break fires only when BOTH of its checks pass; replicate exactly
                    qual = (len(ys) == 1 and not wb(ys[0]).search(gen_text)
                            and not wb(ys[0]).search(q1) and ys[0] != v)
                    c = _cand(mm.group(1), seg, first_only)
                    if c is not None:
                        cand_u.append(c)
                    if first_only and qual:
                        break                     # kv_region stops at the first qualifying passage
        got = []
        for title, q2, f, negs in cand_u[:4]:
            m2 = menu_for(q2, f, negs)
            if m2 is not None:
                got.append((q2, f, m2, 0))
            if len(got) == 2:
                break
        for i, g in enumerate(got):
            qs["unstated" if i == 0 else "unstated2"] = g
        if not qs:
            skip["no_followup"] += 1; print("[dens] %s | SKIP no_followup" % t["id"], flush=True); continue

        # ---- read-outs ----
        full_kv = encode_block(llm, stream, list(range(N)))
        ALL = list(range(N))
        TOS = set(TO)
        NON_TO = [p for p in ALL if p not in TOS]
        gen_pos = [p for r in regs if r["name"] in GENR for p in range(r["s"], r["e"])]
        at = attn_during_traj(stream, gen_pos)
        corpus = [it["paras"][x] for x in it["titles"]]
        _, s0, e0 = t["src"]
        rec = dict(id=t["id"], q1=q1, v=v, n_tok=N, steps=t["steps"], to_tok=len(TO),
                   answer_correct=t["answer_correct"], note_states_gold=t["note_states_gold"],
                   share={n: sum(r["e"] - r["s"] for r in regs if r["name"] == n)
                          for n in ("system", "user", "plan", "tool_in", "tool_out", "scratchpad", "answer")},
                   n_units={g: len(units[g]) for g in units}, attn_traj_ok=int(at is not None),
                   sup=[[a, b] for a, b in sorted(it["sup"])])
        for kind, (q2, gold, m, stated) in qs.items():
            gp = wb(gold)
            rec[kind + "_gold"], rec[kind + "_neg"] = gold, m["neg"]
            rec[kind + "_none"], rec[kind + "_stated"] = m["lo_none"], stated
            lo_full = read_keep(full_kv, ALL, m)
            rec[kind + "_full"] = lo_full
            rec[kind + "_frame"] = read_keep(full_kv, NON_TO, m)     # == region run's drop_tool_out
            for g in (("sentence",) if kind == "unstated2" else ("passage", "sentence", "block")):
                out = []
                for u in units[g]:
                    lo = read_keep(full_kv, [p for p in ALL if not (u["s"] <= p < u["e"])], m)
                    nu = u["e"] - u["s"]
                    out.append(dict(s=u["s"], e=u["e"], n=nu, title=u.get("title"),
                                    sent_id=u.get("sent_id"),
                                    is_src=int(u["s"] < e0 and u["e"] > s0),
                                    is_sup=(int((u.get("title"), u.get("sent_id")) in it["sup"])
                                            if g == "sentence" else None),
                                    has_gold=int(bool(gp.search(u["text"]))),
                                    lo=lo, evict=round(lo_full - lo, 4),
                                    dens=round((lo_full - lo) / nu, 5)))
                rec["%s_%s" % (kind, g)] = out
                torch.cuda.empty_cache()
            if kind == "unstated2":
                continue
            # ---- retention curves over BLOCK units ----
            B = units["block"]
            texts = [u["text"] for u in B]
            blk = rec[kind + "_block"]
            sc = {"oracle": [x["dens"] for x in blk],
                  "bm25_now": _bm25(texts, q2, corpus), "bm25_q1": _bm25(texts, q1, corpus),
                  "idf": _idf(texts, corpus), "recency": [float(u["s"]) for u in B],
                  "random": [rnd.random() for _ in B]}
            if at is not None:
                sc["attn_traj"] = [sum(at[u["s"]:u["e"]]) / (u["e"] - u["s"]) for u in B]
            h2o, snap = attn_from_query(full_kv, ALL, m, N)
            if h2o is not None:
                sc["attn_h2o"] = [sum(h2o[u["s"]:u["e"]]) / (u["e"] - u["s"]) for u in B]
                sc["attn_snap"] = [sum(snap[u["s"]:u["e"]]) / (u["e"] - u["s"]) for u in B]
            for rk, s in sc.items():
                order = sorted(range(len(B)), key=lambda i: -s[i])
                for bud in BUDGETS:
                    cap, got_i, n = int(round(bud * len(TO))), [], 0
                    for i in order:
                        if n >= cap:
                            break
                        got_i.append(i); n += B[i]["e"] - B[i]["s"]
                    ks = set()
                    for i in got_i:
                        ks.update(range(B[i]["s"], B[i]["e"]))
                    rec["ret_%s_%s_%d" % (kind, rk, int(bud * 100))] = \
                        read_keep(full_kv, sorted(set(NON_TO) | ks), m)
            torch.cuda.empty_cache()
        for a, b in (("unstated", "unstated2"), ("stated", "unstated")):
            ka, kb = a + "_sentence", b + "_sentence"
            if ka in rec and kb in rec:
                da = {(x["s"], x["e"]): x["dens"] for x in rec[ka]}
                db = {(x["s"], x["e"]): x["dens"] for x in rec[kb]}
                sh = [k for k in da if k in db]
                rec["rho_%s_%s" % (a, b)] = spearman([da[k] for k in sh], [db[k] for k in sh])
        del full_kv
        torch.cuda.empty_cache()
        rows.append(rec)
        parts = []
        for k in qs:
            sent = rec.get(k + "_sentence") or []
            hit = [x for x in sent if x["has_gold"]]
            oth = [x for x in sent if not x["has_gold"]]
            parts.append("%s: none=%+.2f full=%+.2f frame=%+.2f | sentences n=%d with-value evict=%s rest=%s"
                         % (k, rec[k + "_none"], rec[k + "_full"], rec[k + "_frame"], len(sent),
                            ("%+.2f" % (sum(x["evict"] for x in hit) / len(hit))) if hit else "na",
                            ("%+.2f" % (sum(x["evict"] for x in oth) / len(oth))) if oth else "na"))
        print("[dens] %s | tok=%d to=%d unit %s | %s | rho_uu=%s rho_su=%s"
              % (t["id"], N, len(TO), rec["n_units"], " || ".join(parts),
                 rec.get("rho_unstated_unstated2"), rec.get("rho_stated_unstated")), flush=True)
        if len(rows) % 20 == 0:
            json.dump({"traj": args.traj, "budgets": list(BUDGETS), "rows": rows,
                       "skip": dict(skip)}, open(args.out, "w"))

    json.dump({"traj": args.traj, "budgets": list(BUDGETS), "rankers": list(RANKERS),
               "rows": rows, "skip": dict(skip)}, open(args.out, "w"))
    # ceiling self-check (gate a): does dropping ONE sentence move the read-out at all?
    ev = [abs(x["evict"]) for r in rows for k in ("stated", "unstated")
          for x in (r.get(k + "_sentence") or [])]
    moved = sum(1 for x in ev if x >= 0.5) / len(ev) if ev else 0.0
    print("[dens] rows=%d skip=%s" % (len(rows), dict(skip)))
    print("[dens] ceiling self-check: %d sentence-level units, share with |evict|>=0.5 = %.3f  (<0.10 => per-unit density is pinned at the floor,"
          " report the retention curve only)" % (len(ev), moved))
    print("DENS_DONE")


if __name__ == "__main__":
    main()
