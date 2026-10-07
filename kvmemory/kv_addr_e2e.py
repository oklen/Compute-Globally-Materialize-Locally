"""kv_addr_e2e.py -- donor separation when the ORACLE served set is replaced by a retriever.

kv_causal establishes the paper's headline on a served set we choose: {decoy, root} for the
1-hop readout, {decoy, root, ref1, ref2} for the 3-hop one. §5 flags that as oracle row
selection. This script keeps kv_causal's geometry byte-for-byte -- same trajectories (same RNG
stream, same seed), same position budgets, same native original-position serving, same 160-token
greedy readout -- and changes exactly one thing: WHICH rows are served.

Arms, at a matched budget k = |oracle set|, so no arm buys separation with a bigger window:
  oracle   the paper's set                            (reference)
  lex      LexicalRouter top-k over the 24 event texts
  emb      EmbeddingRouter top-k (Qwen3-Embedding-0.6B, standard vector RAG)   [--embed]

The decomposition that matters. A retriever that pulls the SOURCE event into the served set has
not exercised materialization at all -- the value is then present verbatim in served text, and
any separation is ordinary reading. So every arm is scored twice:

  sep_all       donor separation over all items          (what a system would actually deliver)
  sep_nosrc     donor separation on items where the retriever did NOT serve the source
                (the materialization-dependent subset -- the only place the mechanism is load-bearing)
  p_src         fraction of items whose top-k contains the source

Retrieval ranks the v1=ONLINE rendering with the BARE question (the trace form leaks the answer
tokens ONLINE/OFFLINE into the query and is not usable for selection -- see kv_addr). The two
donors differ only in the source event's text, so the ranking could in principle differ between
members; `xdonor_agree` reports how often the two members' top-k sets are identical, and the
ONLINE-rendered set is the one actually served to both so the pair stays byte-matched.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_EMBED_PATH=/path/to/Qwen3-Embedding-0.6B \
      SPRAG_EMBED_DEVICE=cuda:0 SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_addr_e2e --items 64 --seed 3900 \
        --embed --out ./out/addr_e2e.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES
from kvmemory.kv_causal import STATUS2, N_BODY, DEC_IDX, TRACE, pick_last, encode_variant
from kvmemory.kv_matrix import assemble as _assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_addr import to_segments
from kvmemory.components import LexicalRouter

# Chain slots (root, ref1, ref2). kv_causal hard-codes 13/18/23; the layouts exist to separate
# ROLE from POSITION in the {decoy,root} vs {decoy,ref1} contrast, where root sits at 13 and ref1
# at 18 -- so root is also the earlier row AND the one nearer the source (slots 8-11).
#
# The two cannot simply be swapped. Write-time propagation needs the chain to run in slot order
# (`M1 mirrors S` must be encoded before `M2 mirrors M1`), so in ANY working chain the root is
# necessarily nearer the source than the reference edges are. Position cannot be exchanged with
# role; it can only be COMPRESSED until there is no position difference left to explain the gap.
LAYOUTS = {
    "spread": (13, 18, 23),   # kv_causal's own: root 2-5 after the source, ref1 five slots later
    "packed": (20, 21, 22),   # root 9-12 after the source, ref1 ONE slot after root
}


def ask_forced(llm, H_ids, H, H_kv, tail, full_kv, rowmap, spans, eids, served, qt):
    """Forced-choice candidate-logit readout, same served rows as kv_causal's ask().

    Free generation cannot be used when the served set contains no state word: the model emits
    nothing parseable and the item is lost (60% of the retriever cells in the n=64 run). Reading
    the answer-word distribution directly removes the format dependence -- the same fix the paper
    applies where free generation under-reports a candidate-logit readout. Returns (argmax, probs).
    """
    crows = [pp for i in served for pp in range(spans[i][0], spans[i][0] + len(eids[i]))]
    rt = torch.tensor([rowmap[pp] for pp in crows], dtype=torch.long)
    sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
    raw, probs = answer_with_probs(llm, c, p, llm._ids(qt + tail + "Answer:"), STATUS2)
    del c, sub
    return (max(STATUS2, key=lambda w: probs.get(w, 0.0)),
            {k: round(v, 5) for k, v in probs.items()}, raw[:40])


def ask_raw(llm, H_ids, H, H_kv, tail, full_kv, rowmap, spans, eids, served, qt):
    """kv_causal's ask(), but also returning the raw generation.

    Identical rows, identical prompt, identical 160-token greedy decode -- the only difference is
    that the text is kept, so the separation counts can be re-derived by a second parser from the
    stored output rather than trusted from pick_last() alone. Two of the three measurement errors
    in this line were caught only because a stored intermediate could be re-read.
    """
    crows = [pp for i in served for pp in range(spans[i][0], spans[i][0] + len(eids[i]))]
    rt = torch.tensor([rowmap[pp] for pp in crows], dtype=torch.long)
    sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
    c, p = _assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
    o, _, _ = llm._greedy_pos(c, p, llm._ids(qt + tail), 160)
    del c, sub
    return pick_last(o), o


def ask_text(llm, H_ids, tail, body, served, qt):
    """The same served EVENTS, re-encoded as clean text IN ISOLATION rather than gathered as rows
    from the full-trajectory cache.

    This is a positive control, not a fair economic baseline, and it should be read as one. The
    source event is not in the served set, so its value is nowhere in these tokens and a text arm
    cannot separate the donors above the echo floor by construction -- we deleted the sentence that
    says the answer. What it forecloses is the competing reading of the KV result: that the served
    sentences alone ("M mirrors S") were somehow enough. If the text arm separated, the KV channel
    would not be doing the work.

    The contrast with the KV arm is exactly what materialization means. There, the root's rows were
    computed with the source visible in the same prefill; here they are computed without it. Same
    tokens, same question, same decode -- different write-time context.

    A fair text baseline is a different arm entirely: retrieval over the full pool, which may
    return the source itself and then answers by ordinary reading (`lex|full`, p_src = 1.00).
    """
    evs = "".join(f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                  for i in sorted(served))
    ids = list(H_ids) + list(llm.tok(evs, add_special_tokens=False).input_ids)
    kv = encode_block(llm, ids, list(range(len(ids))))
    c, p = _assemble(llm, [(kv, list(range(len(ids))))])
    o, _, _ = llm._greedy_pos(c, p, llm._ids(qt + tail), 160)
    del c, kv
    return pick_last(o), o


def render(rng_state: dict, v1: str, slots) -> dict:
    """Body dict for one donor member. Only slot m differs across members."""
    C1, C2, C3 = slots
    b = dict(rng_state["fill"])
    b[DEC_IDX] = f"register {rng_state['D']} set to state {rng_state['dm']}"
    b[rng_state["m"]] = f"register {rng_state['S']} set to state {v1}"
    b[C1] = f"register {rng_state['M1']} mirrors register {rng_state['S']}"
    b[C2] = f"register {rng_state['M2']} mirrors register {rng_state['M1']}"
    b[C3] = f"register {rng_state['T']} mirrors register {rng_state['M2']}"
    return b


def build_bases(seed: int, n_items: int):
    """Replay kv_causal's RNG stream so item i is the same trajectory it uses."""
    rng = random.Random(seed)
    out = []
    for it in range(n_items):
        D, S, M1, M2, T = rng.sample(NAMES, 5)
        dm = rng.choice(STATUS2)
        m = rng.randrange(8, 12)
        others = [n for n in NAMES if n not in (D, S, M1, M2, T)]
        fill = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
                for i in range(N_BODY)}
        out.append({"it": it, "D": D, "S": S, "M1": M1, "M2": M2, "T": T,
                    "dm": dm, "m": m, "fill": fill})
        # Part B's draws are consumed verbatim (never used here) so item i stays the same
        # trajectory kv_causal builds at this seed; skipping them desynchronises every later item.
        D2, Sa, Sb, Ma, Mb, T2 = rng.sample(NAMES, 6)
        rng.choice(STATUS2), rng.choice(STATUS2)                      # va, vb
        others2 = [n for n in NAMES if n not in (D2, Sa, Sb, Ma, Mb, T2)]
        {i: f"register {rng.choice(others2)} refreshed to {rng.randrange(100, 999)}"
         for i in range(N_BODY)}                                       # fill2
        for _ in ("LA", "LB"):
            rng.choice(STATUS2)                                        # decoy value, per leaf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3900)
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--forced", action="store_true",
                    help="candidate-logit readout instead of free generation (no unparseable items)")
    ap.add_argument("--anchor", action="store_true",
                    help="always serve the decoy as format anchor; the retriever fills the "
                         "remaining k-1 rows (equalises answer vocabulary across arms)")
    ap.add_argument("--extra", action="store_true",
                    help="add the two reference arms the retriever comparison does not supply: "
                         "`replay` (all 24 events served -- nothing evicted, the ceiling) and "
                         "`txt`/`txtrep` (the same events re-encoded as isolated text). The "
                         "existing arms are computed independently per item, so adding these "
                         "cannot perturb them -- the stored spread/3900 figures are the check")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="spread",
                    help="chain slot positions; 'packed' puts root and ref1 in adjacent slots "
                         "far from the source, so the root-vs-ref1 contrast has almost no "
                         "position difference left to explain it")
    ap.add_argument("--out", default="./out/addr_e2e.json")
    args = ap.parse_args()
    C1, C2, C3 = slots = LAYOUTS[args.layout]
    assert DEC_IDX not in slots and not any(8 <= s < 12 for s in slots), \
        f"layout {slots} collides with the decoy (0) or the source window (8-11)"
    assert C1 < C2 < C3, "chain must run in slot order or the value never propagates at write time"

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head + HDR, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))

    routers = {"lex": LexicalRouter()}
    if args.embed:
        from kvmemory.components import EmbeddingRouter
        from kvmemory.embed import QwenEmbedder
        routers["emb"] = EmbeddingRouter(QwenEmbedder(), use_text=True)

    # The free-generation readout asks the model to reason first and END with the state word
    # (TRACE), and pick_last() reads that last word. A candidate-logit probe reads the FIRST token
    # after "Answer:", where TRACE makes the model begin a reasoning preamble -- both candidates
    # then sit in the tail of the distribution and the argmax is noise. A run with TRACE left in
    # put the reference arm BELOW its own donor-blind floor (oracle|full .047 vs .25), which is how
    # the mismatch was caught. Under --forced the trace instruction is therefore dropped, and every
    # arm -- including oracle -- is re-measured under that prompt so the comparison stays internal.
    A = " Answer ONLINE or OFFLINE." if args.forced else " Answer ONLINE or OFFLINE." + TRACE
    bases = build_bases(args.seed, args.items)
    rows = []

    for base in bases:
        m, M1, T, S = base["m"], base["M1"], base["T"], base["S"]
        oracle = {"root": [DEC_IDX, C1], "full": [DEC_IDX, C1, C2, C3]}
        q = {"root": f"What state is register {M1} in?",
             "full": f"What state is register {T} in?"}
        rec = {"it": base["it"], "m": m, "layout": args.layout, "served": {}, "ans": {}}

        # --- selection: rank the ONLINE rendering with the bare question, matched budget ---
        segs_on = to_segments(render(base, "ONLINE", slots))
        segs_off = to_segments(render(base, "OFFLINE", slots))
        sets = {}
        for qk in ("root", "full"):
            k = len(oracle[qk])
            sets[("oracle", qk)] = (oracle[qk], "kv")
            for rn, r in routers.items():
                if args.anchor:
                    # The oracle set carries the decoy, which is the ONLY served event containing a
                    # state word; a retriever set without it has no answer vocabulary at all. Free
                    # generation then returns nothing (60% unparseable) and a first-token probe
                    # instead measures decoy echo -- the two readouts fail in opposite directions,
                    # so neither can compare the sets. Pinning the decoy in every arm equalises the
                    # scaffold, and the retriever spends its remaining k-1 rows on evidence.
                    cs_on = [s for s in segs_on if s.seg_id != str(DEC_IDX)]
                    cs_off = [s for s in segs_off if s.seg_id != str(DEC_IDX)]
                    pick = sorted({DEC_IDX, *(int(s) for s in r.select(q[qk], cs_on, k - 1))})
                    pick_off = sorted({DEC_IDX, *(int(s) for s in r.select(q[qk], cs_off, k - 1))})
                else:
                    pick = sorted(int(s) for s in r.select(q[qk], segs_on, k))
                    pick_off = sorted(int(s) for s in r.select(q[qk], segs_off, k))
                sets[(rn, qk)] = (pick, "kv")
                rec[f"xdonor_agree_{rn}_{qk}"] = int(pick == pick_off)
                rec[f"src_in_{rn}_{qk}"] = int(m in pick)
            if args.extra:
                # `replay` is the no-eviction ceiling: every event served, so the source is
                # present and the question is ordinary reading -- it prices what Event-KV gives
                # up, not what it adds. `txt`/`txtrep` re-encode the same events as isolated
                # text (see ask_text: positive control, expected at the floor for `txt`).
                allev = list(range(N_BODY))
                for a, sv, mode in (("replay", allev, "kv"),
                                    ("txt", oracle[qk], "text"),
                                    ("txtrep", allev, "text")):
                    sets[(a, qk)] = (sv, mode)
                    rec[f"src_in_{a}_{qk}"] = int(m in sv)
                    rec[f"xdonor_agree_{a}_{qk}"] = 1     # no selection, so nothing to disagree
        rec["served"] = {f"{a}|{qk}": v for (a, qk), (v, _) in sets.items()}

        # --- readout: both donors, every arm, same trajectory encode per donor ---
        src_budget = max(len(llm.tok(f"<step {m+1}>\naction: reg_op()\nobservation: "
                                     f"register {S} set to state {v}\n",
                                     add_special_tokens=False).input_ids) for v in STATUS2)
        for v1 in STATUS2:
            body = render(base, v1, slots)
            evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                   for i in range(N_BODY)]
            kv, rowmap, spans, eids = encode_variant(llm, H_ids, H, evs, {m: src_budget})
            for (arm, qk), (served, mode) in sets.items():
                qt = f"\n\nQuestion: {q[qk]}{A}"
                if mode == "text":
                    w, o = ask_text(llm, H_ids, tail, body, served, qt)
                    rec["ans"][f"{arm}|{qk}|{v1}"] = w
                    rec.setdefault("gen", {})[f"{arm}|{qk}|{v1}"] = o
                elif args.forced:
                    w, pr, raw = ask_forced(llm, H_ids, H, H_kv, tail, kv, rowmap, spans,
                                            eids, served, qt)
                    rec["ans"][f"{arm}|{qk}|{v1}"] = w
                    rec.setdefault("probs", {})[f"{arm}|{qk}|{v1}"] = pr
                    rec.setdefault("raw", {})[f"{arm}|{qk}|{v1}"] = raw
                else:
                    w, o = ask_raw(llm, H_ids, H, H_kv, tail, kv, rowmap, spans,
                                   eids, served, qt)
                    rec["ans"][f"{arm}|{qk}|{v1}"] = w
                    rec.setdefault("gen", {})[f"{arm}|{qk}|{v1}"] = o
            del kv
            torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[e2e] it{base['it']} " + " ".join(
            f"{a}|{qk}:{rec['ans'][f'{a}|{qk}|ONLINE']}/{rec['ans'][f'{a}|{qk}|OFFLINE']}"
            for (a, qk) in sets), flush=True)

    json.dump({"args": vars(args), "rows": rows}, open(args.out, "w"), indent=1)

    arms = sorted({k.rsplit("|", 1)[0] for k in rows[0]["ans"]})
    print(f"\nlayout={args.layout} {slots} (root, ref1, ref2)  seed={args.seed}  "
          f"anchor={args.anchor}")
    print(f"{'arm|query':<16} {'n':>4} {'p_none':>7} {'sep_all':>8} {'p_src':>7} {'n_nosrc':>8} "
          f"{'sep_nosrc':>10} {'xdonor':>7}")
    summary = {}
    for key in arms:
        arm, qk = key.split("|")
        # p_none: readouts that produced no state word at all. Under --forced this is 0 by
        # construction; under free generation it is the format failure that made the 1-hop
        # retriever cells uninterpretable in the first n=64 run.
        pn = sum(1 for r in rows for v in STATUS2 if r["ans"][f"{key}|{v}"] is None) / (2 * len(rows))
        sep = [int(r["ans"][f"{key}|ONLINE"] == "ONLINE"
                   and r["ans"][f"{key}|OFFLINE"] == "OFFLINE") for r in rows]
        if arm == "oracle":
            psrc, nosrc, xd = 0.0, sep, 1.0
        else:
            src = [r[f"src_in_{arm}_{qk}"] for r in rows]
            psrc = sum(src) / len(src)
            nosrc = [s for s, x in zip(sep, src) if not x]
            xd = sum(r[f"xdonor_agree_{arm}_{qk}"] for r in rows) / len(rows)
        sn = (sum(nosrc) / len(nosrc)) if nosrc else float("nan")
        summary[key] = {"n": len(rows), "p_none": pn, "sep_all": sum(sep) / len(sep),
                        "p_src": psrc, "n_nosrc": len(nosrc), "sep_nosrc": sn,
                        "xdonor_agree": xd}
        print(f"{key:<16} {len(rows):>4} {pn:>7.3f} {sum(sep)/len(sep):>8.3f} {psrc:>7.2f} "
              f"{len(nosrc):>8} {sn:>10.3f} {xd:>7.2f}")
    print("\n(donor-blind floor for sep = .25: both members must land on their own donor)")

    if args.forced:
        # Instrument guards. (a) The candidate words must actually hold the probability mass at the
        # probed position; if they sit in the tail, the argmax between them is noise. (b) The oracle
        # arm is the reference and cannot be below its own chance floor -- if it is, the probe is
        # measuring something other than the answer and NO cell in this run is interpretable.
        mass = [sum(r["probs"][k].values()) for r in rows for k in r["probs"]]
        mass.sort()
        print(f"\ncandidate mass at the probed position: median={mass[len(mass)//2]:.3f} "
              f"p10={mass[len(mass)//10]:.3f}  (near 0 => probe is reading the distribution tail)")
        # Only meaningful with enough items: at n=4 a reference arm whose true rate is ~.33 shows
        # zero separations 20% of the time, so a small-sample zero is noise, not a broken probe.
        if len(rows) >= 32:
            for key in arms:
                if key.startswith("oracle") and summary[key]["sep_all"] < 0.25:
                    print(f"  !! INSTRUMENT FAILURE: {key} sep={summary[key]['sep_all']:.3f} is "
                          f"below the .25 donor-blind floor -- do not report this run")
        else:
            print(f"  (n={len(rows)} < 32: floor guard not evaluated, too few items to be evidence)")
        ex = rows[0]["raw"].get("oracle|root|ONLINE", "")
        print(f"first-token context sample (oracle|root): {ex!r}")
    json.dump({"args": vars(args), "rows": rows, "summary": summary},
              open(args.out, "w"), indent=1)
    print("E2E_DONE", flush=True)


if __name__ == "__main__":
    main()
