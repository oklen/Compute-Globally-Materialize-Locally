"""kv_addr.py -- is a materialized carrier ADDRESSABLE? (oracle row selection, removed)

§5 hands the readout the rows we already know carry the value. A serving system has no such
oracle: it sees a query and must pick events out of an index of their TEXT. This script swaps
the oracle for the retrievers a real memory system ships with -- run on the SAME donor-paired
trajectories, at the same seed, so the served sets are comparable item by item.

Two questions, in order:

  ADDRESSABILITY  Given the paper's own question, what rank does each ROLE get -- the
                  (omitted) source, the root carrier, the reference edges, the decoy, the
                  fillers? Crucially the source is IN the candidate pool: we do not assume
                  it is evicted, we measure whether the retriever evicts it.
  END-TO-END      Feed the retriever's top-k as the served set instead of the oracle's.
                  Does the donor separation survive? (--e2e, needs a GPU + model.)

Pre-registered reading. The root carrier "register M mirrors register S" NAMES M; the source
"register S set to state ONLINE" does not. A query about M should therefore rank the carrier
above the source, and a text-indexed memory system would drop the source and keep the carrier
ON ITS OWN RANKING -- making the paper's eviction scenario not adversarial but ordinary. The
competing outcome is that this register-world is too lexically uniform for surface retrieval to
separate the roles at all (both source and carrier share {register, state} with the query), in
which case any apparent addressability is a RECENCY artifact -- the carrier simply sits later.
We report raw overlap scores alongside ranks so the two accounts are distinguishable.

Query form matters and is controlled. The harness question carries a trace instruction
("Locate the mirror statement ...") whose word `mirror` would boost every carrier by
construction. The bare user question is the primary condition; --qform trace is the leakage
control, and is expected to inflate carrier rank for a reason that has nothing to do with
materialization.

    # retrieval only -- no model, no GPU
    python -m kvmemory.kv_addr --items 200 --seed 3900 --out ./out/addr_lex.json

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_EMBED_PATH=/path/to/Qwen3-Embedding-0.6B \
      SPRAG_EMBED_DEVICE=cuda:0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
      python -m kvmemory.kv_addr --items 200 --seed 3900 --embed --out ./out/addr_emb.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.core import Segment
from kvmemory.components import LexicalRouter, _tok
from kvmemory.kv_vartrack import NAMES

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX = 0
TRACE = (" Locate the mirror statement for that register, then end your answer with the "
         "final state word.")
KS = (1, 2, 4, 8)


def build_items(seed: int, n_items: int):
    """Replay kv_causal.py's RNG stream verbatim so item i here IS item i there.

    Part B consumes from the same stream, so it is regenerated even though only Part A feeds
    the primary contrast; dropping it would desynchronise every later item.
    """
    rng = random.Random(seed)
    outA, outB = [], []
    for it in range(n_items):
        # ---- Part A: depth-3 chain, one omitted source ----
        D, S, M1, M2, T = rng.sample(NAMES, 5)
        dm = rng.choice(STATUS2)
        m = rng.randrange(8, 12)
        C1, C2, C3 = 13, 18, 23
        others = [n for n in NAMES if n not in (D, S, M1, M2, T)]
        fill = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
                for i in range(N_BODY)}
        body = dict(fill)
        body[DEC_IDX] = f"register {D} set to state {dm}"
        body[m] = f"register {S} set to state ONLINE"      # role layout is donor-invariant
        body[C1] = f"register {M1} mirrors register {S}"
        body[C2] = f"register {M2} mirrors register {M1}"
        body[C3] = f"register {T} mirrors register {M2}"
        roles = {DEC_IDX: "decoy", m: "source", C1: "root", C2: "ref1", C3: "ref2"}
        outA.append({
            "it": it, "body": body, "roles": roles,
            "q_root": f"What state is register {M1} in?",
            "q_full": f"What state is register {T} in?",
            "oracle_root": [DEC_IDX, C1], "oracle_full": [DEC_IDX, C1, C2, C3],
        })

        # ---- Part B: two grounded chains, swapped leaf ----
        D2, Sa, Sb, Ma, Mb, T2 = rng.sample(NAMES, 6)
        va, vb = rng.choice(STATUS2), rng.choice(STATUS2)
        others2 = [n for n in NAMES if n not in (D2, Sa, Sb, Ma, Mb, T2)]
        fill2 = {i: f"register {rng.choice(others2)} refreshed to {rng.randrange(100, 999)}"
                 for i in range(N_BODY)}
        bodyB = dict(fill2)
        for _lk in ("LA", "LB"):                            # two rng draws, as in kv_causal
            bodyB[DEC_IDX] = f"register {D2} set to state {rng.choice(STATUS2)}"
        bodyB[6] = f"register {Sa} set to state {va}"
        bodyB[8] = f"register {Sb} set to state {vb}"
        bodyB[13] = f"register {Ma} mirrors register {Sa}"
        bodyB[15] = f"register {Mb} mirrors register {Sb}"
        bodyB[23] = f"register {T2} mirrors register {Ma}"  # leaf LA
        outB.append({
            "it": it, "body": bodyB,
            "roles": {DEC_IDX: "decoy", 6: "source", 8: "src_off", 13: "root",
                      15: "root_off", 23: "leaf"},
            "q": f"What state is register {T2} in?",
        })
    return outA, outB


def to_segments(body: dict) -> list[Segment]:
    return [Segment(seg_id=str(i), turn=i,
                    text=f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n")
            for i in range(N_BODY)]


def overlap(query: str, seg: Segment) -> int:
    return len(set(_tok(query)) & set(_tok(seg.text)))


def rank_roles(router, query: str, segs: list[Segment], roles: dict):
    """Full ranking -> per-role rank (0-based) and raw lexical overlap."""
    order = router.select(query, segs, len(segs))
    pos = {sid: r for r, sid in enumerate(order)}
    out = {}
    for slot, role in roles.items():
        out[role] = {"rank": pos[str(slot)], "score": overlap(query, segs[slot])}
    fillers = [pos[str(i)] for i in range(N_BODY) if i not in roles]
    out["_filler_best"] = {"rank": min(fillers),
                           "score": max(overlap(query, segs[i])
                                        for i in range(N_BODY) if i not in roles)}
    out["_order"] = order
    return out


def oracle_recall(recs: list[dict], oracle_slots: list[int]):
    """P(the retriever's top-k contains the oracle's served set) -- the proposal's
    P(the correct event is selected), and the term §5 sets to 1 by fiat."""
    out = {}
    for k in KS:
        hits, full = [], 0
        for r in recs:
            top = set(r["_order"][:k])
            got = sum(1 for s in oracle_slots if str(s) in top)
            hits.append(got / len(oracle_slots))
            full += int(got == len(oracle_slots))
        out[k] = (sum(hits) / len(hits), full / len(recs))
    return out


def summarize(tag: str, recs: list[dict], roles_of_interest: list[str],
              oracle_slots: list[int] | None = None, lexical: bool = True):
    # NOTE: the `overlap` column is bag-of-words overlap with the query -- a property of the TEXT,
    # printed for every router. It is the ranking signal only for LexicalRouter, so the tie
    # diagnosis below is emitted for that router alone; under dense retrieval identical overlap
    # says nothing about why two events were ordered as they were.
    print(f"\n=== {tag}  (n={len(recs)})")
    print(f"{'role':<12} {'mean rank':>10} {'overlap':>8} " +
          " ".join(f"{'r@'+str(k):>6}" for k in KS))
    for role in roles_of_interest:
        rs = [r[role]["rank"] for r in recs if role in r]
        sc = [r[role]["score"] for r in recs if role in r]
        if not rs:
            continue
        rec = [sum(1 for x in rs if x < k) / len(rs) for k in KS]
        print(f"{role:<12} {sum(rs)/len(rs):>10.2f} {sum(sc)/len(sc):>8.2f} " +
              " ".join(f"{v:>6.2f}" for v in rec))
    if "root" in roles_of_interest and "source" in roles_of_interest:
        ok = [r for r in recs if "root" in r and "source" in r]
        above = sum(1 for r in ok if r["root"]["rank"] < r["source"]["rank"]) / len(ok)
        line = f"  P(root ranked above source) = {above:.3f}"
        if lexical:
            tied = sum(1 for r in ok if r["root"]["score"] == r["source"]["score"]) / len(ok)
            line += (f"   |   P(identical overlap score) = {tied:.3f}"
                     f"   <- if both are high, the ordering is a RECENCY artifact, not content")
        print(line)
    if oracle_slots:
        orc = oracle_recall(recs, oracle_slots)
        print(f"  oracle set {oracle_slots}: " + "  ".join(
            f"@{k} frac={f:.2f} exact={e:.2f}" for k, (f, e) in orc.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3900)
    ap.add_argument("--qform", choices=["bare", "trace", "both"], default="both")
    ap.add_argument("--embed", action="store_true", help="add EmbeddingRouter (needs GPU)")
    ap.add_argument("--out", default="./out/addr.json")
    args = ap.parse_args()

    A, B = build_items(args.seed, args.items)
    routers = {"lexical": LexicalRouter()}
    if args.embed:
        from kvmemory.components import EmbeddingRouter
        from kvmemory.embed import QwenEmbedder
        routers["embedding"] = EmbeddingRouter(QwenEmbedder(), use_text=True)

    suffix = {"bare": "", "trace": " Answer ONLINE or OFFLINE." + TRACE}
    forms = ["bare", "trace"] if args.qform == "both" else [args.qform]
    dump = defaultdict(list)

    for rname, router in routers.items():
        for form in forms:
            sfx = suffix[form]
            ROLES_A = ["source", "root", "ref1", "ref2", "decoy", "_filler_best"]
            for qkey, okey in (("q_root", "oracle_root"), ("q_full", "oracle_full")):
                recs = [rank_roles(router, it[qkey] + sfx, to_segments(it["body"]), it["roles"])
                        for it in A]
                dump[f"A|{rname}|{form}|{qkey}"] = recs
                summarize(f"A {rname} / {form} / {qkey}", recs, ROLES_A, A[0][okey],
                          lexical=(rname == "lexical"))

            recsB = [rank_roles(router, it["q"] + sfx, to_segments(it["body"]), it["roles"])
                     for it in B]
            dump[f"B|{rname}|{form}|q"] = recsB
            summarize(f"B {rname} / {form} / leaf-chain", recsB,
                      ["source", "root", "leaf", "root_off", "src_off", "decoy",
                       "_filler_best"], [DEC_IDX, 13, 15, 23], lexical=(rname == "lexical"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "runs": {k: v for k, v in dump.items()}},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("ADDR_DONE", flush=True)


if __name__ == "__main__":
    main()
