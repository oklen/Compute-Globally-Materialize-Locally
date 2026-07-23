"""kv_parabank.py -- FROZEN paraphrase bank: which surface constructions trigger event-local
write-through, and does the model understand the ones that don't?

Verdict-3 P0-2: the paraphrase result (mirrors / "set according to" strong; tracks / "in line
with" at chance) kills the strong "predicate semantics alone decides" claim. The upgraded claim
to test: carryover is CONSTRUCTION-conditioned -- the write gate depends on semantic relation
x surface form. The decisive control is a full-context arm on the same item: if the model
answers correctly with everything visible but the sparse-KV arm is at chance, the model
UNDERSTANDS the construction yet does not write the value through into the event span.

Design (frozen before any run; the bank is fixed, no post-hoc additions or deletions):
  Two cell families, both inherited byte-for-byte from kv_frozen except the carrier wording:
    mirror family: source "register S set to state {v}" DROPPED at serve time; carrier at
      slot 23 = "register {T} <construction> register {S}"; serve {decoy, carrier}; ask T.
      8 constructions: mirrors / tracks / copies / matches / follows / shadows /
                       is kept equal to / is synchronized with
    flag family: operand + explicit verdict DROPPED; carrier at slot 23 = a flag-writing
      construction; serve {decoy, carrier}; ask EXCEEDED/NORMAL.
      8 constructions: accord (set according to the check result)      [anchor: strong]
                       inline (updated in line with the check outcome) [anchor: weak]
                       reflect / record / assign / write / consist / update
  Arms per (item, construction), ONE encode each:
    kv : serve {DEC, CAR} only        -> write-through measurement
    txt: serve ALL event spans        -> full-context understanding control (source visible)
  Readout: trace prompt + menu + 160 tokens + pick-last, uniform across models and cells
  (family-robust protocol; per-construction comparison is within model, and txt normalizes).
  Analysis contract: report per-construction (kv_acc, txt_acc) with construction as the unit
  of variation (no pooling across the bank into a single "paraphrase" number); the headline
  object is the PROFILE. Classes: written-through (kv high, txt high) / understands-but-no-
  write-through (kv ~chance, txt high) / not-understood (txt low; excluded from claims).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_parabank --items 1 --seed 999 --out ./out/pb.json
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

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23
TRACE = (" Locate the relevant statement, then end your answer with the final state word.")

BANK_MIRROR = [
    ("mirrors", "register {a} mirrors register {b}"),
    ("tracks", "register {a} tracks register {b}"),
    ("copies", "register {a} copies register {b}"),
    ("matches", "register {a} matches register {b}"),
    ("follows", "register {a} follows register {b}"),
    ("shadows", "register {a} shadows register {b}"),
    ("kept_eq", "register {a} is kept equal to register {b}"),
    ("synced", "register {a} is synchronized with register {b}"),
]
BANK_FLAG = [
    ("accord", "the alert status flag for sensor {r} was set according to the check result"),
    ("inline", "the alarm flag for {r} was updated in line with the check outcome"),
    ("reflect", "the alert flag for sensor {r} was set to reflect the check result"),
    ("record", "the alert flag for sensor {r} now records the outcome of the check"),
    ("assign", "sensor {r}'s alert flag was assigned based on the threshold comparison"),
    ("write", "the alert flag for sensor {r} was written from the check verdict"),
    ("consist", "the alert flag for sensor {r} was made consistent with the check result"),
    ("update", "following the check, the alert flag for sensor {r} was updated accordingly"),
]


def ev(i, body):
    return f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n"


def pickw_last(o, words):
    u = o.upper()
    hits = [(u.rfind(w), w) for w in words if u.rfind(w) >= 0]
    return max(hits)[1] if hits else None


def build(rng, specials, n=N_BODY):
    others = [x for x in NAMES if x not in specials.get("_names", [])]
    body = {i: t for i, t in specials.items() if i != "_names"}
    for i in range(n):
        if i not in body:
            body[i] = f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
    return [ev(i, body[i]) for i in range(n)]


def run2(llm, H_ids, H, H_kv, tail, evs, q, menu):
    """One encode, two serve sets: kv={DEC,CAR}, txt=ALL. Returns (kv_pick, txt_pick)."""
    eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
    spans, cur = [], H
    for e in eids:
        spans.append((cur, cur + len(e)))
        cur += len(e)
    flat = [t for e in eids for t in e]
    full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
    outs = []
    for served in ([DEC_IDX, CAR_IDX], list(range(len(evs)))):
        crows = [pp for i in served for pp in range(*spans[i])]
        rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
        sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
        c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
        o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 160)
        outs.append(pickw_last(o, menu))
        del c, sub
    del full_kv
    torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--out", default="./out/pb.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    rows = []

    for it in range(args.items):
        # ---- mirror family: one item skeleton, carrier construction sweeps the bank ----
        D, S, T = rng.sample(NAMES, 3)
        gold = rng.choice(STATUS2)
        m = rng.randrange(8, 16)
        qm = (f"\n\nQuestion: What state is register {T} in? "
              "Answer ONLINE or OFFLINE." + TRACE)
        for key, ctpl in BANK_MIRROR:
            sp = {DEC_IDX: f"register {D} completed a routine self-test",
                  m: f"register {S} set to state {gold}",
                  CAR_IDX: ctpl.format(a=T, b=S), "_names": [D, S, T]}
            kv_a, txt_a = run2(llm, H_ids, H, H_kv, tail, build(rng, sp), qm, STATUS2)
            rows.append({"it": it, "fam": "mirror", "key": key, "gold": gold,
                         "kv_a": kv_a, "txt_a": txt_a,
                         "kv": int(kv_a == gold), "txt": int(txt_a == gold)})

        # ---- flag family ----
        D2, S2 = rng.sample(NAMES, 2)
        gv = rng.randrange(100, 999)
        thr = rng.randrange(100, 999)
        while abs(gv - thr) < 30:
            thr = rng.randrange(100, 999)
        vgold = "EXCEEDED" if gv > thr else "NORMAL"
        m2 = rng.randrange(8, 14)
        qf = (f"\n\nQuestion: Was sensor {S2} above the alert threshold? "
              "Answer EXCEEDED or NORMAL." + TRACE)
        for key, ctpl in BANK_FLAG:
            sp = {DEC_IDX: f"register {D2} completed a routine self-test",
                  m2: f"sensor {S2} measured {gv}; the alert threshold is {thr}",
                  m2 + 1: f"alert check result for sensor {S2}: threshold {vgold}",
                  CAR_IDX: ctpl.format(r=S2), "_names": [D2, S2]}
            kv_a, txt_a = run2(llm, H_ids, H, H_kv, tail, build(rng, sp), qf,
                               ["EXCEEDED", "NORMAL"])
            rows.append({"it": it, "fam": "flag", "key": key, "gold": vgold,
                         "kv_a": kv_a, "txt_a": txt_a,
                         "kv": int(kv_a == vgold), "txt": int(txt_a == vgold)})
        done_m = [r for r in rows if r["it"] == it and r["fam"] == "mirror"]
        done_f = [r for r in rows if r["it"] == it and r["fam"] == "flag"]
        print(f"[pb] it{it} mirror kv={''.join(str(r['kv']) for r in done_m)} "
              f"txt={''.join(str(r['txt']) for r in done_m)} | "
              f"flag kv={''.join(str(r['kv']) for r in done_f)} "
              f"txt={''.join(str(r['txt']) for r in done_f)}", flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("\n===== PARAPHRASE BANK (per-construction kv / txt) =====")
    for fam, bank in (("mirror", BANK_MIRROR), ("flag", BANK_FLAG)):
        for key, _ in bank:
            g = [r for r in rows if r["fam"] == fam and r["key"] == key]
            if g:
                print(f"{fam:6s} {key:8s} kv={sum(r['kv'] for r in g)/len(g):.2f} "
                      f"txt={sum(r['txt'] for r in g)/len(g):.2f} n={len(g)}")
    print("PB_DONE", flush=True)


if __name__ == "__main__":
    main()
