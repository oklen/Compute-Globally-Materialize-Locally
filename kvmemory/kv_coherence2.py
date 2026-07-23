"""kv_coherence2.py -- OUT-OF-BAND cache coherence on the verified substrate (co-v2).

What gate v1/v2 settled: an update that is already IN the log needs no repair at all -- serving
the update event's own rows fixes `cur` at 1.00 for free (nat_update). But it costs history:
with the update rows in view, recall of the pre-update value drops .86 -> .64 (interference).
And version tags lifted the substrate: a stale carrier answers the historical question at .863.

So the only case where a synthetic patch is non-trivial is an update that arrives OUT-OF-BAND
(no log event exists), and the real question is not "can an appended patch fix `cur`" (text
competence says yes) but:

    Can an append-only patch fix `cur` WITHOUT paying the historical-interference cost --
    and does VERSION language (supersedes-v1) protect history better than a bare update?

Design (depth 1, TAGGED template from kv_cogate2, `hist` in step-phrasing where the substrate
scored .863):

  PAIRED DONORS: each base item is built once (names, fillers, decoy value dm, update value v2,
  source slot m all fixed) and run as TWO members differing ONLY in the dropped source's value
  v1 in {ONLINE, OFFLINE}. The source event gets a fixed position budget so every SERVED token
  sits at identical positions in both members; since the source is dropped, the served token
  ids are IDENTICAL across members -- any behavioral difference is causally the donor's.
  v2 is one independent draw per base, so each pair contributes exactly one changed (v1!=v2)
  and one noop (v1==v2) member.

  Served rows: {decoy@0, carrier@23} harvested from one causal prefill of the 24-event log.
  The update text (arms 2-5) is prefilled at QUERY time against the served cache, landing at
  top positions -- append-only, O(L_served x L_patch), stored rows untouched.

Arms (append text shown as the step-25 observation):
  stale         nothing appended                              -- floor / the hist reference
  upd_bare      register S set to state {v2}                  -- bare out-of-band update
  upd_ver       register S set to state {v2}   [S version 2]  -- versioned, in-domain
  patch_struct  <PATCH> source=S version=2 supersedes=1 new_state={v2} </PATCH>
  patch_der     ... affected=T derived_state={v2} </PATCH>    -- dependency-informed
  ctl_wrong     upd_ver for an unrelated register L           -- must NOT move T or D
  ctl_dummy     length-matched neutral note                   -- must NOT move anything
  txt_stale     text of the served events only                -- RAG floor
  txt_oracle    full log text + upd_ver appended              -- text competence ceiling

Raw answers (pick()) are stored, not accuracies: the correct `cur` gold differs by arm (v2 if
the arm delivered the update, v1 if it did not), and patch-follow rates on the dm != v2 subset
need the labels. Scoring happens in the analyzer.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_coherence2 --bases 16 --seed 1300 --out ./out/o2.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23
KV_ARMS = ["stale", "upd_bare", "upd_ver", "patch_struct", "patch_der", "noopx",
           "ctl_wrong", "ctl_dummy"]
TXT_ARMS = ["txt_stale", "txt_oracle"]
QUERIES = ["cur", "hist", "unrel"]

SEM = (
    "Semantics of this log. Steps are in chronological order. Every observation of a register "
    "is tagged with that register's version number: `[Kappa version 2]` marks Kappa's SECOND "
    "recorded state, which supersedes version 1 from that step onward; version 1 remains the "
    "correct answer for any question asked about an earlier step. Registers connected by "
    "`mirrors` are LIVE DERIVED VIEWS: if register X mirrors register Y, then X's current "
    "state is always Y's latest state, and X's state at any earlier step is Y's state as of "
    "that step. A `memory_patch()` step delivers an out-of-band correction with the same "
    "authority as a regular observation; `supersedes=1` means it replaces version 1 going "
    "forward without erasing it from the record.\n"
)


TPL = {"orig": {"ev": "<step {i}>\naction: reg_op()\nobservation: {body}\n",
                "pstep": "<step {i}>\naction: memory_patch()\nobservation: {body}\n",
                "unit": "step",
                "state": "register {r} set to state {v}",
                "num": "register {r} refreshed to {v}",
                "mirror": "register {a} mirrors register {b}",
                "mirror_word": "mirrors",
                "noopx": "register {r} was re-checked: state remains {v}; this confirms "
                         "the earlier value"},
       # G8 frozen para surface (kv_frozen's family; mirror verb = "copies", the strongest
       # para-side write-through construction in the kv_parabank profile -- the substrate
       # must work for patch effects to be measurable)
       "para": {"ev": "<entry {i}>\nevent: system_poll()\nresult: {body}\n",
                "pstep": "<entry {i}>\nevent: memory_patch()\nresult: {body}\n",
                "unit": "entry",
                "state": "the status of register {r} was logged as {v}",
                "num": "a reading of {v} was recorded for register {r}",
                "mirror": "register {a} copies register {b}",
                "mirror_word": "copies",
                "noopx": "a re-check of register {r} found the state unchanged at {v}; "
                         "the earlier value is confirmed"}}
CURTPL = {"t": "orig"}


def ev(i, body):
    return TPL[CURTPL["t"]]["ev"].format(i=i + 1, body=body)


def pstep(body):
    return TPL[CURTPL["t"]]["pstep"].format(i=N_BODY + 1, body=body)


def phr(kind, **kw):
    return TPL[CURTPL["t"]][kind].format(**kw)


def pick(o):
    u = o.upper()
    a, b = u.find("ONLINE"), u.find("OFFLINE")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "OFFLINE"
    if b < 0:
        return "ONLINE"
    return "ONLINE" if a < b else "OFFLINE"


def gen_base(rng):
    D, S, T, L = rng.sample(NAMES, 4)
    dm = rng.choice(STATUS2)
    v2 = rng.choice(STATUS2)                 # one draw per base: independent of BOTH donors
    m = rng.randrange(N_BODY // 3, (2 * N_BODY) // 3)
    others = [n for n in NAMES if n not in (D, S, T, L)]
    fill = {}
    for i in range(N_BODY):
        if i in (DEC_IDX, m, CAR_IDX):
            continue
        o = rng.choice(others)
        fill[i] = phr("num", r=o, v=rng.randrange(100, 999)) + f"   [{o} version 1]"
    A = " Answer ONLINE or OFFLINE:"
    q = {
        "cur": (f"\n\nQuestion: What state is register {T} in now, after every step recorded "
                f"above?{A}"),
        "hist": (f"\n\nQuestion: What state was register {T} in as of "
                 f"{TPL[CURTPL['t']]['unit']} {CAR_IDX+1}?{A}"),
        "unrel": (f"\n\nQuestion: What state is register {D} in?{A}"),
    }
    return {"D": D, "S": S, "T": T, "L": L, "dm": dm, "v2": v2, "m": m, "fill": fill, "q": q}


def member_events(base, v1):
    b = {}
    b.update(base["fill"])
    b[DEC_IDX] = phr("state", r=base["D"], v=base["dm"]) + f"   [{base['D']} version 1]"
    b[base["m"]] = phr("state", r=base["S"], v=v1) + f"   [{base['S']} version 1]"
    b[CAR_IDX] = phr("mirror", a=base["T"], b=base["S"])
    return [ev(i, b[i]) for i in range(N_BODY)]


def patch_texts(base, v1):
    S, T, L, v2 = base["S"], base["T"], base["L"], base["v2"]
    return {
        "upd_bare": pstep(phr("state", r=S, v=v2)),
        "upd_ver": pstep(phr("state", r=S, v=v2) + f"   [{S} version 2]"),
        "patch_struct": pstep(f"<PATCH> source={S} version=2 supersedes=1 "
                              f"new_state={v2} </PATCH>"),
        "patch_der": pstep(f"<PATCH> source={S} version=2 supersedes=1 new_state={v2} "
                           f"affected={T} derived_state={v2} </PATCH>"),
        "ctl_wrong": pstep(phr("state", r=L, v=v2) + f"   [{L} version 2]"),
        "ctl_dummy": pstep("routine note: the scheduled maintenance sweep completed its "
                           "pass without incident"),
        # G4: explicitly-unchanged confirmation (no update implicature, stays version 1);
        # interpret on the changed==0 stratum (on changed==1 its content is false -- control)
        "noopx": pstep(phr("noopx", r=S, v=v1) + f"   [{S} version 1 confirmed]"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1300)
    ap.add_argument("--tpl", default="orig", choices=["orig", "para"])
    ap.add_argument("--out", default="./out/o2.json")
    args = ap.parse_args()
    CURTPL["t"] = args.tpl
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    sem = SEM.replace("`mirrors`", "`" + TPL[args.tpl]["mirror_word"] + "`")
    head_full = head + HDR + sem
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    rows = []

    for bi in range(args.bases):
        base = gen_base(rng)

        members = {v1: member_events(base, v1) for v1 in STATUS2}
        eids = {v1: [list(llm.tok(t, add_special_tokens=False).input_ids)
                     for t in members[v1]] for v1 in STATUS2}
        budget = max(len(eids[v1][base["m"]]) for v1 in STATUS2)

        for v1 in STATUS2:
            patches = patch_texts(base, v1)
            plens = {k: len(llm.tok(v, add_special_tokens=False).input_ids)
                     for k, v in patches.items()}
            evs, ei = members[v1], eids[v1]
            spans, cur = [], H
            for i, e in enumerate(ei):
                spans.append((cur, cur + len(e)))
                cur += budget if i == base["m"] else len(e)
            total = cur
            pos, rowmap, r = [], {}, 0
            for i in range(N_BODY):
                for pp in range(spans[i][0], spans[i][0] + len(ei[i])):
                    pos.append(pp)
                    rowmap[pp] = r
                    r += 1
            flat = [t for e in ei for t in e]
            full_kv = encode_block(llm, H_ids + flat, list(range(H)) + pos, keep_a=H)
            crows = [pp for i in (DEC_IDX, CAR_IDX)
                     for pp in range(spans[i][0], spans[i][0] + len(ei[i]))]
            rt = torch.tensor([rowmap[pp] for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            rec = {"base": bi, "v1": v1, "v2": base["v2"], "dm": base["dm"],
                   "changed": int(v1 != base["v2"]), "plens": plens}

            for qk in QUERIES:
                qt = base["q"][qk]
                for arm in KV_ARMS:
                    pre = "" if arm == "stale" else patches[arm]
                    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                    o, _, _ = llm._greedy_pos(c, p, llm._ids(pre + qt + tail), 8)
                    del c
                    rec[f"{arm}__{qk}"] = pick(o)
                for arm in TXT_ARMS:
                    if arm == "txt_stale":
                        t = (head_full + "".join(evs[i] for i in (DEC_IDX, CAR_IDX))
                             + qt + tail)
                    else:
                        t = head_full + "".join(evs) + patches["upd_ver"] + qt + tail
                    o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(t), 8)
                    rec[f"{arm}__{qk}"] = pick(o)

            del full_kv, sub
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[o2] b{bi} v1={v1} chg={rec['changed']} | "
                  f"stale:c{rec['stale__cur']} h{rec['stale__hist']} | "
                  f"upd_ver:c{rec['upd_ver__cur']} h{rec['upd_ver__hist']} | "
                  f"der:c{rec['patch_der__cur']} h{rec['patch_der__hist']}", flush=True)

    json.dump({"rows": rows, "tpl": args.tpl}, open(args.out, "w"), indent=1)
    n = len(rows)
    print(f"\n===== co-v2 quick means (n={n}; full scoring in analyzer) =====")
    for arm in KV_ARMS + TXT_ARMS:
        gc = sum(1 for r in rows
                 if r[f"{arm}__cur"] == (r["v2"] if arm not in
                                         ("stale", "ctl_wrong", "ctl_dummy", "txt_stale",
                                          "noopx")
                                         else r["v1"])) / n
        gh = sum(1 for r in rows if r[f"{arm}__hist"] == r["v1"]) / n
        gu = sum(1 for r in rows if r[f"{arm}__unrel"] == r["dm"]) / n
        print(f"  {arm:12s} cur*={gc:.2f} hist={gh:.2f} unrel={gu:.2f}")
    print("O2_DONE", flush=True)


if __name__ == "__main__":
    main()
