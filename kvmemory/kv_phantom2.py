"""kv_phantom2.py -- meaning vs wording, settled with two donor-pair contrasts.

VeraKV probe (1) and Materialize (L298-301) both say a served row carries the donor's MEANING
(polarity 64%) but rarely its exact WORDING (~10%), and they describe the 8% control differently
(no-donor polarity floor vs isolated re-encoding verbatim rate). This run gives each claim its own
causal contrast, with the same donor-pair logic Materialize uses for 99:0:

  MEANING pairs  the two donors are words from OPPOSITE states (RUNNING vs HALTED). Among pairs whose
                 two harvest answers differ, count follow:anti on the STATE of the answer.
  WORDING pairs  the two donors are different words for the SAME state (RUNNING vs ONLINE). Among
                 pairs whose two harvest answers differ, count follow:anti on the WORD itself.

Within a pair every served token and position is identical; only the omitted prefix differs.

Arms per donor:
  harv  [H] + carrier rows harvested from ONE prefill of [H; prefix(donor); carrier]; prefix omitted
  full  [H; prefix(donor); carrier] all served -- source visible, the ceiling (wording SHOULD track)
  iso   [H] + carrier encoded with no prefix at all -- identical for both donors, so it can never
        separate a pair (greedy); it gives the model's default word, used to read verbatim rates

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_phantom2 --tag qwen3 --out ./out/x2/qwen3.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block
from kvmemory.kv_phantom import HDR, FILLERS, ids_of, serve_and_answer

# (object, field, state-A words, state-B words); sets are disjoint within an object
STATES = [
    ("server", "status", ["ONLINE", "RUNNING", "UP", "ACTIVE"], ["OFFLINE", "HALTED", "DOWN", "INACTIVE"]),
    ("database", "state", ["LOCKED", "SEALED", "FROZEN", "BLOCKED"], ["UNLOCKED", "OPEN", "WRITABLE", "FREE"]),
    ("firewall", "mode", ["ENABLED", "ARMED", "ON", "ENGAGED"], ["DISABLED", "DISARMED", "OFF", "BYPASSED"]),
    ("healthcheck", "result", ["PASSED", "GREEN", "HEALTHY", "OK"], ["FAILED", "RED", "BROKEN", "ERROR"]),
    ("license", "check", ["VALID", "CURRENT", "ACTIVE", "GOOD"], ["EXPIRED", "LAPSED", "INVALID", "VOID"]),
    ("backup_job", "status", ["COMPLETE", "FINISHED", "DONE", "SUCCEEDED"], ["ABORTED", "CANCELLED", "KILLED", "CRASHED"]),
    ("build", "verdict", ["PASSING", "SUCCESS", "CLEAN", "STABLE"], ["FAILING", "BROKEN", "DIRTY", "UNSTABLE"]),
    ("replica", "sync", ["CURRENT", "SYNCED", "FRESH", "ALIGNED"], ["STALE", "LAGGING", "BEHIND", "DRIFTED"]),
    ("quota", "check", ["GRANTED", "APPROVED", "ALLOWED", "ACCEPTED"], ["DENIED", "REJECTED", "REFUSED", "BLOCKED"]),
    ("cache", "state", ["WARM", "HOT", "PRIMED", "LOADED"], ["COLD", "EMPTY", "FLUSHED", "CLEARED"]),
    ("disk", "state", ["MOUNTED", "ATTACHED", "CONNECTED", "LINKED"], ["DETACHED", "UNMOUNTED", "DISCONNECTED", "REMOVED"]),
    ("cert", "check", ["TRUSTED", "VERIFIED", "SIGNED", "APPROVED"], ["REVOKED", "UNTRUSTED", "REJECTED", "FORGED"]),
    ("daemon", "status", ["LIVE", "ALIVE", "RUNNING", "AWAKE"], ["DEAD", "STOPPED", "CRASHED", "ASLEEP"]),
    ("channel", "mode", ["SECURE", "ENCRYPTED", "PRIVATE", "SEALED"], ["EXPOSED", "PLAIN", "PUBLIC", "OPEN"]),
    ("volume", "state", ["ATTACHED", "BOUND", "CLAIMED", "HELD"], ["RELEASED", "FREED", "UNBOUND", "DROPPED"]),
    ("pipeline", "verdict", ["PASS", "APPROVED", "CLEARED", "ACCEPTED"], ["FAIL", "REJECTED", "HALTED", "BLOCKED"]),
]

_W = re.compile(r"[A-Za-z_]+")


def first_word(out: str) -> str:
    m = _W.search(out or "")
    return m.group(0).upper() if m else ""


def prefix_text(obj, field, note, v):
    return (f"<step 1>\naction: check_{obj}()\nobservation: {obj} {field} = {v}\n"
            f"<step 2>\naction: log_note('{note}')\nobservation: ok\n")


def carrier_text(obj, field):
    return (f"<step 3>\naction: record_to_report()\nobservation: the {obj} {field} from the step 1 "
            f"check was copied into the incident report for the auditor.\n")


def question(obj, field):
    return (f"\n\nQuestion: According to the step 1 check, what was the {obj} {field}? "
            f"Answer with the single status word only:")


def run_pair(llm, tail, H_ids, H_kv, obj, field, note, d1, d2):
    """Both donors of one pair -> per-arm first words. Returns None if prefixes differ in length."""
    p1, p2 = ids_of(llm, prefix_text(obj, field, note, d1)), ids_of(llm, prefix_text(obj, field, note, d2))
    if len(p1) != len(p2):
        return None
    H, P = len(H_ids), len(p1)
    E_ids = ids_of(llm, carrier_text(obj, field))
    est, een = H + P, H + P + len(E_ids)
    q = question(obj, field)
    res = {}
    for tag, pref in (("1", p1), ("2", p2)):
        allpos = list(range(H + P)) + list(range(est, een))
        harv_kv = encode_block(llm, H_ids + pref + E_ids, allpos, keep_a=H + P)
        res["harv" + tag] = serve_and_answer(llm, [(H_kv, list(range(H))), (harv_kv, list(range(est, een)))], q, tail, ntok=8)
        full_kv = encode_block(llm, H_ids + pref + E_ids, allpos, keep_a=H)
        res["full" + tag] = serve_and_answer(llm, [(H_kv, list(range(H))), (full_kv, list(range(H, een)))], q, tail, ntok=8)
    iso_kv = encode_block(llm, H_ids + E_ids, list(range(H)) + list(range(est, een)), keep_a=H)
    res["iso"] = serve_and_answer(llm, [(H_kv, list(range(H))), (iso_kv, list(range(est, een)))], q, tail, ntok=8)
    return {k: v.strip()[:40] for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--n_meaning", type=int, default=4, help="opposite-state donor pairs per object per filler")
    ap.add_argument("--n_wording", type=int, default=4, help="same-state donor pairs per object per filler")
    ap.add_argument("--fillers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="./out/x2/out.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = ids_of(llm, head + HDR)
    H_kv = encode_block(llm, H_ids, list(range(len(H_ids))))

    rows = []
    states = STATES[:1] if args.smoke else STATES
    for si, (obj, field, A, B) in enumerate(states):
        for fi in range(args.fillers):
            note = FILLERS[(si + fi) % len(FILLERS)].format(obj=obj)
            meaning = [(a, b) for a in A for b in B]
            rng.shuffle(meaning)
            wording = [(x, y) for S in (A, B) for x, y in itertools.combinations(S, 2)]
            rng.shuffle(wording)
            for kind, cands, want in (("meaning", meaning, args.n_meaning), ("wording", wording, args.n_wording)):
                got = 0
                for d1, d2 in cands:
                    if got >= (args.smoke or want):
                        break
                    if rng.random() < 0.5:   # balance which donor is "1"
                        d1, d2 = d2, d1
                    r = run_pair(llm, tail, H_ids, H_kv, obj, field, note, d1, d2)
                    if r is None:
                        continue
                    got += 1
                    row = {"kind": kind, "obj": obj, "field": field, "filler": fi, "d1": d1, "d2": d2,
                           "A": A, "B": B, **r}
                    for k in ("harv1", "harv2", "full1", "full2", "iso"):
                        row["w_" + k] = first_word(r[k])
                    rows.append(row)
                    print(f"[{args.tag}] {kind:8s} {obj:12s} {d1:>10s}/{d2:<10s} harv {row['w_harv1']}/{row['w_harv2']} "
                          f"full {row['w_full1']}/{row['w_full2']} iso {row['w_iso']}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"tag": args.tag, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"X2_DONE tag={args.tag} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
