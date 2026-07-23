"""kv_harvest3.py -- X8 main experiment: source-omitted three-arm test on REAL dialogue.

Question: does hidden carryover (encode-time absorption of earlier facts into later turns'
KV) exist and carry usable information in real multi-session dialogue? REALTALK = human-human
primary; LOCOMO = controlled comparison (LLM-generated, human-edited).

Frozen QA selection (byte-identical to x8_audit2 strict rules):
  cats 1-3, non-empty gold, official evidence dia_ids resolvable. s_last = session of the
  LAST evidence turn. Carrier candidates = turns in sessions > s_last, not evidence turns,
  sharing >= 2 content words with the question, gold substring absent from the turn AND its
  +/-1-turn window. strong flag = a shared token has document frequency <= 2 over sessions.
  QA qualifies iff >= 1 candidate. Carriers served: top-K (K=4) by deterministic rank
  (strong desc, n_shared desc, session asc, dia asc).

Arms (identical question + readout; carrier TEXT identical across text/iso/harv):
  none      header + question only                              (floor)
  text      carrier turns as visible text                       (RAG baseline, compacted)
  iso       carrier turns' KV, each turn encoded ALONE at its original positions (control:
            cannot contain anything absorbed from the omitted history)
  harv      carrier turns' rows gathered from the ONE full-conversation prefill (the only
            arm that can carry encode-time absorption)
  goldev    official evidence turns as visible text             (oracle reference)
Evidence turns are NEVER served in none/text/iso/harv.

Readout: "Answer with as few words as possible" + 24-token greedy; paired same-model judge
(cross_judge.judge); all answers saved for later cross-family re-judging.

Pre-registered contrasts: PRIMARY harv - iso on the strong stratum, cluster = conversation
(REALTALK additionally: dyad-level + leave-one-participant-out; 3 participant components
disclosed). Secondary: harv - text, all arms vs none/goldev. Prediction: hidden carryover
=> harv > iso; if harv ~= iso, the honest negative (natural mentions rarely absorb enough;
consistent with construction-conditioned write-through).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_harvest3 --ds realtalk --shard 0 --nshards 8 \
        --max_qa 2 --out ./out/h3.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.cross_judge import judge

INSTR = "Answer with as few words as possible (a name, a date, a short phrase)."
K_CARRIERS = 4
_TOKR = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(("the a an to of and in is are was were on at for with as by it this that "
                   "be been you your what which who how when where did does do i we they he "
                   "she from about their his her have has had will would can could there "
                   "not no yes but or if so its it's im i'm".split()))


def toks(s):
    return [t for t in _TOKR.findall(s.lower()) if t not in _STOP and len(t) > 2]


def load_realtalk(path):
    convs = []
    for f in sorted(glob.glob(os.path.join(path, "Chat_*.json"))):
        d = json.load(open(f))
        turns = []
        for k in sorted((k for k in d if re.fullmatch(r"session_\d+", k)),
                        key=lambda k: int(k.split("_")[1])):
            sn = int(k.split("_")[1])
            ts = d.get(f"session_{sn}_date_time", "")
            for t in d[k]:
                txt = (t.get("clean_text") or "").strip()
                if not txt:
                    continue
                turns.append((sn, t.get("dia_id", ""),
                              f"[session {sn} {ts}] {t.get('speaker','')}: {txt}"))
        convs.append({"cid": os.path.basename(f)[:-5].replace("Chat_", "C"),
                      "turns": turns, "qa": d.get("qa", [])})
    return convs


def load_locomo(path):
    data = json.load(open(path))
    convs = []
    for i, el in enumerate(data):
        conv = el["conversation"]
        turns = []
        for k in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                        key=lambda k: int(k.split("_")[1])):
            sn = int(k.split("_")[1])
            ts = conv.get(f"session_{sn}_date_time", "")
            for t in conv[k]:
                utt = (t.get("text") or "").strip()
                cap = t.get("blip_caption")
                if cap:
                    utt += f"  [shares an image: {cap}]"
                if not utt:
                    continue
                turns.append((sn, t.get("dia_id", ""),
                              f"[session {sn} {ts}] {t.get('speaker','')}: {utt}"))
        convs.append({"cid": f"L{i}", "turns": turns, "qa": el.get("qa", [])})
    return convs


def select_qa(cv):
    """Frozen strict selection; returns list of dicts with carriers + goldev turn idxs."""
    turns = cv["turns"]
    dia2idx = {d: i for i, (s, d, _) in enumerate(turns) if d}
    sess_text = defaultdict(list)
    for s, _, b in turns:
        sess_text[s].append(b.lower())
    df = Counter()
    for s in sess_text:
        for w in set(toks(" ".join(sess_text[s]))):
            df[w] += 1
    out = []
    for qa in cv["qa"]:
        cat = qa.get("category", 0)
        gold = str(qa.get("answer", "")).strip()
        if cat not in (1, 2, 3) or not gold:
            continue
        gl = gold.lower()
        ev = [e for e in (qa.get("evidence") or []) if e in dia2idx]
        if not ev:
            continue
        s_last = max(turns[dia2idx[e]][0] for e in ev)
        qtok = set(toks(qa["question"]))
        evset = set(ev)
        cands = []
        for idx, (s, d, b) in enumerate(turns):
            if s <= s_last or d in evset:
                continue
            bl = b.lower()
            share = qtok & set(toks(bl))
            if len(share) < 2 or gl in bl:
                continue
            wl = " ".join(t[2].lower() for t in turns[max(0, idx-1):idx+2])
            if gl in wl:
                continue
            strong = any(df.get(w, 99) <= 2 for w in share)
            cands.append((not strong, -len(share), s, d, idx))
        if not cands:
            continue
        cands.sort()
        chosen = [c[4] for c in cands[:K_CARRIERS]]
        out.append({"cat": cat, "q": qa["question"], "gold": gold,
                    "strong": int(not cands[0][0]), "n_cand": len(cands),
                    "carriers": sorted(chosen), "s_last": s_last,
                    "evset": sorted(evset),
                    "goldev": sorted(dia2idx[e] for e in ev)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["realtalk", "locomo"], required=True)
    ap.add_argument("--rt_path", default="./data/REALTALK")
    ap.add_argument("--lc_path", default="./data/locomo10.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--max_qa", type=int, default=0, help="per conv cap, 0 = all")
    ap.add_argument("--win", type=int, default=0,
                    help="dose test: expand each carrier to +/-win turns (same session-"
                         "after-evidence + leak screens applied per window turn)")
    ap.add_argument("--out", default="./out/h3.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    convs = (load_realtalk(args.rt_path) if args.ds == "realtalk"
             else load_locomo(args.lc_path))
    convs = [c for i, c in enumerate(convs) if i % args.nshards == args.shard]
    rows = []

    for cv in convs:
        qas = select_qa(cv)
        if args.max_qa:
            qas = qas[: args.max_qa]
        if not qas:
            continue
        turns = cv["turns"]
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids)
                for _, _, t in turns]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        total = cur
        print(f"[h3] {cv['cid']} turns={len(turns)} tok={total} qa={len(qas)}", flush=True)
        flat = [t for e in eids for t in e]
        full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)

        for qa in qas:
            units = list(qa["carriers"])
            if args.win:
                gl = qa["gold"].lower()
                evs_set = set(qa["evset"])
                uni = set()
                for i in qa["carriers"]:
                    for j in range(i - args.win, i + args.win + 1):
                        if (0 <= j < len(turns) and turns[j][0] > qa["s_last"]
                                and turns[j][1] not in evs_set
                                and gl not in turns[j][2].lower()):
                            uni.add(j)
                units = sorted(uni)
            qt = f"\n\nQuestion: {qa['q']}\n{INSTR}" + tail
            qids = llm._ids(qt)
            rec = {"ds": args.ds, "conv": cv["cid"], "cat": qa["cat"],
                   "strong": qa["strong"], "n_cand": qa["n_cand"], "gold": qa["gold"],
                   "carrier_dia": [turns[i][1] for i in qa["carriers"]]}
            car_txt = "".join(turns[i][2] + "\n" for i in units)
            rec["n_units"] = len(units)
            gev_txt = "".join(turns[i][2] + "\n" for i in qa["goldev"])

            def gen_text(body):
                from transformers import DynamicCache
                ids = llm._ids(head + body + qt)
                o, _, _ = llm._greedy(DynamicCache(), 0, ids, 24)
                return o.strip()

            rec["ans_none"] = gen_text("")
            rec["ans_text"] = gen_text(car_txt)
            rec["ans_goldev"] = gen_text(gev_txt)

            crows = [pp for i in units for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(Kk.index_select(2, rt), Vv.index_select(2, rt)) for Kk, Vv in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, qids, 24)
            rec["ans_harv"] = o.strip()
            del c, sub

            iso_parts = []
            runs, cur_run = [], [units[0]]
            for i in units[1:]:
                if i == cur_run[-1] + 1:
                    cur_run.append(i)
                else:
                    runs.append(cur_run)
                    cur_run = [i]
            runs.append(cur_run)
            for run in runs:
                a, b = spans[run[0]][0], spans[run[-1]][1]
                ids_run = [t for i in run for t in eids[i]]
                kv_i = encode_block(llm, ids_run, list(range(a, b)))
                iso_parts.append((kv_i, list(range(a, b))))
            c, p = assemble(llm, [(H_kv, list(range(H)))] + iso_parts)
            o, _, _ = llm._greedy_pos(c, p, qids, 24)
            rec["ans_iso"] = o.strip()
            del c, iso_parts
            torch.cuda.empty_cache()

            for arm in ("none", "text", "iso", "harv", "goldev"):
                rec[f"j_{arm}"] = int(judge(llm, head, tail, qa["q"], qa["gold"],
                                            rec[f"ans_{arm}"]))
            rows.append(rec)
            print(f"[h3] {cv['cid']} cat{qa['cat']} s{qa['strong']} | "
                  f"n{rec['j_none']} t{rec['j_text']} i{rec['j_iso']} "
                  f"h{rec['j_harv']} g{rec['j_goldev']} | gold={qa['gold'][:30]}", flush=True)
        del full_kv
        torch.cuda.empty_cache()

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    if n:
        print(f"\n===== {args.ds} shard {args.shard} (n={n}) =====")
        for arm in ("none", "text", "iso", "harv", "goldev"):
            print(f"  {arm:7s} {sum(r[f'j_{arm}'] for r in rows)/n:.3f}")
        st = [r for r in rows if r["strong"]]
        if st:
            print(f"  strong stratum n={len(st)}: " + " ".join(
                f"{a}={sum(r[f'j_{a}'] for r in st)/len(st):.3f}"
                for a in ("none", "text", "iso", "harv", "goldev")))
    print("H3_DONE", flush=True)


if __name__ == "__main__":
    main()
