"""x5_analyze.py -- summarize kv_pbprobe units: per model x design x construction.

Columns: kv_gen (paper protocol, source dropped), txt_gen (source present), kv_fc_auc / txt_fc_auc (forced-choice
logit, threshold-free), V_car / V_prev / V_dec / H_read (linear probe, best layer). Max over layers inflates chance a
little; V_dec (before every source) and, in design B, V_prev (cannot see which source is named) go through the same
max, so they show where the floor sits.

    python x5_analyze.py ../out/x5 > ../out/x5/summary.md
"""
import json
import os
import sys

PAPER = {  # materialize Table 2 write-through: Qwen3 / Ministral-3 / Gemma-4
    "follows": (.83, .81, .55), "mirrors": (.80, .89, .52), "copies": (.78, .77, .55), "shadows": (.78, .95, .50),
    "kept_eq": (.72, .84, .58), "matches": (.64, .94, .67), "tracks": (.55, .45, .66), "synced": (.45, .48, .56),
    "consist": (.94, .95, .47), "inline": (.69, .56, .42), "accord": (.64, .62, .45), "record": (.58, .53, .45),
    "reflect": (.52, .64, .53), "update": (.52, .52, .44), "write": (.52, .56, .41), "assign": (.48, .53, .47)}
COL = {"qwen3": 0, "ministral3": 1, "gemma4": 2}
ORDER = list(PAPER)


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else None


def main(root):
    out = {}
    for tag in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(("q_", "logs"))):
        for design in ("A", "B"):
            rows = []
            for key in ORDER:
                f = os.path.join(root, tag, f"{design}_{key}.json")
                if not os.path.exists(f):
                    continue
                r = json.load(open(f))
                pb = r["probe"]
                rows.append(dict(key=key, fam=r["fam"], n=r["n"], kv_gen=r["kv_gen"], kv_gen_n=r.get("kv_gen_n", r["n"]),
                                 txt_gen=r["txt_gen"], kv_auc=r.get("kv_fc_auc"), txt_auc=r.get("txt_fc_auc"),
                                 V_car=max(pb["V_car"]), V_prev=max(pb["V_prev"]), V_dec=max(pb["V_dec"]),
                                 H_read=max(pb["H_read"]), chance=pb["chance"],
                                 paper=PAPER[key][COL[tag]] if tag in COL else None))
            if not rows:
                continue
            out[f"{tag}/{design}"] = rows
            print(f"\n### {tag} · design {design} · {len(rows)}/16 constructions\n")
            print("| construction | n | kv_gen | paper | txt_gen | kv AUC | txt AUC | V carrier | V preceding | V decoy | read-time hidden |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for x in rows:
                pp = f"{x['paper']:.2f}" if x["paper"] is not None else "—"
                print(f"| {x['key']} | {x['n']} | {x['kv_gen']:.2f} ({x['kv_gen_n']}) | {pp} | {x['txt_gen']:.2f} | "
                      f"{x['kv_auc']} | {x['txt_auc']} | {x['V_car']:.2f} | {x['V_prev']:.2f} | {x['V_dec']:.2f} | {x['H_read']:.2f} |")
            g = [x["kv_gen"] for x in rows]
            for col in ("V_car", "H_read", "kv_auc"):
                v = [x[col] for x in rows]
                if None in v:
                    continue
                print(f"\n- {col}: {min(v):.2f}–{max(v):.2f}; Spearman with kv_gen {spearman(g, v)}")
            print(f"- kv_gen: {min(g):.2f}–{max(g):.2f}; V_prev {min(x['V_prev'] for x in rows):.2f}–{max(x['V_prev'] for x in rows):.2f}; "
                  f"V_dec {min(x['V_dec'] for x in rows):.2f}–{max(x['V_dec'] for x in rows):.2f}")
            if rows[0]["paper"] is not None:
                print(f"- kv_gen vs paper Table 2: Spearman {spearman(g, [x['paper'] for x in rows])}")
    json.dump(out, open(os.path.join(root, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../out/x5")
