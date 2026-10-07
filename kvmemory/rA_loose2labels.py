"""rA_loose2labels.py -- rewrite rA_labels.json with the LLM-judged roles, so rA_rank can measure
pi under the loose regime with no other change.

NOTE ON FRAMING.  The plan called strict and loose the two ENDPOINTS of an interval, on the
assumption that the LLM pass can only ADD rows the lexical rules missed.  The measurement says
otherwise: over 1,531 questions it added 1,786 value rows and 10,203 mention rows, but mean |V|
went 1.72 -> 1.64 and mean |M| 12.54 -> 11.60.  It is also REMOVING rows -- substring V hits that
do not actually state the answer, and turns that share two content words while being about
something else.  So this is not a one-sided widening and must not be reported as a bound; it is a
second, differently-calibrated labelling, and pi is reported under both regimes.

    PYTHONPATH=. python -m kvmemory.rA_loose2labels --labels ./out/rA_labels.json \\
        --loose ./out/rA_loose.json --out ./out/rA_labels_loose.json
"""
from __future__ import annotations

import argparse
import json
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--loose", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = json.load(open(args.labels))
    loose = {(str(r["conv"]), r["qi"]): r for r in json.load(open(args.loose))["records"]}
    kept, dropped = [], 0
    for r in base["records"]:
        l = loose.get((str(r["conv"]), r["qi"]))
        if not l or not l["V_loose"]:
            dropped += 1        # no value row under this regime -> pi is undefined for the item
            continue
        r = dict(r)
        r["V"] = {k: list(l["V_loose"]) for k in ("N0", "N1", "N2", "N3")}
        r["nV"] = {k: len(l["V_loose"]) for k in ("N0", "N1", "N2", "N3")}
        r["M_strict"] = [i for i in l["M_loose"] if i not in set(l["V_loose"])]
        r["nM"] = len(r["M_strict"])
        r["regime"] = "loose"
        kept.append(r)
    base["records"] = kept
    base["regime"] = "loose"
    json.dump(base, open(args.out, "w"))
    print(f"[loose2labels] {len(kept)} items kept, {dropped} dropped (no value row under loose)")
    print(f"  mean |V| {statistics.fmean(len(r['V']['N1']) for r in kept):.2f}  "
          f"mean |M| {statistics.fmean(r['nM'] for r in kept):.2f}")
    print("LOOSE2LABELS_DONE")


if __name__ == "__main__":
    main()
