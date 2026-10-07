"""X3 multi-model summary: for each model x payload type, the best-layer probe accuracy beside the model's own menu-readout accuracy.

Usage: python3 x3m_summary.py <dir_with_{tag}_{kind}.json> [out.json]
Columns: chance / the model's own readout (source dropped, present) / probes: carrier hidden, carrier V, unrelated neutral row V, V that never saw the source.
For three-digit payloads the probe reads the hundreds digit, and 'greater than 500' is reported separately.
"""
import glob
import json
import os
import sys

d = sys.argv[1]
summ = {}
for p in sorted(glob.glob(os.path.join(d, "*_*.json"))):
    r = json.load(open(p))
    for kind, o in r.get("kinds", {}).items():
        tgt = "value" if "value" in o["probe"] else "hundreds"
        pt = o["probe"][tgt]
        row = {"chance": pt["chance"], "read_harv": o["readout"]["harv"], "read_full": o["readout"]["full"],
               "read_iso": o["readout"]["iso"],
               "hidden_harv": max(pt["hidden_harv"]), "hidden_iso": max(pt["hidden_iso"]),
               "V_harv": max(pt["V_harv"]), "V_neutral": max(pt["V_neutral"]), "V_iso": max(pt["V_iso"])}
        if kind == "num3":
            g = o["probe"]["gt500"]
            row.update({"gt500_chance": g["chance"], "gt500_V_harv": max(g["V_harv"]), "gt500_V_neutral": max(g["V_neutral"]),
                        "gt500_read_harv": o["readout_gt500"]["harv"], "gt500_read_full": o["readout_gt500"]["full"]})
        summ.setdefault(r["tag"], {})[kind] = row
for tag, kinds in summ.items():
    print(f"== {tag}")
    for kind in ("bin", "k4", "k8", "num3"):
        if kind not in kinds:
            continue
        x = kinds[kind]
        print(f"  {kind:4s} chance {x['chance']:.3f} | own readout dropped {x['read_harv']:.3f} present {x['read_full']:.3f} | "
              f"probe hidden {x['hidden_harv']:.3f} (no source {x['hidden_iso']:.3f}) V carrier {x['V_harv']:.3f} unrelated row {x['V_neutral']:.3f} no source {x['V_iso']:.3f}"
              + (f" | >500: read {x['gt500_read_harv']:.3f} V {x['gt500_V_harv']:.3f}" if kind == "num3" else ""))
if len(sys.argv) > 2:
    json.dump(summ, open(sys.argv[2], "w"), indent=1)
