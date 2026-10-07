"""X4: on real agent trajectories, why does dropping the source still leave the answer correct?

Data: region.json (per-item output of kv_region.py; 247 Qwen3-8B HotpotQA agent trajectories).
Key fields (kv_region Q2): source segment s = the tool output where the answer value v first appears. The counterfactual stream swaps v for v' inside s only; downstream text is byte-identical.
  q2_allbutsrc_orig / _cf   gold log-odds with s dropped and everything else kept in place (original / counterfactual world)
  q2_allbutsrc_d            their difference = carry-over that can only come from KV (downstream text identical)
  q2_srconly_d              the same difference when only s is served (positive control)
  note_states_gold / v_in_other_tool_out / answer   whether a textual copy exists elsewhere

Usage: python3 x4_eviction_attribution.py region.json
"""
import json
import re
import statistics as st
import sys

R = [r for r in json.load(open(sys.argv[1]))["rows"] if r.get("q2_allbutsrc_orig") is not None]
STOP = set("the a an of in on at to and or for with by from is was are were be been its it's his her their this that these those".split())


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t not in STOP]


def text_copy(r):
    """A textual copy of the value exists elsewhere: the notes state it / another tool output has it / the answer has >= 50% of its content words."""
    if r.get("note_states_gold") or r.get("v_in_other_tool_out"):
        return 1
    vt, at = toks(r["v"]), set(toks(r.get("answer", "")))
    return int(bool(vt) and sum(t in at for t in vt) / len(vt) >= 0.5)


still = [r for r in R if r["q2_allbutsrc_orig"] > 0]
robust = [r for r in still if r["q2_allbutsrc_cf"] > 0]
copies = [r for r in still if text_copy(r)]
d = [r["q2_allbutsrc_d"] for r in R]
print(f"n={len(R)}")
print(f"still correct with the source segment dropped (gold vs hard-negative log-odds > 0): {len(still)}/{len(R)}")
print(f"of these, still correct when the dropped source carried a different value: {len(robust)}/{len(still)}")
print(f"of these, with a textual copy elsewhere (notes / other tool output / answer): {len(copies)}/{len(still)}")
print(f"KV carry-over (original - counterfactual, same downstream text): mean {st.mean(d):+.2f}, median {st.median(d):+.2f}; "
      f"positive control serving the source segment only {st.mean(r['q2_srconly_d'] for r in R):+.1f}; "
      f"placebo {st.mean(r['q2_placebo_d'] for r in R if r.get('q2_placebo_d') is not None):+.2f}"
      f" ({sum(r.get('q2_placebo_d') is not None for r in R)} items have a placebo)")
