# Analysis scripts (v2)

These scripts turn the outputs of the `kvmemory` modules into the numbers reported in the revision.
Run each one from inside `analysis/` after copying in the files it reads (the modules write them
under `./out/` by default). The outputs themselves are not shipped: several contain text derived from
REALTALK, which is licensed for evaluation only. `wcrfast.py` is a shared library: a wild-cluster
bootstrap-*t* over conversations that enumerates all 2^G sign vectors, so at G = 10 the smallest
attainable two-sided *p* is .002.

Labels in file names: **E1** is the real-dialog retrieval ranking (§7), **E2** the retriever-selected
rows (§8, App. I), **E3** the repaired positive control and the drift probe (§8, App. I). **X1–X5**
are the October 2026 runs (App. J–L and one §9 paragraph); they are unrelated to the X8–X11 labels of
the main README. Names such as `e2k_par.sh` in docstrings are the original runner scripts, which are
not included. Each script prints its own description of the readout.

| script | reads | reports |
|---|---|---|
| **E2 — judge, log-odds and budget sweep** | | |
| `an_e2k.py` | `e2_k.json` | the k = 1/2/3/5 sweep of the log-odds channels with intervals; the k = 1 null-by-construction check (Table "two channels") |
| `an1.py` … `an10.py` | `e2_k.json` (`an3`, `an6`, `an7` also `e2_sets.json`) | follow-ups on the sweep: consistency across k and against the placebo, answer flips by category and conversation, reproduction of the published k = 5 contrast, difference-in-differences, case listing, downstream-of-evidence rows, shrinkage toward the prior, a permutation test of the jitter, three arms on two rulers |
| `an_e2ro.py`, `an_e2ro2.py`, `an_e2ro3.py` | `e2_ro.json` | the log-odds readout against the R1 hard negative; the scale-versus-subset control; why the judged and log-odds readouts are different tasks |
| `an_span.py` | `e2_ro.json`, `e2_sets.json` | the joint-encoding gain by where the served rows sit |
| `rep1.py`, `rep2.py` | `e2_k.json`, `ro_sh?.log` | two independent launches agree on every recorded number (App. I determinism) |
| `tie1.py`, `tie2.py`, `f1a.py` | `e2_ro.json`, `ro_sh?.log` | the judge ties on 87% of items, both arms wrong on most ties; token-F1 per arm |
| `sc1.py`, `sc3.py`, `sc4.py` | `e2_ro.json` (`sc3`, `sc4` also `e2_k.json`, `e2_sets.json`) | answer length as a confound; harv − self per ruler and subgroup with the placebo |
| `an_e2floor.py`, `an_e2floor2.py` | `e2floor_full.log` (`an_e2floor` also `e2_rejudged.json`) | the `none` floor arm, the primary split and the cross-family re-judge; robustness to the truncated conversation |
| **E2 — chain-of-thought and hot window** | | |
| `cotparse2.py` → `cotres2.py`, `cotrob3.py` | `cot_*.log` → `cot_all.json` | the channel under a chain-of-thought readout (rebuilt from logs); a blind saturation filter |
| `chunkaudit.py` | `cot_*.log`, `cot_all.json`, `e2_k.json` | whether `SPRAG_ENCODE_CHUNK` drift during the CoT run moved anything (only `harv` can move) |
| `hotparse.py` → `hotres.py` | `hot_*.log` → `hot_all.json` | E2 at hot window 0/4/16 |
| `hotparse2.py` → `hotres2.py` | `hot2_*.log` → `hot2_all.json` | the deployed `bhot` arm: `bhot − self` at the deployed operating point |
| **E2 — narrow form, swap and cost** | | |
| `natres.py`, `natres_extra.py` | `nat_*.log`, `hot2_*.log`, `e2_natsets.json`, `e2_sets.json` | natural serve sets: 40.3% serve no value row; the narrow-form contrast (+1.87) and its robustness |
| `swapres.py` | `sw_sh*.log` | the value swap: original vs counterfactual world, interaction *I* |
| `natcost.py` | `e2cost.json` | latency per arm (build / TTFT) |
| **E3 — repaired control and drift** | | |
| `an_e3rank.py`, `an_e3rank2.py` | `e3_rank12.json`, `e3_rep12.json` | ranking readout over the candidate pool; reproduction; leave-one-conversation-out |
| `an_div.py`, `an_div2.py` | `e3div_*/e3div.*.json` (`an_div2` also `../data/locomo10.json`) | per-turn KV drift: floor, position guard, decay, selectivity |
| `div3_join2.py` → `sel3.py`, `sel4.py`, `sel5.py` | `e3_div3.json`, `../data/REALTALK`, `../data/locomo10.json` → `div3_join2.json` | delimiter vs content drift, residualized, with the placebo question; threshold robustness |
| **§9 — consequences** | | |
| `regionres.py` | `region.json` | the seven-region ablation and the source counterfactual |
| `x4_eviction_attribution.py` | `region.json` | X4: why dropping the source rarely costs accuracy (textual copies vs the cache channel) |
| `densres.py`, `densres2.py` … `densres5.py` | `dens.json` (path as argv[1]; `densres5` reads `dens.json` beside it) | attention-ranked eviction (Table "whose attention"); position control; the below-random lexical rankers; supporting facts |
| `ansres.py` | `ans.json` | the previous answer's KV as session memory |
| `citeres.py` | `cite.json`, `cite*.log` | counterfactual citation verification: decorative vs used-not-cited |
| **App. J–L — October 2026 runs** | | |
| `x5_analyze.py` | `kv_pbprobe` unit directory (default `../out/x5`) | X5: carrier-row probes per model × design × construction (Table "carrier row") |
| `x3m_summary.py` | directory of `kv_payprobe2`/`kv_payprobe3` outputs `{tag}_{kind}.json` | X3: payload probes beside the model's own readout (Table "payloads") |
| `x2_analyze.py`, `x2_polarity.py` | `kv_phantom2` outputs, one JSON per model | X2: meaning vs wording follow:anti counts; donor-word recurrence |
| `x1_analyze.py` | `ans_*.jsonl` from `kv_sidechannel2` / `kv_sidechannel3` | X1: AMA-Bench arms, accuracy differences and signed flips |
