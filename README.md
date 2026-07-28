# Compute Globally, Materialize Locally
### The Memory Contract of Sparse Event-KV

Zefeng Cai, Zerui Cai · independent researchers

📄 **Paper:** [arXiv:2607.23693](https://arxiv.org/abs/2607.23693) · this repository reproduces every
experiment in it.

## Abstract

Long-horizon agents increasingly reuse their KV cache as memory: a serving system keeps a subset of
cached entries and drops the rest. Eviction and episodic-memory schemes therefore rest on a premise
rarely tested directly, that a retained event is still informative once the observations that
produced it are gone. We test it by omitting one earlier observation from what is served, across
otherwise identical agent histories. Among items sensitive to that observation, the answer
overwhelmingly follows the omitted value, though no served span says which value is correct. We call
this **semantic materialization**: a downstream event's cached rows act as an independently servable
view of computation whose inputs are gone. It can also be written *on purpose*. A deliberately
phrased, answer-free event raises donor-aligned recovery from 6% to 51% on Qwen3-8B without ever
naming the value, whereas passively harvesting natural mentions from long-term dialog yields no
detected advantage. What such a row carries is specific and bounded. Compact state survives, larger
payloads decay toward chance, and whether a construction writes at all turns on phrasing rather than
on meaning alone, so two phrasings the model comprehends equally well can diverge sharply. The
result is a memory contract for sparse event-KV serving: what to write, where it lands, and what
survives once the source is gone. For anyone who evicts the corollary is that dropping a source
event and observing no accuracy loss does not show the source was unnecessary.

## The experiment, in brief

An agent's history is a sequence of events prefilled once, so every event's cache rows are
*contextualized* by everything before them. Serving that memory is sparse: a query touches a handful
of events, the system keeps those rows and drops the rest, routinely including the very observation
that produced the value being asked about.

So we drop that observation deliberately. One event writes `register S is ONLINE`. A later **root**
event says only `M mirrors S`, naming no value. We omit the first, serve the second, and ask for `M`:

```
full history    [ source: S = ONLINE ] .... [ root: "M mirrors S" ] ....
                           |                           ^
                           +- attends during prefill --+

served                 (dropped)                   [ root ] + decoy + query("M?")
```

Nothing in the served text names `ONLINE`. The model answers `ONLINE`, and switches to `OFFLINE`
when we flip the omitted write. That flip is a **donor pair**: two histories byte-identical in every
served token and position, differing only in the value the *omitted* event wrote. If the answer
tracks the flip, the served rows carry more than their visible text.

Accuracy alone cannot establish this, because it conflates genuine recall with priors and with
decoy-elimination. The donor pair, not accuracy, is therefore the unit of evidence throughout, and
decoys are sampled independently rather than as complements of the answer.

## What we find

**The phenomenon is real and directional.** Among donor-sensitive pairs the answer follows the
omitted source **99:0** on Qwen3-8B (exact sign test, *p* = 3.2×10⁻³⁰). A 256-pair replication gives
the prevalence behind that direction: 131 of 256 pairs are donor-dependent, and among those the
split is 130:0. The direction never reverses across serving cells, models, or readouts, and the
narrowest cell we measure anywhere is still 39:18. What crosses is state rather than string: a
hidden donor phrase transfers verbatim in only 9.7% of pairs, indistinguishable from the 8% of an
isolated re-encoding control, while its semantic polarity transfers in 64%.

**A three-part contract governs it.**

| Part | What it says | Where |
|---|---|---|
| **Trigger** *which events write* | Surface form, not meaning alone. Over a 16-construction bank comprehended at .98 mean (min .77), write-through spans chance to .95, and **no construction writes on all three recent checkpoints**. Materialization and its readout must be calibrated per model, not assumed. | §4 |
| **Landing** *where the value lives* | The **root** carries the dominant donor-aligned signal. A served **reference edge**, one hop further from the source, mostly routes a query back to the root instead of carrying the value itself. What a row yields depends on what is co-served with it. | §5 |
| **Access** *what a readout reaches* | A compact-state envelope, not a general channel. Binary state recovers at **.934** against .5 chance; four-way (.223) and eight-way (.156) sit near their chance rates; three-digit numbers are never recovered exactly (**0/192**). | §6 |

**The primitive is programmable, and that is the engineering handle.** A deliberately written,
answer-free carrier event lifts donor-aligned recovery from **6% to 51%** on Qwen3-8B without ever
naming the value. With the query pinned at a fixed absolute position, the shared downstream row
alone yields .00, adding the carrier beside it yields .42 under a natural-language compute
directive, and 1.00 under an explicit textual record. State needed after its operands leave the
serving set can be committed on purpose, calibrated per checkpoint, with explicit text as the
reliable fallback.

**Passive harvesting is not a dependable interface.** On REALTALK and LoCoMo long-term dialogs,
harvesting natural late mentions from the full-history cache yields no detectable benefit over
isolated encoding of the same text. On Gemma-4, served natively at original positions (the geometry
an eviction-style system actually presents), the LoCoMo cell is *equivalent* to isolated encoding
within a ±.05 band (−.011, 95% CI [−.026, +.004]). The controlled arms above use synthetic
donor-paired trajectories; the natural-dialog result bounds how far they generalise.

**One consequence for anyone who evicts.** An ablation that drops a source event and observes no
accuracy loss has not shown the source was unnecessary: it may have retained a row that already
carried the answer. Corrections then follow as served patches rather than recomputation, which is
also the cheap direction (77 ms to append a patch at *L* = 9.2k, against 1033 ms to recompute).

## Setup

```bash
pip install -r requirements.txt              # 2025 stack: torch 2.4.1 / transformers 4.51.3
export SPRAG_MODEL_PATH=/path/to/Qwen3-8B     # a local HuggingFace model dir (see "Models")
```

**Exact checkpoints, dtypes, both environment stacks (the 2026 Gemma-4 runs need a separate
transformers-5 env), GPU, env vars, and the experiment→figure/table map are in
[MODEL_AND_ENV_MANIFEST.md](MODEL_AND_ENV_MANIFEST.md).**

Each experiment is a module run with `python -m kvmemory.<name>`; results write to the `--out` path
(default under `./out/`). Greedy decoding throughout; the frozen model recovers state through its
native interface — no fine-tuning, no trained reader. A single GPU is sufficient for the recent
models.

### Models

Set `SPRAG_MODEL_PATH` to a local HuggingFace directory. The paper's recent models — which carry
the headline claims — are **Qwen3-8B**, **Ministral-3-8B**, and **Gemma-4-12B**; four
2024 checkpoints (**Qwen2.5-7B**, **Gemma-2-9B**, **Mistral-7B-v0.3**, **Llama-3.1-8B**) appear only
as exploratory legacy diagnostics. Most experiments run on Qwen3-8B; 2026 / long-context checkpoints
may need chunked prefill (`export SPRAG_ENCODE_CHUNK=4096`) and a recent `transformers`. Each model
is evaluated at one frozen revision, so claims are at the checkpoint level rather than the family
level.

### Data (real-dialog experiments only)

The X8 real-dialog audits read two corpora from `./data/` (override with `--lc_path` / `--rt_path`):

- **LoCoMo** — `locomo10.json` (public; place at `./data/locomo10.json`).
- **REALTALK** — licensed for evaluation only; **obtain separately** and place under `./data/REALTALK`.

## Experiments → scripts

| Paper section | Script(s) |
|---|---|
| **Discovery** — donor-pair 99:0, swap probe, overwrite | `kv_causal`, `kv_phantom` |
| **Trigger** — 16-construction write bank (mirror + flag families), logit probe; length- and background-controlled rerun | `kv_parabank`, `kv_bankprobe`, `kv_parabank2` |
| **Landing** — 2×2 factorial, true reference edge, X11 position-control, multi-hop | `kv_causal2`, `kv_causal3`, `kv_poscontrol`, `kv_vardecomp`, `kv_hops`, `kv_refcarrier` |
| **Access** — cardinality envelope, orthogonal bits, capacity, digits, compute-note, decoy audit | `kv_sweepmenu`, `kv_orthobit2`, `kv_bicap`, `kv_digit`, `kv_noteknob2`, `kv_decoyctl2` |
| **Programming** — X9 carrier arms; X10 serve-set ablation | `kv_mater2`, `kv_mater3` |
| **Real-dialog stress test** — X8 coverage audit, harvest, injected-carrier control; native (original-position, key-masked) serving vs. the compact path | `x8_audit2`, `kv_harvest3`, `kv_harvest4`, `kv_harvnative` |
| **Update side** — patch vs. recomputation, timing | `kv_coherence2`, `kv_recomp`, `kv_timing` |
| **Faithfulness gates** — KV-path compatibility, numerical + top-1 margin (App. A); Gemma-4 sliding-window oracle (App. D) | `kv_ident`, `kv_identm`, `kv_swgate` |
| Robustness (frozen seeds / untuned templates) | `kv_frozen` |

**Shared modules:** `llm_hf` (HF backend + KV reuse), `kv_matrix` (KV encode / assemble helpers),
`kv_select_smoke` (chat-template wrapper), `kv_vartrack` (step-vocabulary templates), `kv_fail2`
(forced-choice logit probe), `cross_judge` (LLM judge).

Each script's module docstring carries a runnable example, e.g.:

```bash
SPRAG_MODEL_PATH=/path/to/Qwen3-8B CUDA_VISIBLE_DEVICES=0 \
    python -m kvmemory.kv_causal --items 24 --seed 3900 --out ./out/cz.json
```

## Citation

```bibtex
@misc{cai2026materialize,
  title  = {Compute Globally, Materialize Locally: The Memory Contract of Sparse Event-KV},
  author = {Cai, Zefeng and Cai, Zerui},
  year   = {2026},
  note   = {https://arxiv.org/abs/2607.23693}
}
```

## License

MIT — see [LICENSE](LICENSE).
