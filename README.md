# Compute Globally, Materialize Locally
### The Memory Contract of Sparse Event-KV

Reproduction code for the paper *Compute Globally, Materialize Locally: The Memory Contract of
Sparse Event-KV* — Zefeng Cai, Zerui Cai (independent researchers).

📄 **Paper:** [arXiv](https://arxiv.org/abs/2607.23693)

## What this is

Long-horizon agents reuse their KV cache as memory: a serving system keeps a subset of cached
entries and drops the rest. This code tests whether a **retained event still carries state derived
from an evicted one** — the phenomenon we call **semantic materialization**.

Using **donor pairs** — two agent histories byte-identical in every served token and position,
differing only in one omitted observation — we find that a downstream event's cached rows can answer
from the omitted value even though the served text never states it (99:0 on Qwen3-8B). We
characterize this as a three-part **memory contract**: *trigger* (which events write the state),
*landing* (where the value lives — roots carry, reference edges route), and *access* (what readout
recovers it).

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
may need chunked prefill (`export SPRAG_ENCODE_CHUNK=4096`) and a recent `transformers`.

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
