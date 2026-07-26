# Model and environment manifest

The reported numbers were produced with the checkpoints and environments below. All models are run
in **bf16 compute** with **SDPA attention** and **greedy decoding**. Set `SPRAG_MODEL_PATH` to a
local snapshot of the relevant repo. We evaluate one released revision of each checkpoint; claims are
at the checkpoint / model-instance level, not the family level.

## Checkpoints

| role | model | HF repo | notes |
|---|---|---|---|
| **2025** | Qwen3-8B | `Qwen/Qwen3-8B` | full attention; carries most experiments |
| recent | Ministral-3-8B-Instruct | `mistralai/Ministral-3-8B-Instruct-2512` | `Mistral3ForConditionalGeneration`, loaded via `AutoModelForImageTextToText` on the **tf5 stack** (not the classic loader); shipped FP8, dequantized to bf16 compute |
| recent (2026) | Gemma-4-12B-it | `google/gemma-4-12B-it` | **sliding-window attention** (40/48 layers, window 1024); long-context uses chunked prefill |
| legacy (2024) | Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | exploratory diagnostics only |
| legacy (2024) | Gemma-2-9B | `google/gemma-2-9b-it` | exploratory diagnostics only |
| legacy (2024) | Mistral-7B-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | comprehension-failed; excluded from write claims |
| legacy (2024) | Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | exploratory diagnostics only |

Checkpoints were pulled from each repo's `main` branch during the 2026-07 experiment window. The
`main` commit of each, as of 2026-07-25, is:

| model | HF repo | revision (`main` @ 2026-07-25) |
|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Ministral-3-8B-Instruct | `mistralai/Ministral-3-8B-Instruct-2512` | `5b26027e7b19eeb4b7352e1fed3926375dd2cb4d` |
| Gemma-4-12B-it | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Gemma-2-9b-it | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |

Pin to these SHAs for replication. Two repos were updated inside the experiment window (Gemma-4 last
modified 2026-07-20, Ministral-3 2026-07-15), so individual earlier runs may have used an immediately
preceding commit; the SHAs above are the released `main` revisions we replicate against. To read a
local snapshot's own commit, call `huggingface_hub.snapshot_download(...)` (it returns the revision)
or read `<snapshot>/refs/main`.

## Environments

Two Python environments are used (the 2026-generation checkpoints need transformers 5).

**2025 stack** (Qwen3-8B on the classic `AutoModelForCausalLM` loader, and all four legacy models —
every synthetic run *except* the Ministral-3 and Gemma-4 runs, which use the tf5 stack):

```
python 3.11
torch==2.4.1            # +cu121
transformers==4.51.3
accelerate
```

**2026 (tf5) stack** (Gemma-4-12B and Ministral-3-8B, both loaded through
`AutoModelForImageTextToText`; `Mistral3ForConditionalGeneration` / Gemma-4 are not mapped under
`AutoModelForCausalLM`):

```
python 3.11
torch==2.9.1            # +cu128 (cu12x, so a 535 driver can run it)
transformers==5.12.1    # >=5.10
accelerate tokenizers huggingface_hub
```

Hardware for the reported runs: a single **NVIDIA A100-SXM4-80GB** per job.

## Environment variables

| var | value | when |
|---|---|---|
| `SPRAG_MODEL_PATH` | local snapshot dir | always |
| `SPRAG_ATTN_IMPL` | `sdpa` | always |
| `SPRAG_ENCODE_CHUNK` | `4096` | long-context (Gemma-4 real-dialog); chunked prefill |
| `SPRAG_MAX_CTX` | `30000` | default context cap |
| `CUDA_VISIBLE_DEVICES` | `0` | single-GPU |

## Experiments -> scripts -> paper

| paper element | script(s) |
|---|---|
| Table 1 / Fig 1 -- discovery 99:0, swap probe, overwrite | `kv_causal`, `kv_phantom` |
| Table 2 / Fig 5 -- 16-construction write bank, logit probe | `kv_parabank`, `kv_bankprobe` |
| §4 regularity (1) -- length- and background-controlled bank rerun | `kv_parabank2` |
| Table 3, Table 4 / Fig 2 -- landing 2x2 factorial, true edge, multi-hop | `kv_causal2`, `kv_causal3`, `kv_vardecomp`, `kv_refcarrier` |
| App B -- where in a mirror chain the value sits (depth-3 localisation) | `kv_hops` |
| App G -- X11 position-controlled replication | `kv_poscontrol` |
| Fig 3 -- access cardinality envelope, decoy audit (App E) | `kv_sweepmenu`, `kv_bicap`, `kv_digit`, `kv_noteknob2`, `kv_decoyctl2`, `kv_fail2` |
| App B -- two orthogonal bits over the four-way payload | `kv_orthobit2` |
| Table 7/8/9, Table 10/11 / Fig 4 -- X9 carriers, X10 serve-set, legacy panel | `kv_mater2`, `kv_mater3` |
| Table 5, Table 6 -- real-dialog passive harvest + injected-carrier control (X8) | `kv_harvest4`, `x8_audit2`, `kv_harvest3` |
| Table 6 (Gemma-4 rows), App D -- native original-position serving vs. the compact path | `kv_harvnative` |
| App A -- KV-path compatibility gate (numerical) | `kv_ident`, `kv_identm` |
| **Table 12 / App D -- sliding-window oracle gate (Gemma-4)** | **`kv_swgate`** |
| App F -- update patch vs recomputation, timing | `kv_coherence2`, `kv_recomp`, `kv_timing` |
| Robustness (fresh seeds / untuned paraphrase template) | `kv_frozen` |

Each module prints a runnable example in its docstring, e.g.:

```bash
SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa CUDA_VISIBLE_DEVICES=0 \
    python -m kvmemory.kv_causal --items 24 --seed 3900 --out ./out/cz.json
```

The sliding-window oracle gate (App D) must run on the tf5 stack:

```bash
SPRAG_MODEL_PATH=/path/to/Gemma-4-12B SPRAG_ATTN_IMPL=sdpa SPRAG_ENCODE_CHUNK=4096 \
CUDA_VISIBLE_DEVICES=0 tf5env/bin/python -m kvmemory.kv_swgate --items 32 --gap 2600 --seed 1
```

`--gap` is the filler budget between the far row and the near row, not the query-to-far-row
distance the paper's Table 12 reports; each run records the realised distance per item as
`gap_far` (`--gap 120 / 1200 / 2600` realise ~200 / ~1590 / ~3430).
