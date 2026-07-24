# Model and environment manifest

The reported numbers were produced with the checkpoints and environments below. All models are run
in **bf16 compute** with **SDPA attention** and **greedy decoding**. Set `SPRAG_MODEL_PATH` to a
local snapshot of the relevant repo. We evaluate one released revision of each checkpoint; claims are
at the checkpoint / model-instance level, not the family level.

## Checkpoints

| role | model | HF repo | notes |
|---|---|---|---|
| **primary (2025)** | Qwen3-8B | `Qwen/Qwen3-8B` | full attention; the headline results |
| recent (2025) | Ministral-3-8B-Instruct | `mistralai/Ministral-3-8B-Instruct-2512` | shipped FP8; dequantized to bf16 compute |
| recent (2026) | Gemma-4-12B-it | `google/gemma-4-12B-it` | **sliding-window attention** (40/48 layers, window 1024); long-context uses chunked prefill |
| legacy (2024) | Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | exploratory diagnostics only |
| legacy (2024) | Gemma-2-9B | `google/gemma-2-9b-it` | exploratory diagnostics only |
| legacy (2024) | Mistral-7B-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | comprehension-failed; excluded from write claims |
| legacy (2024) | Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | exploratory diagnostics only |

Snapshots were downloaded from the `main` revision at experiment time (2026-07). Record the exact
commit SHA of your snapshot (`huggingface_hub.snapshot_download` prints it, or read
`<snapshot>/.cache/huggingface`) if you need bit-level provenance.

## Environments

Two Python environments are used (the 2026-generation checkpoints need transformers 5).

**2025 stack** (Qwen3-8B, Ministral-3 on the classic loader, all legacy models, all synthetic runs):

```
python 3.11
torch==2.4.1            # +cu121
transformers==4.51.3
accelerate
```

**2026 (tf5) stack** (Gemma-4-12B, and tf5 loaders):

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
| Table 1 / Fig 1 -- discovery 99:0, overwrite | `kv_causal`, `kv_phantom` |
| Table 2 / Fig 2(appx) -- 16-construction write bank | `kv_parabank`, `kv_bankprobe` |
| Table 3 / Fig 2 -- landing 2x2 factorial, true edge, X11 | `kv_causal2`, `kv_causal3`, `kv_poscontrol` |
| Fig 3 -- access cardinality envelope | `kv_sweepmenu`, `kv_orthobit2`, `kv_bicap`, `kv_digit`, `kv_noteknob2`, `kv_decoyctl2` |
| Table 7/8/9 / Fig 4 -- X9 carriers, X10 serve-set | `kv_mater2`, `kv_mater3` |
| Table 6 -- real-dialog passive harvest (X8) | `kv_harvest4`, `x8_audit2`, `kv_harvest3` |
| App A -- KV-path compatibility gate (numerical) | `kv_ident`, `kv_identm` |
| **App D -- sliding-window oracle gate (Gemma-4)** | **`kv_swgate`** |
| App E -- update patch vs recomputation, timing | `kv_coherence2`, `kv_recomp`, `kv_timing` |

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
