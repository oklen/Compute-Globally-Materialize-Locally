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
| v2 probes | Qwen3-32B | `Qwen/Qwen3-32B` | larger reader in App. J (design B) and App. L; one 80 GB card |
| v2 probes | Ministral-3-8B-Instruct (BF16 release) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | the Ministral-3 checkpoint of App. J and App. K |
| retrieval (v2) | Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | dense stage of §7 / App. I; `SPRAG_EMBED_PATH` or `./models/` |
| retrieval (v2) | bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | cross-encoder over the fused top-50 (§7 / App. I); `./models/` |

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

The v2 runs (2026-08 to 2026-10) also pulled `main`. For every repo they use, `main` was last
modified before those runs, so the revision below is the one they used (Qwen3-8B and Gemma-4-12B-it
are at the revisions above):

| model | HF repo | revision | `main` last modified |
|---|---|---|---|
| Qwen3-32B | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | 2025-07-26 |
| Ministral-3-8B-Instruct (BF16) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` | 2026-07-15 |
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 2026-04-20 |
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 2024-06-24 |

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

**v2 runs.** The real-dialog follow-up (§7, App. I) and §9 run Qwen3-8B on the 2025 stack (their
logs record torch 2.4.1+cu121 / transformers 4.51.3). The October 2026 probes (App. J–L) used:

```
# Qwen3-8B, Qwen3-32B and Ministral-3 in App. J; Ministral-3 in App. K
python 3.10, torch==2.6.0 (+cu124), transformers==5.7.0, tokenizers==0.22.2, accelerate==1.13.0
# App. L (AMA-Bench side channel): the same torch with the VeraKV harness's pins overlaid
transformers==4.51.3 tokenizers==0.21.1 huggingface_hub==0.30.2 accelerate==1.6.0
# Gemma-4-12B in App. J and App. K (loads as transformers' gemma4_unified model)
torch==2.7.1 (+cu126), transformers==5.10.4, torchvision==0.22.1
```

The environment of App. K's Qwen3-8B run was not recorded.

Hardware for the reported runs: a single GPU per job — **NVIDIA A100 80 GB** for the v1 runs and
most v2 runs; the E2 primary pass (App. I) ran on a 32 GB V100 (sm70), where the full-conversation
prefill needs `SPRAG_ENCODE_CHUNK=2048` (bf16 kept); the latency table of App. I is from one A800.

**LLM endpoint.** The LLM-written hard negatives (`kv_e3negs`, `kv_e2negs`, `kv_hnegs`), the LLM
labelling of mention turns (`rA_loose`), the cross-family re-judge (`kv_e2_rejudge`) and graph
ingestion (`rA_graph`) call an OpenAI-compatible endpoint configured by `JUDGE_*` (no defaults).
The reported runs used a commercial chat model from a non-Qwen family with reasoning disabled; a
different model will change the numbers that depend on it. The primary E2 judge and the App. L
judge are the local reader itself.

## Environment variables

| var | value | when |
|---|---|---|
| `SPRAG_MODEL_PATH` | local snapshot dir | always |
| `SPRAG_ATTN_IMPL` | `sdpa` | always |
| `SPRAG_ENCODE_CHUNK` | `4096` | long-context (Gemma-4 real-dialog); chunked prefill |
| `SPRAG_MAX_CTX` | `30000` | default context cap |
| `CUDA_VISIBLE_DEVICES` | `0` | single-GPU |
| `SPRAG_ENCODE_CHUNK` (E2) | `4096`; `2048` on the V100 and part of the CoT run | audited by `analysis/chunkaudit.py` |
| `SPRAG_DTYPE` | `bfloat16` (default; every reported number) | `float16` exists for sm70 GPUs and is a numerics change |
| `SPRAG_DEVICE_MAP` | unset (`{"": 0}`) | `auto` shards a model across GPUs; not used for reported runs |
| `SPRAG_ROPE_FACTOR` | unset | YaRN factor for the long-context stress test (§11) |
| `SPRAG_EMBED_PATH`, `SPRAG_EMBED_DEVICE` | Qwen3-Embedding-0.6B dir, `cuda:0` | `kv_addr`, `kv_addr_e2e` |
| `RA_DEVICE`, `RA_EMB_PATH` | device for the dense / rerank arms; embedder dir (default `./models/Qwen3-Embedding-0.6B`) | `rA_*`, `kv_e2sets`, `kv_e2natsets` |
| `KUZU_THREADS`, `KUZU_BUFFER_BYTES` | `4`, 1.5 GiB | `rA_graph` |
| `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`, `JUDGE_EXTRA_BODY` | OpenAI-compatible endpoint; extra JSON for the request body | see "LLM endpoint" |

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
| §7, App H -- capacity vs addressability, position control, reference arms | `kv_addr`, `kv_addr_e2e` |
| §7 -- real-dialog retrieval ranking (E1) | `rA_roles`, `rA_loose`, `rA_loose2labels`, `rA_rank`, `rA_mpos`, `rA_graph`, `rA_graph_plant` |
| §7 -- keeping rows vs a summary (TTFT, storage) | `kv_econ` |
| §8, App I -- repaired positive control (E3) | `kv_e3negs`, `kv_e3ctrl`, `kv_e3probe`, `kv_e3clean`, `kv_e3rank` |
| §8, App I -- retriever-selected rows (E2) | `kv_e2sets`, `kv_e2`, `kv_e2stats`, `kv_e2_rejudge`, `kv_e2power`, `kv_e2negs`, `kv_e2ro`, `kv_e2k`, `kv_e2cot5`, `kv_e2hot`, `kv_e2hot2`, `kv_e2floor` |
| §8, App I -- narrow form, value swap, latency | `kv_e2natsets`, `kv_e2swap`, `kv_e2cost` |
| §8, App I -- representation-level drift | `kv_e3div`, `kv_e3div2`, `kv_e3div3` |
| §9 -- consequences on HotpotQA agent trajectories | `kv_traj`, `kv_hnegs`, `kv_region`, `kv_dens`, `kv_ans`, `kv_cite` |
| App J -- carrier-row probes over the construction bank | `kv_pbprobe` |
| App J -- payload probes vs the model's readout | `kv_payprobe2`, `kv_payprobe3`, `kv_payprobe`, `x3_readout` |
| App K -- meaning vs wording on three checkpoints | `kv_phantom2` |
| §8, App L -- AMA-Bench side channel (8B, 32B) | `kv_sidechannel2`, `kv_sidechannel3` |

The analysis scripts that turn these outputs into the reported numbers are in `analysis/`
(inputs per script in `analysis/README.md`).

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
