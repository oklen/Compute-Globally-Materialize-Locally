# Data

The synthetic experiments (donor-paired trajectories) are self-contained and need no data. The others
read from here:

- **LoCoMo** — download `locomo10.json` and place it at `data/locomo10.json`. Used by the X8 audits
  (`x8_audit2`, `kv_harvest3`, `kv_harvest4`; override with `--lc_path`), the real-dialog retrieval
  ranking (`rA_*`), and the retriever-selected-rows runs (`kv_e2*`, `kv_e3*`).
- **REALTALK** — licensed for **evaluation only**; obtain it from the authors and place the corpus
  under `data/REALTALK/` (override with `--rt_path`). Used by the X8 audits and the repaired control
  (`kv_e3*`).
- **HotpotQA** — the distractor-setting validation split (`hotpot_qa`, config `distractor`), one JSON
  object per line, at `data/hotpotqa/hotpot_dev_distractor.jsonl`. Each record needs `id`, `question`,
  `answer`, `context` (`title`, `sentences`) and `supporting_facts` (`title`), as in the HF dataset. Used by §9 (`kv_traj`,
  `kv_hnegs`, `kv_region`, `kv_dens`, `kv_ans`, `kv_cite`).
- **AMA-Bench** — the test split as JSONL at `data/ama_test.jsonl`, one episode per line
  (`episode_id`, `domain`, `task_type`, `num_turns`, `total_tokens`, `success`, `trajectory` of
  `{turn_idx, action, observation}`, `qa_pairs` of `{question, answer, type, question_uuid}`). Used by
  App. L (`kv_sidechannel2`, `kv_sidechannel3`).

Model directories for the retrieval runs (`Qwen3-Embedding-0.6B/`, `bge-reranker-v2-m3/`) go under
`./models/`, not here.
