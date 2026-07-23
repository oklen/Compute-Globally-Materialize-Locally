# Data

The real-dialog experiments (`x8_audit2`, `kv_harvest3`, `kv_harvest4`) read two corpora from here.
Override the locations with `--lc_path` / `--rt_path`.

- **LoCoMo** — download `locomo10.json` and place it at `data/locomo10.json`.
- **REALTALK** — licensed for **evaluation only**; obtain it from the authors and place the corpus
  under `data/REALTALK/`.

All other experiments are self-contained (synthetic donor-paired trajectories) and need no data.
