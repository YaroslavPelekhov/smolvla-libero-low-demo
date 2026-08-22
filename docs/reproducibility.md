# Reproducibility protocol

This document is the operational counterpart of the four-page report. It distinguishes a fresh
execution from verification of the committed evidence.

## Frozen design

- target tasks: LIBERO-Goal simulator ids 0, 1, and 2;
- budgets: nested prefixes of 1, 2, 5, 10, and 25 demonstrations;
- training seeds: 17 and 29;
- evaluation: 20 paired initial states per task, checkpoint, and seed; reset seeds 1000/2000/3000/4000
  belong to separate registered phases;
- baseline: full action-expert fine-tuning with target-only moments;
- phase II: two 75/25 full-checkpoint interpolations per task/budget cell (CSWI);
- phase III: interpolation of target task vectors around seed-specific seen anchors (CSTVI);
- phase IV: CSTVI for 5/10 demos and a frozen dispersion gate for 25 demos (BDAM);
- ceiling extension: unchanged baseline/CSWI at 1/2/5/10/25 demos on reset seed 4000;
- action bridge: dataset gripper `1=open, 0=closed` to LIBERO `-1=open, +1=closed` via
  `g_env = 1 - 2*g_data`.

The predictions and stopping rules are content-addressed by
`artifacts/preregistration_v2.json` through `artifacts/preregistration_v5.json`, plus the separately
hashed `artifacts/preregistration_ceiling.json`.

## Fresh execution in WSL2

Keep datasets, checkpoints, and videos in the Linux filesystem. From the repository root:

```bash
sudo VLA_VENV_PATH=/home/$USER/.venvs/vla-budget \
  VLA_LEROBOT_SRC=/home/$USER/src/lerobot-vla-budget \
  bash scripts/bootstrap_wsl.sh

source /home/$USER/.venvs/vla-budget/bin/activate
export VLA_PROJECT=/mnt/c/path/to/world
export VLA_DATA=/home/$USER/data/LIBERO_LeRobot_v3
export VLA_OUTPUT=/home/$USER/outputs/vla-benchmark-final
export VLA_BASE=/home/$USER/models/smolvla_base
export VLA_VLM=/home/$USER/models/SmolVLM2-500M-Video-Instruct
bash scripts/download_assets_wsl.sh
bash scripts/run_submission_wsl.sh
bash scripts/run_v4_confirmatory_wsl.sh
bash scripts/run_v5_confirmatory_wsl.sh
bash scripts/run_ceiling_extension_wsl.sh
bash scripts/continue_postprocess_wsl.sh
```

All expensive units are resumable: a run is skipped only when its `COMPLETED` marker exists. The
first phase trains and evaluates baseline plus the SAN-FT ablation. The second phase constructs
content-addressed CSWI checkpoints, evaluates the 18 registered cells, aggregates the final matrix,
performs paired tests, and extracts real failures.

`run_ceiling_extension_wsl.sh` reuses the frozen seen and 5/10/25 checkpoints, trains only the
strict one- and two-demo prefixes, and evaluates one complete fresh-reset matrix. The script first
writes `protocol_manifest_ceiling.json` and content-addresses `predictions_ceiling.yaml`; all later
steps fail closed if either identity changes.

The ceiling config writes an operational checkpoint at step 8k and enables exact LeRobot resume.
Checkpoint cadence does not change examples, optimizer updates, scheduler, random seed, or the final
10k estimand; it prevents a host/WSL interruption from invalidating an otherwise deterministic unit.

The reference run completed all 72 ceiling evaluation cells. Its retained result is
64/95/118/115/111 baseline successes versus 66/92/115/111/110 CSWI successes at budgets
1/2/5/10/25. The strict preregistered superiority criterion is false. Runtime recovery details are
recorded separately in `artifacts/ceiling_runtime_incident.json`; recovery did not alter the
estimand or any preregistered hyperparameter.

`continue_postprocess_wsl.sh` is the exact Bonus B pipeline: it trains the seen-video temporal
critic, scores the same deterministic rollout subset with that critic and Robometer-4B, compares
rankings, and performs a finite-bank proxy-reward optimization audit. The audit selects one of four
frozen policies per task/budget by five proxy-scored videos and reveals success only on the other
15 simulator episodes after selection. It is deliberately labeled zeroth-order selection rather
than gradient policy learning.

## Local verification and report

```powershell
python -m pip install -e ".[dev]"
ruff check src tests report
pytest -q
vla-verify --without-pdf
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
vla-verify
```

The report builder derives all numerical tables and the vector cost curve from committed CSV/JSON,
runs XeLaTeX twice, rejects any page count other than four, and renders every page to PNG for visual
inspection. It never replaces absent measurements with zero.

## Evidence boundaries

The 500-step, LoRA, replanning, global-target-normalization, pre-fix gripper, checkpoint-distance,
CSWI-k25, and sign-consensus runs are diagnostics. They may motivate a hypothesis but are never
pooled into a confirmatory matrix. Results v3/v4/v5/ceiling contain 3,840 checked outcomes in total;
the negative matrices remain first-class evidence rather than being moved to `audit/`.
