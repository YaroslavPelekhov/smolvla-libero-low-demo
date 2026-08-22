# Low-demo adaptation of SmolVLA on LIBERO

This repository contains a preregistered, leakage-safe study of SmolVLA adaptation to three held-out
LIBERO-Goal tasks. The final ceiling extension evaluates the difficult 1/2-demo regime together with
5/10/25 demos on a fresh reset seed. Its conclusion is unambiguous: Cross-Seed Weight Interpolation
(**CSWI**) does not robustly improve ordinary target-only fine-tuning.

| Target demonstrations | 1 | 2 | 5 | 10 | 25 |
|---:|---:|---:|---:|---:|---:|
| Naive target-stat FT (seed 4000) | 64/120 | **95/120** | **118/120** | **115/120** | **111/120** |
| CSWI (seed 4000) | **66/120** | 92/120 | 115/120 | 111/120 | 110/120 |
| Paired change, percentage points | +1.7 | -2.5 | -2.5 | -3.3 | -0.8 |
| Exact McNemar `p` | 0.791 | 0.678 | 0.375 | 0.344 | 1.000 |

The preregistered criterion required a positive change at every budget. It is rejected at four of
five budgets; the only positive point, `k=1`, is two net successes and is not statistically resolved.
The task decomposition also exposes a floor: at `k=1`, task 0 remains 0/40 for both methods.

The earlier seed-1000 matrix remains useful as hypothesis-generating evidence: CSWI changed
`110/120, 114/120, 114/120` into `117/120, 114/120, 117/120`. Its 5-demo gain was +5.8 points
(10 wins, 3 losses; exact `p=0.092`), but CSTVI on seed 2000 changed the curve by
`+0.8/+5.0/-4.2` points and BDAM on seed 3000 by `-4.2/-2.5/-2.5` points. All negative confirmations
are retained; no favorable matrix is selected after the fact.

## What was actually tested

- Base policy: `lerobot/smolvla_base` (~450M parameters).
- Seen data: all available `nvidia/LIBERO_LeRobot_v3/libero_90` data (73 tasks, 3,921 episodes,
  569,249 frames).
- Held-out tasks: simulator task ids 0--2 from `libero_goal`.
- Budgets: immutable nested prefixes of 1, 2, 5, 10, and 25 demonstrations per task.
- Optimization: 100k seen updates, then 10k target updates; batch 8; seeds 17 and 29.
- Evaluation: 20 episodes per task/checkpoint, reset seeds 1000, 2000, 3000, and 4000; horizon 300.
- Checked evidence: 192 unique evaluation cells and 3,840 Boolean outcomes across CSWI, CSTVI, BDAM,
  their paired baselines, and two complete control blocks.
- Every follow-up has a content-addressed prediction file frozen before its evaluation.

For each task and budget, ordinary fine-tuning produces two branches, `w17` and `w29`. CSWI builds
two conservative interpolations:

```text
w'17 = 0.75 w17 + 0.25 w29
w'29 = 0.75 w29 + 0.25 w17
```

Every generated checkpoint records source and output SHA-256 hashes, tensor counts, shapes, dtypes,
and interpolation weights. Each anchor keeps its own target-only processor.

## The integration error that mattered

Two silent interface problems were found before final reporting.

1. LeRobot's episode filter did not restrict `meta/stats.json`. The first engineering run therefore
   normalized with all 52,042 target-suite frames. It is quarantined under `artifacts/audit/`.
2. The NVIDIA dataset stores the binary gripper as `1=open, 0=closed`, while LIBERO expects
   `-1=open, +1=closed`. Direct forwarding removed the close command without raising an exception.
   The pinned bridge implements `g_env = 1 - 2*g_data`.

On a causal checkpoint probe, fixing only the evaluation bridge changed 0/8 to 8/8; the registered
20-episode rerun produced 19/20. Training checkpoints were unaffected, but all 48 evaluation cells
were rerun. The patch is pinned at
[`patches/lerobot_libero_binary_gripper.patch`](patches/lerobot_libero_binary_gripper.patch).

## Repository map

```text
configs/benchmark.yaml          phase I: baseline and SAN-FT ablation
configs/final.yaml              phase II: registered CSWI follow-up (seed 1000)
configs/v4.yaml                 independent CSTVI check (seed 2000)
configs/v5.yaml                 independent BDAM check (seed 3000)
configs/ceiling.yaml            fresh 1/2/5/10/25 ceiling extension (seed 4000)
predictions_v2.yaml             frozen phase-I hypotheses
predictions_v3.yaml             frozen CSWI coefficient and stopping rule
predictions_ceiling.yaml        frozen low-demo ceiling hypotheses and stopping rule
artifacts/results_v3/           final baseline/CSWI evidence
artifacts/results_v4/           baseline/CSTVI robustness evidence
artifacts/results_v5/           baseline/BDAM robustness evidence
artifacts/results_ceiling/      full low-demo ceiling extension
artifacts/results/              valid SAN-FT ablation and Bonus B
artifacts/audit/                explicitly excluded diagnostics
src/vla_budget/                 experiment, aggregation, and verification code
report/main.tex                 four-page scientific report source
output/pdf/                     delivery PDF
```

The detailed [evidence index](artifacts/README.md) marks what is confirmatory, ablative, and
excluded. The [reproducibility protocol](docs/reproducibility.md) gives the exact fresh-run and
verification commands.

## Quick verification

Python 3.11 or 3.12 is supported.

```bash
python -m pip install -e ".[dev]"
ruff check src tests report
pytest -q
vla-verify
```

`vla-verify` checks five preregistration hashes, both protocol-manifest digests, all 192 retained cells,
the 20-outcome vectors, recomputed paired statistics for four matrices, three real failure artifacts,
42 Bonus B rankings, and the final PDF signature.

## Fresh experiment in WSL2

The reference environment is Windows 11 + WSL2 Ubuntu, headless EGL, and an RTX 4070 Ti SUPER 16 GB:
Python 3.12.3, PyTorch 2.7.1+cu126, TorchCodec 0.5, MuJoCo 3.8.1, `hf-libero` 0.1.4, PEFT 0.20.0,
and LeRobot commit `9e14584904d14eb79dc960b27bb40220f85bb993`.

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
# Independent checks after the seed-1000 phase:
bash scripts/run_v4_confirmatory_wsl.sh
bash scripts/run_v5_confirmatory_wsl.sh
# Assignment-recommended ceiling regime and Bonus B:
bash scripts/run_ceiling_extension_wsl.sh
bash scripts/continue_postprocess_wsl.sh
```

The runner is resumable and writes `COMPLETED` only after a successful unit. Training is serial;
evaluation defaults to two concurrent jobs for 16 GB VRAM.

## Exact command-to-task map

| Assignment item | Fresh command | Auditable output |
|---|---|---|
| 1. Seen, zero-shot, language, baseline | `bash scripts/run_submission_wsl.sh` | `artifacts/results_v3/` |
| 2. Method and ceiling regime | `bash scripts/run_ceiling_extension_wsl.sh` | `artifacts/results_ceiling/` |
| 3. Three real failures | included in `run_submission_wsl.sh` via `vla-failures` | `artifacts/results_v3/failures/` |
| 4B. Video reward and optimization audit | `bash scripts/continue_postprocess_wsl.sh` | `artifacts/results/reward_*.json` |
| 5. Predictions, graveyard, discussion | `powershell -File scripts/build_report.ps1` | `output/pdf/vla_budget_report.pdf` |

The GPU commands require WSL2/Linux because LIBERO is not executed natively on Windows. Unit tests,
evidence verification, and report generation are separate so a reviewer can audit committed results
without retraining 450M-parameter policies.

## Report

The four-page report is generated from committed CSV/JSON evidence. Tables and the vector cost curve
are rebuilt rather than transcribed by hand; missing data aborts the build.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
vla-verify
```

The delivery file is `output/pdf/vla_budget_report.pdf`. The build rejects any page count other than
four and renders every page through Poppler for visual QA.

## Bonus B: reward without simulator access

An action-free temporal critic trained on seen videos was compared with Robometer-4B on the same
deterministic rollout subset. Across 42 checkpoint cells, the small critic nearly collapsed (score
span 0.0013, Spearman rho -0.323), while Robometer reached rho 0.252 with span 0.298. Robometer is the
better ranker here, but the correlation is too modest for unaudited policy optimization. Neither
critic selected a confirmatory checkpoint or influenced CSWI.

The separate optimization audit then maximized each proxy over four frozen candidate policies in
each of nine task/budget groups. Five videos per candidate drive selection; true success is computed
only on the other 15 episodes. The small critic selected policies with 85.9% held-out success, 7.4
points below the 93.3% baseline reference, and missed the simulator oracle in 5/9 groups. Robometer
selected 91.1%, 2.2 points below baseline, with 5.9 points mean oracle regret. This is direct evidence
that maximizing the collapsed temporal proxy rewards the wrong policy; it is a finite-bank
zeroth-order test, not a claim of gradient-based SmolVLA optimization.

## Scope of the conclusion

The study covers three LIBERO-Goal tasks, two training seeds, four reset seeds, relative control,
and this exact v3 conversion. Zero-shot success is at the floor, so the language control is
inconclusive. CSWI can repair seed-specific failures on one matrix, but weight-space interpolation is
not behaviorally robust across reset distributions. A defensible superiority claim requires more
tasks, more paired states, and a frozen validation/evaluation split.

References: [SmolVLA](https://arxiv.org/abs/2506.01844),
[LIBERO](https://arxiv.org/abs/2306.03310),
[LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero), and
[Robometer](https://robometer.github.io/).
