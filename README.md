# Low-demo SmolVLA adaptation on LIBERO

This repository is a reproducible, leakage-controlled evaluation of SmolVLA adaptation to the first
three held-out `libero_goal` tasks. It contains the executed experiment matrix, immutable data split,
raw run metadata, measured results, failure videos, Bonus B evaluators, and a four-page report.

## Result

The requested matrix is complete: 2 seen-pretraining runs, 36 target-adaptation runs, and 48 evaluation
runs. Every evaluation run contains 20 simulator episodes, for 960 Bernoulli outcomes in total.

| Target demos | Zero-shot | Naive fine-tuning | LoRA + image augmentation |
|---:|---:|---:|---:|
| 0 | 0.0% (0/120) | - | - |
| 5 | - | 0.0% (0/120) | 0.0% (0/120) |
| 10 | - | 6.7% (8/120) | 0.0% (0/120) |
| 25 | - | 0.0% (0/120) | 0.0% (0/120) |

The proposed method did not beat the baseline. All eight successes came from task 0 under naive
fine-tuning with ten demonstrations (5/20 for seed 17 and 3/20 for seed 29). The non-monotone baseline
and complete LoRA failure are retained as measured; no result was imputed or replaced.

The correct-prompt and wrong-prompt zero-shot controls both achieved 0/120, so this experiment cannot
identify language sensitivity at the observed performance floor.

## Protocol

- Base policy: `lerobot/smolvla_base` (~450M parameters).
- Seen pool: the full available `nvidia/LIBERO_LeRobot_v3/libero_90` conversion: 73 task ids,
  3,921 episodes, and 569,249 frames.
- Seen training: 5,000 updates for seeds 17 and 29.
- Held-out tasks: simulator task ids 0-2 from `libero_goal`.
- Adaptation: 500 updates from the matching seen checkpoint for every task, budget, and seed.
- Baseline: ordinary action-expert fine-tuning.
- Proposed method: rank-16 LoRA (`alpha=16`) plus at most two mild random image transforms.
- Budgets: exact nested prefixes of the first 5, 10, and 25 dataset episodes.
- Evaluation: 20 episodes per task/checkpoint, 300-step horizon, reset seed 1000.

The 5,000/500 update counts are the only compute reduction. The requested tasks, budgets, two training
seeds, and 20-episode evaluation denominators were not reduced.

## Data identity audit

The simulator and converted dataset do not use the same task ordering. Target demonstrations are
therefore joined by exact canonical instruction rather than numeric id:

| Simulator id | Dataset task index | Canonical instruction | First five episode ids |
|---:|---:|---|---|
| 0 | 9 | open the middle drawer of the cabinet | 20, 26, 31, 42, 58 |
| 1 | 7 | put the bowl on the stove | 13, 15, 16, 22, 36 |
| 2 | 4 | put the wine bottle on top of the cabinet | 6, 10, 17, 18, 25 |

The public 40-task checkpoint was not used because its training universe overlaps the held-out goal
tasks. The exact prefixes and prompt mapping are frozen in `artifacts/protocol_manifest.json`.

## Environment

The completed run used Windows 11 with WSL2 Ubuntu and headless EGL on an RTX 4070 Ti SUPER 16 GB:

- Python 3.12.3;
- PyTorch 2.7.1+cu126 and TorchCodec 0.5;
- LeRobot commit `9e14584904d14eb79dc960b27bb40220f85bb993`;
- hf-libero 0.1.4, MuJoCo 3.8.1, PEFT 0.20.0.

Install the pinned environment inside WSL:

```bash
sudo VLA_VENV_PATH=/home/$USER/.venvs/vla-budget \
  VLA_LEROBOT_SRC=/home/$USER/src/lerobot-vla-budget \
  bash scripts/bootstrap_wsl.sh

source /home/$USER/.venvs/vla-budget/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
vla-doctor --config configs/benchmark.yaml --environment
```

Keep video datasets and checkpoints in the native Linux filesystem; decoding through `/mnt/c` is
substantially slower.

## Reproduction

Create and freeze the protocol before any policy run:

```bash
vla-download --config configs/benchmark.yaml --suite both
vla-manifest --config configs/benchmark.yaml
vla-preregister --config configs/benchmark.yaml --predictions predictions.yaml
ruff check src tests report
pytest -q
```

Run the complete matrix:

```bash
VLA_PROJECT=/mnt/c/path/to/world \
VLA_DATA=/home/$USER/data/LIBERO_LeRobot_v3 \
VLA_OUTPUT=/home/$USER/outputs/vla-benchmark \
VLA_BASE=/home/$USER/models/smolvla_base \
VLA_VLM=/home/$USER/models/SmolVLM2-500M-Video-Instruct \
bash scripts/run_benchmark_wsl.sh
```

Each unit writes `command.txt`, `run_meta.json`, and `COMPLETED` only after success. `--skip-completed`
makes interrupted runs resumable without overwriting completed evidence. Evaluation can be parallelized,
but the final three units were run serially after the NVIDIA driver returned a transient CUDA error
under four concurrent jobs; the failed-attempt logs are retained outside the aggregation tree.

Aggregate and extract failures:

```bash
vla-aggregate --config configs/benchmark.yaml \
  --eval-root /home/$USER/outputs/vla-benchmark/eval
vla-failures --eval-root /home/$USER/outputs/vla-benchmark/eval \
  --output artifacts/results/failures
```

LeRobot records all 20 success outcomes per run but its v0.6.2 batched MP4 writer retained ten videos
per run (480 MP4s total). Failure analysis uses proposed-method runs, where every outcome is a failure.
Bonus B uses the same deterministic five-video subset per checkpoint and compares its scores with the
complete 20-episode checkpoint success rate.

## Bonus B: evaluation without simulator reward

The small critic is trained only on RGB from seen demonstrations with normalized temporal position as
the target; it never reads actions. Robometer-4B receives the same target rollout subset and task text.

```bash
vla-progress train \
  --dataset-root /home/$USER/data/LIBERO_LeRobot_v3/libero_90 \
  --output artifacts/results/progress_critic.pt --epochs 3 --stride 16
vla-progress score \
  --checkpoint artifacts/results/progress_critic.pt \
  --eval-root /home/$USER/outputs/vla-benchmark/eval \
  --output artifacts/results/progress_scores.json --max-episodes-per-run 5
vla-bonus robometer-video \
  --eval-root /home/$USER/outputs/vla-benchmark/eval \
  --model-path /home/$USER/models/Robometer-4B \
  --processor-path /home/$USER/models/Qwen3-VL-4B-Instruct \
  --output artifacts/results/robometer_scores.json --max-episodes-per-run 5
vla-bonus compare
```

Neither score is used to optimize the policy. A frozen simulator audit is required before doing so;
critic improvement with declining simulator success is the preregistered reward-hacking criterion.

## Repository outputs

- `artifacts/preregistration.json`: timestamped SHA-256 of predictions made before policy runs.
- `artifacts/protocol_manifest.json`: exact prompt join and episode prefixes.
- `artifacts/results/cost_curve.csv`: pooled measured curve with Wilson intervals.
- `artifacts/results/per_run.csv`: every task/seed cell and denominator.
- `artifacts/results/language_control.csv`: paired correct/wrong prompt control.
- `artifacts/results/failures/`: three real contact sheets, catalog, and source video paths.
- `artifacts/results/reward_ranking.json`: Bonus B checkpoint ranking.
- `docs/appendix.md`: implementation notes and interpretation constraints.
- `output/pdf/vla_budget_report.pdf`: final four-page report.

Rebuild the report with `python report/build_report.py`. The builder refuses to export when required
measured artifacts are missing.

References: [SmolVLA](https://arxiv.org/abs/2506.01844),
[LIBERO](https://arxiv.org/abs/2306.03310),
[LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero), and
[Robometer](https://robometer.github.io/).
