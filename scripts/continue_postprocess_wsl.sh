#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
PROJECT=${VLA_PROJECT:-/mnt/c/Users/psgpe/Downloads/world}
VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
ROBOMETER=${VLA_ROBOMETER:-/home/root/models/Robometer-4B}
PROCESSOR=${VLA_ROBOMETER_PROCESSOR:-/home/root/models/Qwen3-VL-4B-Instruct}

exec >> "$OUTPUT/postprocess.log" 2>&1

printf '[%s] waiting for completed confirmatory matrix\n' "$(date -Is)"
while [[ ! -f "$OUTPUT/BENCHMARK_COMPLETED" ]]; do
  sleep 30
done

cd "$PROJECT"
export HF_HUB_OFFLINE=1
mkdir -p artifacts/results

printf '[%s] training seen-only action-free progress critic\n' "$(date -Is)"
"$VENV/bin/vla-progress" train \
  --dataset-root "$DATA/libero_90" \
  --output artifacts/results/progress_critic.pt \
  --epochs 3 \
  --stride 16

printf '[%s] scoring final rollout subset with progress critic\n' "$(date -Is)"
"$VENV/bin/vla-progress" score \
  --checkpoint artifacts/results/progress_critic.pt \
  --eval-root "$OUTPUT/eval" \
  --output artifacts/results/progress_scores.json \
  --max-episodes-per-run 5

printf '[%s] scoring the identical subset with Robometer\n' "$(date -Is)"
"$VENV/bin/vla-bonus" robometer-video \
  --eval-root "$OUTPUT/eval" \
  --model-path "$ROBOMETER" \
  --processor-path "$PROCESSOR" \
  --output artifacts/results/robometer_scores.json \
  --max-episodes-per-run 5

"$VENV/bin/vla-bonus" compare \
  --small-scores artifacts/results/progress_scores.json \
  --robometer-scores artifacts/results/robometer_scores.json \
  --output artifacts/results/reward_ranking.json

"$VENV/bin/vla-bonus" optimize-selection \
  --ranking artifacts/results/reward_ranking.json \
  --per-run artifacts/results/per_run.csv \
  --proxy-scores artifacts/results/progress_scores.json \
  --output artifacts/results/reward_optimization.json

"$VENV/bin/python" report/build_latex_assets.py \
  --results artifacts/results \
  --report-dir report

printf 'ok\n' > "$OUTPUT/POSTPROCESS_COMPLETED"
printf '[%s] POSTPROCESS COMPLETE\n' "$(date -Is)"
