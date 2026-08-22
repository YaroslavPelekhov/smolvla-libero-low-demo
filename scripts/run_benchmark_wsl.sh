#!/usr/bin/env bash
set -euo pipefail

VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
PROJECT=${VLA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG=${VLA_CONFIG:-$PROJECT/configs/benchmark.yaml}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}
JOBS=${VLA_EVAL_JOBS:-2}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

cd "$PROJECT"

run_stage() {
  local stage=$1
  local kind=$2
  local jobs=$3
  printf '[%s] START stage=%s kind=%s jobs=%s\n' "$(date -Is)" "$stage" "$kind" "$jobs"
  "$VENV/bin/vla-run" \
    --config "$CONFIG" \
    --stage "$stage" \
    --kind "$kind" \
    --jobs "$jobs" \
    --execute \
    --skip-completed \
    --data-root "$DATA" \
    --output-root "$OUTPUT" \
    --base-model "$BASE" \
    --vlm-model "$VLM"
  printf '[%s] DONE stage=%s kind=%s\n' "$(date -Is)" "$stage" "$kind"
}

"$VENV/bin/vla-doctor" --config "$CONFIG" --environment
run_stage seen train 1
run_stage zero eval "$JOBS"
run_stage language eval "$JOBS"
run_stage baseline train 1
run_stage baseline eval "$JOBS"
run_stage method train 1
run_stage method eval "$JOBS"

"$VENV/bin/vla-aggregate" --config "$CONFIG" --eval-root "$OUTPUT/eval"
"$VENV/bin/vla-failures" --eval-root "$OUTPUT/eval" --output artifacts/results/failures
printf 'ok\n' > "$OUTPUT/BENCHMARK_COMPLETED"
printf '[%s] BENCHMARK COMPLETE output=%s\n' "$(date -Is)" "$OUTPUT"
