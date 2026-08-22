#!/usr/bin/env bash
set -euo pipefail

VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
PROJECT=${VLA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
VALIDATION=${VLA_V4_OUTPUT:-/home/root/outputs/vla-benchmark-v4-confirmatory}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}
JOBS=${VLA_EVAL_JOBS:-2}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

cd "$PROJECT"

"$VENV/bin/vla-preregister" --config configs/v4.yaml --predictions predictions_v4.yaml
"$VENV/bin/vla-task-vector" \
  --output-root "$SOURCE" \
  --partner-weight 0.25 \
  --skip-completed

mkdir -p "$VALIDATION/train"
for stage in seen baseline task_vector; do
  target="$SOURCE/train/$stage"
  link="$VALIDATION/train/$stage"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || {
      printf 'unexpected symlink target: %s\n' "$link" >&2
      exit 1
    }
  elif [[ -e "$link" ]]; then
    printf 'refusing to replace existing non-symlink: %s\n' "$link" >&2
    exit 1
  else
    ln -s "$target" "$link"
  fi
done

run_eval() {
  local stage=$1
  printf '[%s] v4 eval stage=%s seed=2000\n' "$(date -Is)" "$stage"
  "$VENV/bin/vla-run" \
    --config configs/v4.yaml \
    --stage "$stage" \
    --kind eval \
    --jobs "$JOBS" \
    --execute \
    --skip-completed \
    --data-root "$DATA" \
    --output-root "$VALIDATION" \
    --base-model "$BASE" \
    --vlm-model "$VLM"
}

run_eval baseline
run_eval task_vector

"$VENV/bin/vla-aggregate" --config configs/v4.yaml --eval-root "$VALIDATION/eval"
"$VENV/bin/vla-compare" \
  --per-run artifacts/results_v4/per_run.csv \
  --method task_vector \
  --output artifacts/results_v4/paired_comparison.json
"$VENV/bin/vla-failures" \
  --eval-root "$VALIDATION/eval" \
  --stage task_vector \
  --output artifacts/results_v4/failures

printf '[%s] v4 confirmatory matrix complete: %s\n' "$(date -Is)" "$VALIDATION"
