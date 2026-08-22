#!/usr/bin/env bash
set -euo pipefail

VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
PROJECT=${VLA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
VALIDATION=${VLA_V5_OUTPUT:-/home/root/outputs/vla-benchmark-v5-confirmatory}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}
JOBS=${VLA_EVAL_JOBS:-2}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

cd "$PROJECT"
"$VENV/bin/vla-preregister" --config configs/v5.yaml --predictions predictions_v5.yaml
"$VENV/bin/vla-adaptive-merge" --output-root "$SOURCE" --skip-completed

mkdir -p "$VALIDATION/train"
for stage in baseline adaptive_merge; do
  target="$SOURCE/train/$stage"
  link="$VALIDATION/train/$stage"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]]
  elif [[ -e "$link" ]]; then
    printf 'refusing existing non-symlink: %s\n' "$link" >&2
    exit 1
  else
    ln -s "$target" "$link"
  fi
done

for stage in baseline adaptive_merge; do
  "$VENV/bin/vla-run" \
    --config configs/v5.yaml \
    --stage "$stage" \
    --kind eval \
    --jobs "$JOBS" \
    --execute \
    --skip-completed \
    --data-root "$DATA" \
    --output-root "$VALIDATION" \
    --base-model "$BASE" \
    --vlm-model "$VLM"
done

"$VENV/bin/vla-aggregate" --config configs/v5.yaml --eval-root "$VALIDATION/eval"
"$VENV/bin/vla-compare" \
  --per-run artifacts/results_v5/per_run.csv \
  --method adaptive_merge \
  --output artifacts/results_v5/paired_comparison.json

printf '[%s] v5 confirmatory matrix complete: %s\n' "$(date -Is)" "$VALIDATION"
