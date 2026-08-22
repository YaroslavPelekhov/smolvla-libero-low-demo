#!/usr/bin/env bash
set -euo pipefail

VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
PROJECT=${VLA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
OUTPUT=${VLA_CEILING_OUTPUT:-/home/root/outputs/vla-ceiling-v6}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}
JOBS=${VLA_EVAL_JOBS:-2}
CONFIG=${VLA_CEILING_CONFIG:-$PROJECT/configs/ceiling.yaml}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

link_tree() {
  local source=$1
  local destination=$2
  if [[ -e "$destination" || -L "$destination" ]]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"
  ln -s "$source" "$destination"
}

cd "$PROJECT"
"$VENV/bin/vla-manifest" --config "$CONFIG" --data-root "$DATA"
"$VENV/bin/vla-preregister" --config "$CONFIG" --predictions predictions_ceiling.yaml

# Reuse only already-frozen checkpoints. New 1/2-demo checkpoints remain isolated in OUTPUT.
link_tree "$SOURCE_OUTPUT/train/seen" "$OUTPUT/train/seen"
for budget in 5 10 25; do
  link_tree "$SOURCE_OUTPUT/train/baseline/budget-$budget" \
    "$OUTPUT/train/baseline/budget-$budget"
  link_tree "$SOURCE_OUTPUT/train/stabilized/budget-$budget" \
    "$OUTPUT/train/stabilized/budget-$budget"
done

"$VENV/bin/vla-run" --config "$CONFIG" --stage baseline --kind train --jobs 1 \
  --execute --skip-completed --resume-incomplete --data-root "$DATA" --output-root "$OUTPUT" \
  --base-model "$BASE" --vlm-model "$VLM"

"$VENV/bin/vla-soup" --output-root "$OUTPUT" --budgets 1 2 --partner-weight 0.25

"$VENV/bin/vla-run" --config "$CONFIG" --stage zero --kind eval --jobs "$JOBS" \
  --execute --skip-completed --data-root "$DATA" --output-root "$OUTPUT" \
  --base-model "$BASE" --vlm-model "$VLM"
"$VENV/bin/vla-run" --config "$CONFIG" --stage language --kind eval --jobs "$JOBS" \
  --execute --skip-completed --data-root "$DATA" --output-root "$OUTPUT" \
  --base-model "$BASE" --vlm-model "$VLM"
"$VENV/bin/vla-run" --config "$CONFIG" --stage baseline --kind eval --jobs "$JOBS" \
  --execute --skip-completed --data-root "$DATA" --output-root "$OUTPUT" \
  --base-model "$BASE" --vlm-model "$VLM"
"$VENV/bin/vla-run" --config "$CONFIG" --stage stabilized --kind eval --jobs "$JOBS" \
  --execute --skip-completed --data-root "$DATA" --output-root "$OUTPUT" \
  --base-model "$BASE" --vlm-model "$VLM"

"$VENV/bin/vla-aggregate" --config "$CONFIG" --eval-root "$OUTPUT/eval"
"$VENV/bin/vla-compare" \
  --per-run artifacts/results_ceiling/per_run.csv \
  --output artifacts/results_ceiling/paired_comparison.json

printf 'ok\n' > "$OUTPUT/CEILING_COMPLETED"
printf '[%s] ceiling extension complete: %s\n' "$(date -Is)" "$OUTPUT"
