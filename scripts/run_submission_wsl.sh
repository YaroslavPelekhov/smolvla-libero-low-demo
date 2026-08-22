#!/usr/bin/env bash
set -euo pipefail

VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
PROJECT=${VLA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}
JOBS=${VLA_EVAL_JOBS:-2}

export VLA_VENV="$VENV"
export VLA_PROJECT="$PROJECT"
export VLA_OUTPUT="$OUTPUT"
export VLA_DATA="$DATA"
export VLA_BASE="$BASE"
export VLA_VLM="$VLM"
export VLA_EVAL_JOBS="$JOBS"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

cd "$PROJECT"

printf '[%s] phase 1/2: baseline and registered SAN-FT ablation\n' "$(date -Is)"
bash scripts/run_benchmark_wsl.sh

printf '[%s] phase 2/2: registered CSWI follow-up\n' "$(date -Is)"
"$VENV/bin/vla-preregister" --config configs/final.yaml --predictions predictions_v3.yaml
"$VENV/bin/vla-soup" --output-root "$OUTPUT" --partner-weight 0.25
"$VENV/bin/vla-run" \
  --config configs/final.yaml \
  --stage stabilized \
  --kind eval \
  --jobs "$JOBS" \
  --execute \
  --skip-completed \
  --data-root "$DATA" \
  --output-root "$OUTPUT" \
  --base-model "$BASE" \
  --vlm-model "$VLM"
"$VENV/bin/vla-aggregate" --config configs/final.yaml --eval-root "$OUTPUT/eval"
"$VENV/bin/vla-compare" \
  --per-run artifacts/results_v3/per_run.csv \
  --output artifacts/results_v3/paired_comparison.json
"$VENV/bin/vla-failures" \
  --eval-root "$OUTPUT/eval" \
  --stage stabilized \
  --output artifacts/results_v3/failures
"$VENV/bin/vla-verify" --without-pdf

printf '[%s] submission evidence complete: %s/artifacts/results_v3\n' "$(date -Is)" "$PROJECT"
