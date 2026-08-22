#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "17" && "$1" != "29" ) ]]; then
  echo "usage: $0 {17|29}" >&2
  exit 2
fi

SEED=$1
VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
PROJECT=${VLA_PROJECT:-/mnt/c/Users/psgpe/Downloads/world}
CONFIG=${VLA_CONFIG:-$PROJECT/configs/benchmark.yaml}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

exec >> "$OUTPUT/resume-seed-$SEED.log" 2>&1
cd "$PROJECT"
printf '[%s] resuming seen seed=%s from latest complete checkpoint\n' "$(date -Is)" "$SEED"
"$VENV/bin/vla-run" \
  --config "$CONFIG" \
  --stage seen \
  --kind train \
  --train-seed "$SEED" \
  --execute \
  --skip-completed \
  --resume-incomplete \
  --data-root "$DATA" \
  --output-root "$OUTPUT" \
  --base-model "$BASE" \
  --vlm-model "$VLM"
printf '[%s] resumed seen seed=%s complete\n' "$(date -Is)" "$SEED"
