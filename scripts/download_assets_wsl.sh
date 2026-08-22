#!/usr/bin/env bash
set -euo pipefail

VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}
DATA=${VLA_DATA:-/home/root/data/LIBERO_LeRobot_v3}
BASE=${VLA_BASE:-/home/root/models/smolvla_base}
VLM=${VLA_VLM:-/home/root/models/SmolVLM2-500M-Video-Instruct}
ROBOMETER=${VLA_ROBOMETER:-/home/root/models/Robometer-4B}
PROCESSOR=${VLA_ROBOMETER_PROCESSOR:-/home/root/models/Qwen3-VL-4B-Instruct}

"$VENV/bin/hf" download nvidia/LIBERO_LeRobot_v3 \
  --repo-type dataset --local-dir "$DATA" \
  --include 'libero_90/**' 'libero_goal/**'
"$VENV/bin/hf" download lerobot/smolvla_base --local-dir "$BASE"
"$VENV/bin/hf" download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir "$VLM"
"$VENV/bin/hf" download robometer/Robometer-4B --local-dir "$ROBOMETER"
"$VENV/bin/hf" download Qwen/Qwen3-VL-4B-Instruct --local-dir "$PROCESSOR"

printf '[%s] all dataset and model assets are present\n' "$(date -Is)"
