#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 POLICY TASK_ID N_ACTION_STEPS OUTPUT_DIR [EPISODES]" >&2
  exit 2
fi

POLICY=$1
TASK_ID=$2
N_ACTION_STEPS=$3
OUTPUT_DIR=$4
EPISODES=${5:-8}
VENV=${VLA_VENV:-/home/root/.venvs/vla-budget}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite existing diagnostic output: $OUTPUT_DIR" >&2
  exit 2
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

"$VENV/bin/lerobot-eval" \
  --policy.path="$POLICY" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[$TASK_ID]" \
  '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
  --env.control_mode=relative \
  --env.gripper_action_mode=binary_01 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=4 \
  --eval.n_episodes="$EPISODES" \
  --eval.batch_size=4 \
  --eval.use_async_envs=false \
  --eval.recording=false \
  --seed=1000 \
  --output_dir="$OUTPUT_DIR"
