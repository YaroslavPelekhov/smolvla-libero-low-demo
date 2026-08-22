#!/usr/bin/env bash
set -euo pipefail

LEROBOT_REVISION="9e14584904d14eb79dc960b27bb40220f85bb993"
VENV_PATH="${VLA_VENV_PATH:-$HOME/.venvs/vla-budget}"
LEROBOT_PATH="${VLA_LEROBOT_SRC:-$HOME/src/lerobot-vla-budget}"
PROJECT_PATH="${VLA_PROJECT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is for Linux/WSL2." >&2
  exit 2
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake ffmpeg git ninja-build pkg-config \
    libegl1-mesa-dev libgl1-mesa-dev libglvnd-dev libx11-dev
fi

python3.12 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_PATH/bin/python" -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
  --index-url https://download.pytorch.org/whl/cu126

mkdir -p "$(dirname "$LEROBOT_PATH")"
if [[ ! -d "$LEROBOT_PATH/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_PATH"
fi
git -C "$LEROBOT_PATH" fetch --tags --force
git -C "$LEROBOT_PATH" checkout --detach "$LEROBOT_REVISION"

# NVIDIA/LIBERO_LeRobot_v3 stores the gripper as 1=open, 0=closed. The pinned
# LIBERO environment consumes native robosuite commands (-1=open, +1=closed),
# so expose an explicit evaluation mode. The reverse check keeps reruns idempotent.
GRIPPER_PATCH="$PROJECT_PATH/patches/lerobot_libero_binary_gripper.patch"
if git -C "$LEROBOT_PATH" apply --check "$GRIPPER_PATCH"; then
  git -C "$LEROBOT_PATH" apply "$GRIPPER_PATCH"
elif ! git -C "$LEROBOT_PATH" apply --check --reverse "$GRIPPER_PATCH"; then
  echo "LeRobot gripper patch does not match pinned revision $LEROBOT_REVISION" >&2
  exit 2
fi

"$VENV_PATH/bin/python" -m pip install \
  -e "$LEROBOT_PATH[libero,smolvla,peft,robometer]"
"$VENV_PATH/bin/python" -m pip install -e "$PROJECT_PATH[dev,bonus]"

# hf-libero asks an interactive question on first import. Accept package-local defaults.
if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
  printf 'n\n' | "$VENV_PATH/bin/python" -c 'import libero.libero'
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
"$VENV_PATH/bin/vla-doctor" --config "$PROJECT_PATH/configs/experiment.yaml"
