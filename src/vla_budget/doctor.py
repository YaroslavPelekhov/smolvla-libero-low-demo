from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import load_config, resolve
from .protocol import manifest_digest


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: str, *args: str) -> dict[str, Any]:
    path = shutil.which(command)
    if path is None:
        return {"ok": False, "path": None, "version": None}
    completed = subprocess.run(
        [path, *args], check=False, capture_output=True, text=True, timeout=20
    )
    first_line = (completed.stdout or completed.stderr).splitlines()
    return {
        "ok": completed.returncode == 0,
        "path": path,
        "version": first_line[0] if first_line else None,
    }


def _git_revision() -> dict[str, Any]:
    candidates: list[Path] = []
    if override := os.environ.get("VLA_LEROBOT_SRC"):
        candidates.append(Path(override))
    try:
        direct_url = importlib.metadata.distribution("lerobot").read_text("direct_url.json")
        if direct_url:
            parsed = urlparse(json.loads(direct_url)["url"])
            if parsed.scheme == "file":
                candidates.append(Path(unquote(parsed.path)))
    except (importlib.metadata.PackageNotFoundError, KeyError, TypeError, ValueError):
        pass
    candidates.extend(
        [Path.home() / "src" / "lerobot-vla-budget", Path("/home/root/src/lerobot-vla-budget")]
    )
    source = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    probe = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return {"ok": False, "source": str(source), "revision": None, "dirty": None}
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    )
    return {"ok": True, "source": str(source), "revision": revision, "dirty": dirty}


def _gpu_check() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        return {"ok": False, "error": str(error)}
    result: dict[str, Any] = {
        "ok": False,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return result
    generator = torch.Generator(device="cuda").manual_seed(20260811)
    left = torch.randn((2048, 2048), generator=generator, device="cuda")
    right = left @ left
    torch.cuda.synchronize()
    result.update(
        {
            "ok": bool(torch.isfinite(right).all().item()),
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "checksum": round(float(right[0, 0]), 6),
        }
    )
    del left, right
    torch.cuda.empty_cache()
    return result


def _protocol_check(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve(cfg["paths"]["manifest"])
    prereg_path = resolve(cfg["paths"]["preregistration"])
    if not manifest_path.exists() or not prereg_path.exists():
        return {"ok": False, "error": "manifest or preregistration is missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preregistration = json.loads(prereg_path.read_text(encoding="utf-8"))
    predictions_file = preregistration.get("predictions_file")
    if not predictions_file:
        return {"ok": False, "error": "preregistration has no predictions_file"}
    predictions_path = resolve(predictions_file)
    if not predictions_path.exists():
        return {
            "ok": False,
            "error": f"registered predictions file is missing: {predictions_path}",
        }
    prediction_sha = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    digest_ok = manifest.get("sha256") == manifest_digest(manifest)
    prereg_ok = preregistration.get("sha256") == prediction_sha
    return {
        "ok": digest_ok and prereg_ok,
        "manifest_sha256": manifest.get("sha256"),
        "manifest_digest_ok": digest_ok,
        "preregistration_sha256": preregistration.get("sha256"),
        "preregistration_digest_ok": prereg_ok,
        "predictions_file": str(predictions_path),
        "seen_tasks": manifest.get("seen", {}).get("total_tasks"),
        "seen_episodes": manifest.get("seen", {}).get("total_episodes"),
        "target_task_ids": sorted(int(key) for key in manifest["target"]["tasks"]),
    }


def _environment_check(cfg: dict[str, Any]) -> dict[str, Any]:
    if platform.system() != "Linux":
        return {"ok": False, "error": "LIBERO environment validation requires Linux"}
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    from lerobot.envs.configs import LiberoEnv

    task_id = int(cfg["dataset"]["target_task_ids"][0])
    env_cfg = LiberoEnv(
        task=cfg["dataset"]["target_suite"],
        task_ids=[task_id],
        observation_height=256,
        observation_width=256,
        camera_name_mapping={
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "wrist_image",
        },
        gripper_action_mode="binary_01",
    )
    envs = env_cfg.create_envs(1, use_async_envs=False)
    env = envs[cfg["dataset"]["target_suite"]][task_id]
    try:
        observation, _ = env.reset(seed=int(cfg["protocol"]["eval_seed"]))
        prompt = env.unwrapped.envs[0].task_description
        manifest = json.loads(resolve(cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
        manifest_prompt = manifest["target"]["tasks"][str(task_id)]["prompt"]
        dataset_noop = np.zeros((1, 7), dtype=np.float32)
        dataset_noop[:, -1] = 1.0  # dataset convention: open; bridge maps this to native -1
        _, reward, terminated, truncated, info = env.step(dataset_noop)
        pixels = observation["pixels"]
        return {
            "ok": prompt == manifest_prompt,
            "suite": cfg["dataset"]["target_suite"],
            "task_id": task_id,
            "prompt": prompt,
            "manifest_prompt": manifest_prompt,
            "prompt_matches_manifest": prompt == manifest_prompt,
            "camera_keys": sorted(pixels),
            "camera_shapes": {key: list(value.shape) for key, value in pixels.items()},
            "state_components": {
                "eef_pos": list(observation["robot_state"]["eef"]["pos"].shape),
                "eef_quat": list(observation["robot_state"]["eef"]["quat"].shape),
                "gripper_qpos": list(
                    observation["robot_state"]["gripper"]["qpos"].shape
                ),
                "flattened_policy_state": [1, 8],
            },
            "gripper_action_mode": "binary_01 (1=open, 0=closed -> -1=open, +1=closed)",
            "step": {
                "reward": reward.tolist(),
                "terminated": terminated.tolist(),
                "truncated": truncated.tolist(),
                "is_success": info["is_success"].tolist(),
            },
        }
    finally:
        env.close()


def run_checks(cfg: dict[str, Any], *, environment: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "platform": {
            "ok": platform.system() == "Linux",
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "gpu": _gpu_check(),
        "packages": {
            name: _version(name)
            for name in ["lerobot", "hf-libero", "mujoco", "peft", "torchcodec"]
        },
        "ffmpeg": _command_version("ffmpeg", "-version"),
        "cmake": _command_version("cmake", "--version"),
        "lerobot_source": _git_revision(),
        "protocol": _protocol_check(cfg),
    }
    checks["packages"]["ok"] = all(
        checks["packages"][name] is not None
        for name in ["lerobot", "hf-libero", "mujoco", "peft"]
    )
    if environment:
        try:
            checks["environment"] = _environment_check(cfg)
        # A doctor command must serialize third-party MuJoCo/OpenGL failures instead of
        # losing them in a traceback; these libraries do not share a stable exception base.
        except Exception as error:  # noqa: BLE001
            checks["environment"] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    required = ["platform", "gpu", "packages", "ffmpeg", "cmake", "lerobot_source", "protocol"]
    if environment:
        required.append("environment")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if all(checks[name]["ok"] for name in required) else "fail",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CUDA/LeRobot/LIBERO execution stack")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--environment", action="store_true", help="Reset and step a real LIBERO env")
    parser.add_argument("--output", default="artifacts/system_validation.json")
    args = parser.parse_args()
    report = run_checks(load_config(args.config), environment=args.environment)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
