from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _checkpoint(train_dir: Path) -> Path:
    preferred = train_dir / "checkpoints" / "last" / "pretrained_model"
    if preferred.is_dir():
        return preferred
    candidates = sorted((train_dir / "checkpoints").glob("*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"checkpoint not found under {train_dir}")
    return candidates[-1]


def checkpoint_distance(left: Path, right: Path) -> dict:
    from safetensors import safe_open

    left_files = [left / "model.safetensors"]
    right_files = [right / "model.safetensors"]
    if not left_files or not right_files:
        raise FileNotFoundError("both checkpoints must contain safetensors files")

    def load(files: list[Path]) -> dict:
        tensors = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as stream:
                for key in stream.keys():  # noqa: SIM118 - safe_open is not a dict
                    if key in tensors:
                        raise ValueError(f"duplicate tensor key: {key}")
                    tensors[key] = stream.get_tensor(key)
        return tensors

    left_tensors = load(left_files)
    right_tensors = load(right_files)
    if left_tensors.keys() != right_tensors.keys():
        raise ValueError("checkpoint tensor keys differ")

    squared_delta = squared_norm = 0.0
    elements = floating_tensors = 0
    for key, left_tensor in left_tensors.items():
        right_tensor = right_tensors[key]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise ValueError(f"incompatible tensor: {key}")
        if not left_tensor.is_floating_point():
            continue
        left_float = left_tensor.float()
        right_float = right_tensor.float()
        delta = left_float - right_float
        squared_delta += float((delta * delta).sum())
        squared_norm += 0.5 * float(
            (left_float * left_float).sum() + (right_float * right_float).sum()
        )
        elements += left_tensor.numel()
        floating_tensors += 1
    return {
        "rms_delta": math.sqrt(squared_delta / elements),
        "relative_l2": math.sqrt(squared_delta / squared_norm),
        "floating_tensors": floating_tensors,
        "elements": elements,
    }


def matrix(root: Path, budgets: list[int], tasks: list[int], seeds: tuple[int, int]) -> list[dict]:
    records = []
    for budget in budgets:
        for task in tasks:
            paths = [
                _checkpoint(root / f"budget-{budget}" / f"task-{task}" / f"seed-{seed}")
                for seed in seeds
            ]
            records.append(
                {
                    "budget": budget,
                    "task_id": task,
                    "seeds": list(seeds),
                    "checkpoints": [str(path) for path in paths],
                    **checkpoint_distance(*paths),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure reward-free distance between seed branches")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs=2, default=[17, 29])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "metric": "relative_l2_between_full_floating_point_checkpoints",
        "records": matrix(args.train_root, args.budgets, args.tasks, tuple(args.seeds)),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
