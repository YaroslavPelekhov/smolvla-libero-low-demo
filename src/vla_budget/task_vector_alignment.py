from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .checkpoint_distance import _checkpoint


def task_vector_alignment(
    seen_left: Path,
    target_left: Path,
    seen_right: Path,
    target_right: Path,
) -> dict:
    from safetensors import safe_open

    paths = {
        "seen_left": seen_left / "model.safetensors",
        "target_left": target_left / "model.safetensors",
        "seen_right": seen_right / "model.safetensors",
        "target_right": target_right / "model.safetensors",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    handles = {
        label: safe_open(path, framework="pt", device="cpu") for label, path in paths.items()
    }
    keys = set(handles["target_left"].keys())
    if any(set(handle.keys()) != keys for handle in handles.values()):
        raise ValueError("task-vector source tensor keys differ")

    dot = norm_left = norm_right = 0.0
    floating = elements = 0
    for key in sorted(keys):
        tensors = [handle.get_tensor(key) for handle in handles.values()]
        if len({tensor.shape for tensor in tensors}) != 1:
            raise ValueError(f"incompatible tensor shape across sources: {key}")
        if not tensors[1].is_floating_point():
            continue
        left = tensors[1].float() - tensors[0].float()
        right = tensors[3].float() - tensors[2].float()
        dot += float((left * right).sum())
        norm_left += float((left * left).sum())
        norm_right += float((right * right).sum())
        floating += 1
        elements += left.numel()
    denominator = math.sqrt(norm_left * norm_right)
    if denominator == 0.0:
        raise ValueError("zero-norm task vector")
    return {
        "cosine": dot / denominator,
        "l2_left": math.sqrt(norm_left),
        "l2_right": math.sqrt(norm_right),
        "floating_tensors": floating,
        "elements": elements,
    }


def matrix(output_root: Path, budgets: list[int], tasks: list[int], seeds: list[int]) -> list[dict]:
    if len(seeds) != 2:
        raise ValueError("alignment requires exactly two seed branches")
    records = []
    seen = {
        seed: _checkpoint(output_root / "train" / "seen" / f"seed-{seed}") for seed in seeds
    }
    for budget in budgets:
        for task in tasks:
            targets = {
                seed: _checkpoint(
                    output_root
                    / "train"
                    / "baseline"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{seed}"
                )
                for seed in seeds
            }
            records.append(
                {
                    "budget": budget,
                    "task_id": task,
                    "seeds": seeds,
                    **task_vector_alignment(
                        seen[seeds[0]], targets[seeds[0]], seen[seeds[1]], targets[seeds[1]]
                    ),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cross-seed target task-vector alignment")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs=2, default=[17, 29])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "metric": "cosine_between_cross_seed_target_task_vectors",
        "records": matrix(args.output_root, args.budgets, args.tasks, args.seeds),
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
