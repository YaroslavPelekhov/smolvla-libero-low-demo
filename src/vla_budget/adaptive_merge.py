from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import tempfile
from pathlib import Path

from .checkpoint_distance import _checkpoint, checkpoint_distance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def high_dispersion_tasks(distances: dict[int, float]) -> tuple[set[int], float]:
    if not distances:
        raise ValueError("at least one task distance is required")
    threshold = float(statistics.median(distances.values()))
    return {task for task, value in distances.items() if value >= threshold}, threshold


def _copy_selected_checkpoint(source: Path, output: Path, manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        temporary = Path(temp_name) / "checkpoint"
        shutil.copytree(source, temporary)
        (temporary / "adaptive_merge_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            existing = output / "adaptive_merge_manifest.json"
            if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")) == manifest:
                return
            raise FileExistsError(f"refusing to replace existing checkpoint: {output}")
        temporary.replace(output)


def build_matrix(
    output_root: Path,
    budgets: list[int],
    tasks: list[int],
    seeds: list[int],
    low_budget_max: int = 10,
    skip_completed: bool = False,
) -> None:
    if len(seeds) != 2:
        raise ValueError("adaptive merging requires exactly two seed branches")
    dispersion: dict[int, dict[int, float]] = {}
    thresholds: dict[int, float] = {}
    selected: dict[int, set[int]] = {}
    for budget in budgets:
        if budget <= low_budget_max:
            continue
        distances = {}
        for task in tasks:
            checkpoints = [
                _checkpoint(
                    output_root
                    / "train"
                    / "baseline"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{seed}"
                )
                for seed in seeds
            ]
            distances[task] = float(checkpoint_distance(*checkpoints)["relative_l2"])
        selected[budget], thresholds[budget] = high_dispersion_tasks(distances)
        dispersion[budget] = distances

    for budget in budgets:
        for task in tasks:
            if budget <= low_budget_max:
                source_stage = "task_vector"
                rule = f"budget <= {low_budget_max}: share seed-specific target task vectors"
            elif task in selected[budget]:
                source_stage = "stabilized"
                rule = "relative_l2 >= within-budget median: use full-checkpoint CSWI"
            else:
                source_stage = "baseline"
                rule = "relative_l2 < within-budget median: retain baseline"
            for seed in seeds:
                source = _checkpoint(
                    output_root
                    / "train"
                    / source_stage
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{seed}"
                )
                destination = (
                    output_root
                    / "train"
                    / "adaptive_merge"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{seed}"
                    / "checkpoints"
                    / "last"
                    / "pretrained_model"
                )
                completed = destination.parents[2] / "COMPLETED"
                if skip_completed and completed.is_file() and (
                    destination / "adaptive_merge_manifest.json"
                ).is_file():
                    print(f"skip completed: {destination}")
                    continue
                source_model = source / "model.safetensors"
                manifest = {
                    "schema_version": 1,
                    "method": "budget_and_dispersion_adaptive_merging",
                    "budget": budget,
                    "task_id": task,
                    "seed": seed,
                    "selection_rule": rule,
                    "source_stage": source_stage,
                    "source_checkpoint": str(source),
                    "source_sha256": _sha256(source_model),
                    "low_budget_max": low_budget_max,
                    "relative_l2": dispersion.get(budget, {}).get(task),
                    "within_budget_median": thresholds.get(budget),
                }
                _copy_selected_checkpoint(source, destination, manifest)
                completed.write_text("ok\n", encoding="utf-8")
                print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build budget- and dispersion-adaptive policies")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs=2, default=[17, 29])
    parser.add_argument("--low-budget-max", type=int, default=10)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    build_matrix(
        args.output_root,
        args.budgets,
        args.tasks,
        args.seeds,
        args.low_budget_max,
        args.skip_completed,
    )


if __name__ == "__main__":
    main()
