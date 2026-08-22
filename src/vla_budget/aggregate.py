from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config, resolve


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def collect(eval_root: Path) -> list[dict]:
    rows: list[dict] = []
    for info_path in sorted(eval_root.rglob("eval_info.json")):
        meta_path = info_path.parent / "run_meta.json"
        completed_path = info_path.parent / "COMPLETED"
        if not meta_path.exists() or not completed_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        per_task = info.get("per_task", [])
        if len(per_task) != 1:
            raise ValueError(f"Expected one target task in {info_path}, got {len(per_task)}")
        successes = [bool(value) for value in per_task[0]["metrics"]["successes"]]
        row = dict(meta)
        row.update(
            {
                "n": len(successes),
                "successes": sum(successes),
                "success_rate": float(np.mean(successes)),
                "success_vector": json.dumps(successes, separators=(",", ":")),
                "eval_info": str(info_path),
            }
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        method = "zero" if row["stage"] == "zero" else row["method"]
        if method == "wrong":
            continue
        grouped[(method, int(row["budget"]))].append(row)
    summary = []
    for (method, budget), group in sorted(grouped.items()):
        total_n = sum(row["n"] for row in group)
        total_s = sum(row["successes"] for row in group)
        lo, hi = _wilson(total_s, total_n)
        summary.append(
            {
                "method": method,
                "budget": budget,
                "success_rate": total_s / total_n,
                "ci_low": lo,
                "ci_high": hi,
                "n": total_n,
                "tasks": len({row["task_id"] for row in group}),
                "seeds": len({row["seed"] for row in group}),
            }
        )
    return summary


def summarize_language(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["stage"], int(row["task_id"]), int(row["seed"])): row
        for row in rows
        if row["stage"] in {"zero", "language"}
    }
    output = []
    for (_, task_id, seed), correct in sorted(indexed.items()):
        if correct["stage"] != "zero":
            continue
        wrong = indexed.get(("language", task_id, seed))
        if wrong is None:
            continue
        output.append(
            {
                "task_id": task_id,
                "seed": seed,
                "n_correct": correct["n"],
                "n_wrong": wrong["n"],
                "correct_rate": correct["success_rate"],
                "wrong_rate": wrong["success_rate"],
                "drop": correct["success_rate"] - wrong["success_rate"],
            }
        )
    return output


def validate_coverage(rows: list[dict], cfg: dict) -> None:
    task_ids = [int(value) for value in cfg["dataset"]["target_task_ids"]]
    seeds = [int(value) for value in cfg["protocol"]["train_seeds"]]
    budgets = [int(value) for value in cfg["protocol"]["budgets"]]
    observed_keys = [
        (str(row["stage"]), int(row["budget"]), int(row["task_id"]), int(row["seed"]))
        for row in rows
    ]
    observed = set(observed_keys)
    expected = set()
    method_stage = str(cfg.get("method", {}).get("stage", "method"))
    include_controls = bool(cfg["protocol"].get("include_controls", True))
    for task_id in task_ids:
        for seed in seeds:
            if include_controls:
                expected.add(("zero", 0, task_id, seed))
                expected.add(("language", 0, task_id, seed))
            for stage in ("baseline", method_stage):
                for budget in budgets:
                    expected.add((stage, budget, task_id, seed))
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(key for key, count in Counter(observed_keys).items() if count > 1)
    if missing:
        preview = "\n".join(map(str, missing[:12]))
        raise ValueError(f"Experiment matrix incomplete ({len(missing)} missing):\n{preview}")
    if unexpected:
        preview = "\n".join(map(str, unexpected[:12]))
        raise ValueError(f"Experiment matrix contains {len(unexpected)} unexpected cells:\n{preview}")
    if duplicates:
        preview = "\n".join(map(str, duplicates[:12]))
        raise ValueError(f"Experiment matrix contains {len(duplicates)} duplicate cells:\n{preview}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    fieldnames.extend(key for row in rows for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, summary: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 3.7))
    method_name = next(
        (row["method"] for row in summary if row["method"] not in {"zero", "baseline"}),
        "method",
    )
    for method, color in [("baseline", "#7c8799"), (method_name, "#1267d5")]:
        points = [row for row in summary if row["method"] in {method, "zero"}]
        points = sorted({row["budget"]: row for row in points}.values(), key=lambda row: row["budget"])
        if not points:
            continue
        x = [row["budget"] for row in points]
        y = [100 * row["success_rate"] for row in points]
        lower = [100 * (row["success_rate"] - row["ci_low"]) for row in points]
        upper = [100 * (row["ci_high"] - row["success_rate"]) for row in points]
        axis.errorbar(x, y, yerr=[lower, upper], marker="o", linewidth=2, color=color, label=method)
    ticks = sorted({int(row["budget"]) for row in summary})
    axis.set(xlabel="Target demonstrations", ylabel="Success (%)", xticks=ticks, ylim=(0, 100))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate measured success without fabricating gaps")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--eval-root", help="Override the evaluation root directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    eval_root = Path(args.eval_root) if args.eval_root else resolve(cfg["paths"]["outputs"]) / "eval"
    rows = collect(eval_root)
    if not rows:
        raise SystemExit("No eval_info.json files found; refusing to create an empty/fake curve")
    method_stage = str(cfg.get("method", {}).get("stage", "method"))
    include_controls = bool(cfg["protocol"].get("include_controls", True))
    included_stages = {"baseline", method_stage}
    if include_controls:
        included_stages.update({"zero", "language"})
    rows = [row for row in rows if row["stage"] in included_stages]
    required_n = int(cfg["protocol"]["eval_episodes"])
    bad = [row for row in rows if row["n"] != required_n]
    if bad:
        raise ValueError(f"Runs not equal to the registered {required_n} episodes: {bad}")
    validate_coverage(rows, cfg)
    summary = summarize(rows)
    language = summarize_language(rows)
    result_dir = resolve(cfg["paths"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "per_run.csv", rows)
    _write_csv(result_dir / "cost_curve.csv", summary)
    _write_csv(result_dir / "language_control.csv", language)
    validation = {
        "schema_version": 1,
        "status": "pass",
        "evaluation_units": len(rows),
        "unique_cells": len(
            {
                (row["stage"], row["budget"], row["task_id"], row["seed"])
                for row in rows
            }
        ),
        "episodes_per_unit": required_n,
        "total_outcomes": sum(int(row["n"]) for row in rows),
        "stages": {
            stage: sum(row["stage"] == stage for row in rows)
            for stage in sorted(included_stages)
        },
    }
    (result_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    _plot(result_dir / "cost_curve.png", summary)
    print(result_dir)


if __name__ == "__main__":
    main()
