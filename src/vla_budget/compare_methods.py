from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["budget"] = int(row["budget"])
        row["task_id"] = int(row["task_id"])
        row["seed"] = int(row["seed"])
        row["success_vector"] = json.loads(row["success_vector"])
    return rows


def compare(rows: list[dict], method: str = "stabilized") -> dict:
    indexed = {
        (row["method"], row["budget"], row["task_id"], row["seed"]): row
        for row in rows
        if row["method"] in {"baseline", method}
    }
    budgets = sorted({key[1] for key in indexed})
    tasks = sorted({key[2] for key in indexed})
    seeds = sorted({key[3] for key in indexed})
    aggregate = []
    task_rows = []
    for budget in budgets:
        wins = losses = baseline_success = method_success = total = 0
        per_task = defaultdict(lambda: [0, 0, 0])
        for task in tasks:
            for seed in seeds:
                baseline = indexed[("baseline", budget, task, seed)]["success_vector"]
                candidate = indexed[(method, budget, task, seed)]["success_vector"]
                if len(baseline) != len(candidate):
                    raise ValueError("Paired vectors have different lengths")
                for before, after in zip(baseline, candidate, strict=True):
                    baseline_success += int(before)
                    method_success += int(after)
                    total += 1
                    per_task[task][0] += int(before)
                    per_task[task][1] += int(after)
                    per_task[task][2] += 1
                    wins += int(after and not before)
                    losses += int(before and not after)
        discordant = wins + losses
        p_value = 1.0 if discordant == 0 else binomtest(wins, discordant, 0.5).pvalue
        aggregate.append(
            {
                "budget": budget,
                "n": total,
                "baseline_successes": baseline_success,
                "method_successes": method_success,
                "delta_percentage_points": 100 * (method_success - baseline_success) / total,
                "wins": wins,
                "losses": losses,
                "mcnemar_exact_p": p_value,
            }
        )
        for task, (before, after, n) in sorted(per_task.items()):
            task_rows.append(
                {
                    "budget": budget,
                    "task_id": task,
                    "n": n,
                    "baseline_successes": before,
                    "method_successes": after,
                    "baseline_rate": before / n,
                    "method_rate": after / n,
                }
            )
    return {"schema_version": 1, "method": method, "aggregate": aggregate, "per_task": task_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of baseline and follow-up method")
    parser.add_argument("--per-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="stabilized")
    args = parser.parse_args()
    result = compare(_read(args.per_run), args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
