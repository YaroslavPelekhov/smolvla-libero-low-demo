from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .bonus import _heldout_success, optimize_selection_payload
from .compare_methods import compare
from .protocol import manifest_digest


class SubmissionError(ValueError):
    """Raised when the frozen evidence package is internally inconsistent."""


def _json(path: Path) -> dict | list:
    if not path.is_file():
        raise SubmissionError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_preregistration_record(root: Path, record_name: str, label: str) -> None:
    record = _json(root / "artifacts" / record_name)
    prediction = root / str(record["predictions_file"])
    if not prediction.is_file():
        raise SubmissionError(f"registered prediction file is missing: {prediction}")
    actual = _sha256(prediction)
    if actual != record["sha256"]:
        raise SubmissionError(
            f"{label} prediction hash mismatch: expected {record['sha256']}, got {actual}"
        )


def _check_preregistration(root: Path, version: int) -> None:
    _check_preregistration_record(root, f"preregistration_v{version}.json", f"v{version}")


def _read_final_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SubmissionError(f"missing required artifact: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["budget"] = int(row["budget"])
        row["task_id"] = int(row["task_id"])
        row["seed"] = int(row["seed"])
        row["n"] = int(row["n"])
        row["successes"] = int(row["successes"])
        row["success_vector"] = json.loads(row["success_vector"])
    return rows


def _check_matrix(
    root: Path,
    result_name: str,
    method: str,
    expected_validation: dict,
) -> tuple[int, int]:
    result_dir = root / "artifacts" / result_name
    validation = _json(result_dir / "validation.json")
    for key, value in expected_validation.items():
        if validation.get(key) != value:
            raise SubmissionError(
                f"final validation mismatch for {key}: expected {value!r}, got {validation.get(key)!r}"
            )

    rows = _read_final_rows(result_dir / "per_run.csv")
    expected_units = int(expected_validation["evaluation_units"])
    if len(rows) != expected_units:
        raise SubmissionError(f"expected {expected_units} rows in {result_name}, got {len(rows)}")
    keys = {(row["stage"], row["budget"], row["task_id"], row["seed"]) for row in rows}
    if len(keys) != len(rows):
        raise SubmissionError("final per_run.csv contains duplicate experiment cells")
    for row in rows:
        vector = row["success_vector"]
        if row["n"] != 20 or len(vector) != 20:
            raise SubmissionError(f"cell has a non-registered denominator: {row}")
        if sum(bool(value) for value in vector) != row["successes"]:
            raise SubmissionError(f"success count disagrees with Boolean vector: {row}")

    stored = _json(result_dir / "paired_comparison.json")
    recomputed = compare(rows, method)
    if stored != recomputed:
        raise SubmissionError(f"{result_name}/paired_comparison.json does not match per_run.csv")
    return len(rows), sum(row["n"] for row in rows)


def _check_failures(root: Path) -> int:
    failure_dir = root / "artifacts" / "results_v3" / "failures"
    catalog = _json(failure_dir / "catalog.json")
    if not isinstance(catalog, list) or len(catalog) != 3:
        raise SubmissionError("failure catalog must contain exactly three real rollouts")
    cells = set()
    for index, record in enumerate(catalog, start=1):
        figure = root / str(record["figure"])
        if not figure.is_file() or figure.stat().st_size < 10_000:
            raise SubmissionError(f"missing or empty failure figure: {figure}")
        cells.add((record["task_id"], record["budget"], record["seed"], record["episode_index"]))
        if figure.name != f"failure-{index}.png":
            raise SubmissionError("failure catalog order disagrees with numbered figures")
    if len(cells) != 3:
        raise SubmissionError("failure examples are not three distinct evaluation cells")
    return len(catalog)


def verify(root: Path, require_pdf: bool = True) -> dict:
    root = root.resolve()
    manifest = _json(root / "artifacts" / "protocol_manifest.json")
    if manifest_digest(manifest) != manifest.get("sha256"):
        raise SubmissionError("protocol_manifest.json SHA-256 is invalid")
    for version in (2, 3, 4, 5):
        _check_preregistration(root, version)
    _check_preregistration_record(
        root, "preregistration_ceiling.json", "ceiling extension"
    )
    ceiling_manifest = _json(root / "artifacts" / "protocol_manifest_ceiling.json")
    if manifest_digest(ceiling_manifest) != ceiling_manifest.get("sha256"):
        raise SubmissionError("protocol_manifest_ceiling.json SHA-256 is invalid")
    matrices = [
        (
            "results_v3",
            "stabilized",
            {
                "status": "pass",
                "evaluation_units": 48,
                "unique_cells": 48,
                "episodes_per_unit": 20,
                "total_outcomes": 960,
                "stages": {"zero": 6, "language": 6, "baseline": 18, "stabilized": 18},
            },
        ),
        (
            "results_v4",
            "task_vector",
            {
                "status": "pass",
                "evaluation_units": 36,
                "unique_cells": 36,
                "episodes_per_unit": 20,
                "total_outcomes": 720,
                "stages": {"baseline": 18, "task_vector": 18},
            },
        ),
        (
            "results_v5",
            "adaptive_merge",
            {
                "status": "pass",
                "evaluation_units": 36,
                "unique_cells": 36,
                "episodes_per_unit": 20,
                "total_outcomes": 720,
                "stages": {"adaptive_merge": 18, "baseline": 18},
            },
        ),
        (
            "results_ceiling",
            "stabilized",
            {
                "status": "pass",
                "evaluation_units": 72,
                "unique_cells": 72,
                "episodes_per_unit": 20,
                "total_outcomes": 1440,
                "stages": {"zero": 6, "language": 6, "baseline": 30, "stabilized": 30},
            },
        ),
    ]
    checked = [_check_matrix(root, *matrix) for matrix in matrices]
    units = sum(item[0] for item in checked)
    outcomes = sum(item[1] for item in checked)
    failures = _check_failures(root)

    bonus = _json(root / "artifacts" / "results" / "reward_ranking.json")
    if bonus.get("n_checkpoints") != 42 or len(bonus.get("records", [])) != 42:
        raise SubmissionError("Bonus B evidence must contain exactly 42 ranked checkpoint cells")
    optimization = _json(root / "artifacts" / "results" / "reward_optimization.json")
    if optimization.get("n_task_budget_groups") != 9:
        raise SubmissionError("Bonus B optimization audit must contain nine task/budget groups")
    if optimization.get("candidates_per_group") != 4:
        raise SubmissionError("Bonus B optimization audit must compare four policies per group")
    if optimization.get("heldout_audit_episodes_per_candidate") != 15:
        raise SubmissionError("Bonus B optimization audit must hold out fifteen episodes")
    selectors = optimization.get("selectors", {})
    for name in ("small_critic", "robometer"):
        selector = selectors.get(name, {})
        if len(selector.get("selections", [])) != 9:
            raise SubmissionError(f"Bonus B selector {name} must retain nine decisions")
    audit_truth, audit_counts = _heldout_success(
        root / "artifacts" / "results" / "per_run.csv",
        root / "artifacts" / "results" / "progress_scores.json",
    )
    recomputed_optimization = optimize_selection_payload(bonus, audit_truth, audit_counts)
    if optimization != recomputed_optimization:
        raise SubmissionError("Bonus B optimization audit does not match its source evidence")

    pdf = root / "output" / "pdf" / "vla_budget_report.pdf"
    if require_pdf:
        if not pdf.is_file() or pdf.stat().st_size < 100_000:
            raise SubmissionError(f"final report is missing or unexpectedly small: {pdf}")
        if pdf.read_bytes()[:5] != b"%PDF-":
            raise SubmissionError(f"final report has no PDF signature: {pdf}")

    return {
        "status": "pass",
        "protocol_sha256": manifest["sha256"],
        "preregistrations": 5,
        "final_evaluation_units": units,
        "final_outcomes": outcomes,
        "failure_examples": failures,
        "bonus_ranked_cells": bonus["n_checkpoints"],
        "bonus_optimized_groups": optimization["n_task_budget_groups"],
        "pdf_checked": require_pdf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen submission evidence")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--without-pdf", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, require_pdf=not args.without_pdf), indent=2))


if __name__ == "__main__":
    main()
