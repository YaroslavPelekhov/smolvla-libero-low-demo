import pytest

from vla_budget.aggregate import _write_csv, summarize, validate_coverage


def test_summary_pools_bernoulli_counts_and_keeps_zero():
    rows = [
        {"stage": "zero", "method": "correct", "budget": 0, "task_id": 0, "seed": 17, "n": 20, "successes": 2},
        {"stage": "zero", "method": "correct", "budget": 0, "task_id": 1, "seed": 17, "n": 20, "successes": 4},
        {"stage": "baseline", "method": "baseline", "budget": 5, "task_id": 0, "seed": 17, "n": 20, "successes": 10},
        {"stage": "baseline", "method": "baseline", "budget": 5, "task_id": 0, "seed": 29, "n": 20, "successes": 14},
    ]
    summary = summarize(rows)
    zero = next(row for row in summary if row["method"] == "zero")
    baseline = next(row for row in summary if row["method"] == "baseline")
    assert zero["success_rate"] == 6 / 40
    assert baseline["success_rate"] == 24 / 40
    assert baseline["seeds"] == 2


def test_csv_writer_accepts_method_specific_metadata(tmp_path):
    output = tmp_path / "rows.csv"
    _write_csv(output, [{"stage": "baseline"}, {"stage": "stabilized", "mix": 0.25}])
    assert output.read_text(encoding="utf-8").splitlines() == [
        "stage,mix",
        "baseline,",
        "stabilized,0.25",
    ]


def _coverage_cfg():
    return {
        "dataset": {"target_task_ids": [0]},
        "protocol": {"train_seeds": [17], "budgets": [5]},
    }


def _complete_rows():
    return [
        {"stage": "zero", "budget": 0, "task_id": 0, "seed": 17},
        {"stage": "language", "budget": 0, "task_id": 0, "seed": 17},
        {"stage": "baseline", "budget": 5, "task_id": 0, "seed": 17},
        {"stage": "method", "budget": 5, "task_id": 0, "seed": 17},
    ]


def test_coverage_rejects_duplicate_cells():
    rows = _complete_rows()
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_coverage(rows, _coverage_cfg())


def test_coverage_rejects_unexpected_cells():
    rows = _complete_rows()
    rows.append({"stage": "diagnostic", "budget": 5, "task_id": 0, "seed": 17})
    with pytest.raises(ValueError, match="unexpected"):
        validate_coverage(rows, _coverage_cfg())


def test_coverage_can_exclude_zero_and_language_controls():
    cfg = _coverage_cfg()
    cfg["protocol"]["include_controls"] = False
    rows = [
        {"stage": "baseline", "budget": 5, "task_id": 0, "seed": 17},
        {"stage": "method", "budget": 5, "task_id": 0, "seed": 17},
    ]
    validate_coverage(rows, cfg)
