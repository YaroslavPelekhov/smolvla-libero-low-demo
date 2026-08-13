from vla_budget.aggregate import summarize


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
