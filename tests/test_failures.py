from pathlib import Path

from vla_budget.failures import _select_diverse


def test_failure_selection_covers_tasks_and_staggers_budgets():
    candidates = []
    for task_id in range(3):
        for budget in (5, 10, 25):
            meta = {
                "method": "method",
                "budget": budget,
                "task_id": task_id,
                "seed": 17,
            }
            candidates.append((meta, 0, Path(f"task-{task_id}-{budget}.mp4")))
    selected = _select_diverse(candidates)
    assert [item[0]["task_id"] for item in selected] == [0, 1, 2]
    assert [item[0]["budget"] for item in selected] == [5, 10, 25]
