import json
from pathlib import Path

from vla_budget.failures import _candidate_failures, _select_diverse


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


def test_candidate_failures_keeps_only_failed_video_indices(tmp_path: Path):
    run = tmp_path / "eval" / "method" / "task-0"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(
        json.dumps({"stage": "method", "method": "method", "task_id": 0, "budget": 5}),
        encoding="utf-8",
    )
    (run / "eval_info.json").write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "metrics": {
                            "successes": [False, True, False],
                            "video_paths": [
                                str(run / "eval_episode_0.mp4"),
                                str(run / "eval_episode_1.mp4"),
                                str(run / "eval_episode_2.mp4"),
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = _candidate_failures(tmp_path / "eval")

    assert [item[1] for item in candidates] == [0, 2]
