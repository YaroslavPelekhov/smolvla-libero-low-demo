import json

from vla_budget.progress_critic import _evaluation_episodes


def test_evaluation_subset_uses_full_run_success_rate(tmp_path):
    run = tmp_path / "eval" / "run"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(
        json.dumps({"stage": "baseline", "method": "baseline", "budget": 10, "task_id": 0, "seed": 17})
    )
    metrics = {
        "successes": [True, False, False, True],
        "video_paths": ["one.mp4", "two.mp4"],
    }
    (run / "eval_info.json").write_text(json.dumps({"per_task": [{"metrics": metrics}]}))
    rows = list(_evaluation_episodes(tmp_path, max_per_run=2))
    assert len(rows) == 2
    assert {row[2] for row in rows} == {0.5}
