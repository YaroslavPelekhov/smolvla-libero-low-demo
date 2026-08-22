from __future__ import annotations

import csv
import json

import pytest

from vla_budget.bonus import _heldout_success, _safe_spearman, optimize_selection_payload


def test_safe_spearman_reports_constant_metric_as_unidentified():
    assert _safe_spearman([0.0, 0.0, 0.0], [0.1, 0.2, 0.3]) is None


def test_safe_spearman_returns_finite_ranking():
    assert _safe_spearman([0.0, 0.5, 1.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_reward_optimization_audits_selected_policy_against_truth():
    records = []
    for budget in (5,):
        for task_id in (0, 1):
            for method, seed, truth, small, robo in (
                ("baseline", 17, 0.9, 0.1, 0.9),
                ("baseline", 29, 0.8, 0.2, 0.8),
                ("method", 17, 1.0, 0.3, 1.0),
                ("method", 29, 0.4, 0.9, 0.4),
            ):
                records.append(
                    {
                        "key": [method, budget, task_id, seed],
                        "true_success": truth,
                        "small_critic": small,
                        "robometer": robo,
                    }
                )
    result = optimize_selection_payload({"records": records})
    assert result["n_task_budget_groups"] == 2
    assert result["baseline_reference_success"] == pytest.approx(0.85)
    assert result["selectors"]["small_critic"]["mean_selected_true_success"] == 0.4
    assert result["selectors"]["robometer"]["mean_selected_true_success"] == 1.0
    assert result["selectors"]["small_critic"]["suboptimal_selections"] == 2


def test_heldout_success_excludes_all_proxy_scored_episodes(tmp_path):
    per_run = tmp_path / "per_run.csv"
    with per_run.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "budget", "task_id", "seed", "success_vector"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "budget": 5,
                "task_id": 0,
                "seed": 17,
                "success_vector": json.dumps([True] * 5 + [False] * 15),
            }
        )
    proxy_scores = tmp_path / "scores.json"
    proxy_scores.write_text(
        json.dumps(
            [
                {
                    "method": "baseline",
                    "budget": 5,
                    "task_id": 0,
                    "seed": 17,
                    "episode_index": index,
                }
                for index in range(5)
            ]
        ),
        encoding="utf-8",
    )
    truth, counts = _heldout_success(per_run, proxy_scores)
    key = ("baseline", 5, 0, 17)
    assert truth[key] == 0.0
    assert counts[key] == 15
