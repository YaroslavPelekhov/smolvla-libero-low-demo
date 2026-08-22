from vla_budget.compare_methods import compare


def test_paired_comparison_counts_wins_and_losses():
    rows = []
    for method, vector in (("baseline", [True, False, True]), ("stabilized", [True, True, False])):
        rows.append(
            {
                "method": method,
                "budget": 5,
                "task_id": 0,
                "seed": 17,
                "success_vector": vector,
            }
        )
    result = compare(rows)
    row = result["aggregate"][0]
    assert row["wins"] == 1
    assert row["losses"] == 1
    assert row["delta_percentage_points"] == 0
    assert row["mcnemar_exact_p"] == 1
