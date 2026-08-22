import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "report" / "build_latex_assets.py"
_SPEC = importlib.util.spec_from_file_location("build_latex_assets", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_ceiling_text = _MODULE._ceiling_text
_paired_table = _MODULE._paired_table
_task_table = _MODULE._task_table


def test_ceiling_text_uses_all_budgets_and_strict_verdict():
    rows = [
        {"method": "zero", "budget": "0", "success_rate": "0", "n": "120"},
        {"method": "baseline", "budget": "1", "success_rate": "0.50", "n": "120"},
        {"method": "baseline", "budget": "2", "success_rate": "0.75", "n": "120"},
        {"method": "stabilized", "budget": "1", "success_rate": "0.60", "n": "120"},
        {"method": "stabilized", "budget": "2", "success_rate": "0.75", "n": "120"},
    ]

    summary, verdict = _ceiling_text(rows)

    assert "$k=1/2$" in summary
    assert "[60/90]/120" in summary
    assert "[72/90]/120" in summary
    assert "$k=2$" in verdict


def test_paired_table_discovers_registered_budgets():
    rows = []
    for budget in (1, 2):
        for task in range(3):
            for seed in (17, 29):
                rows.extend(
                    [
                        {
                            "method": "baseline",
                            "budget": str(budget),
                            "task_id": str(task),
                            "seed": str(seed),
                            "success_vector": "[false, true]",
                        },
                        {
                            "method": "stabilized",
                            "budget": str(budget),
                            "task_id": str(task),
                            "seed": str(seed),
                            "success_vector": "[true, true]",
                        },
                    ]
                )

    table = _paired_table(rows)

    assert "1 & +50.0" in table
    assert "2 & +50.0" in table


def test_task_table_discovers_low_demo_budgets():
    rows = []
    for budget in (1, 2, 5):
        for task in range(3):
            for seed in (17, 29):
                rows.extend(
                    [
                        {
                            "method": "baseline",
                            "budget": str(budget),
                            "task_id": str(task),
                            "seed": str(seed),
                            "successes": "10",
                            "n": "20",
                        },
                        {
                            "method": "stabilized",
                            "budget": str(budget),
                            "task_id": str(task),
                            "seed": str(seed),
                            "successes": "11",
                            "n": "20",
                        },
                    ]
                )

    table = _task_table(rows)

    assert "задача & 1 & 2 & 5" in table
    assert table.count(r"50.0$\to$55.0") == 9
