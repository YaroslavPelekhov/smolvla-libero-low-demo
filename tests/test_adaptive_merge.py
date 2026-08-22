import pytest

from vla_budget.adaptive_merge import high_dispersion_tasks


def test_high_dispersion_gate_is_inclusive_at_median():
    selected, threshold = high_dispersion_tasks({0: 0.30, 1: 0.10, 2: 0.20})
    assert threshold == pytest.approx(0.20)
    assert selected == {0, 2}


def test_high_dispersion_gate_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        high_dispersion_tasks({})
