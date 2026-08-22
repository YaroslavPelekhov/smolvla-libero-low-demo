import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from vla_budget.checkpoint_distance import checkpoint_distance


def _checkpoint(path, values):
    path.mkdir(parents=True)
    safetensors.save_file(
        {"weight": torch.tensor(values, dtype=torch.float32)}, path / "model.safetensors"
    )


def test_checkpoint_distance_reads_safe_open_keys(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _checkpoint(left, [1.0, 2.0])
    _checkpoint(right, [2.0, 4.0])

    result = checkpoint_distance(left, right)

    assert result["rms_delta"] == pytest.approx((2.5) ** 0.5)
    assert result["elements"] == 2
