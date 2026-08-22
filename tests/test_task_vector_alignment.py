import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from vla_budget.task_vector_alignment import task_vector_alignment


def _checkpoint(path, values):
    path.mkdir(parents=True)
    safetensors.save_file(
        {"weight": torch.tensor(values, dtype=torch.float32)}, path / "model.safetensors"
    )


def test_task_vector_alignment_uses_seed_specific_seen_anchors(tmp_path):
    seen_a, target_a = tmp_path / "seen_a", tmp_path / "target_a"
    seen_b, target_b = tmp_path / "seen_b", tmp_path / "target_b"
    _checkpoint(seen_a, [10.0, 20.0])
    _checkpoint(target_a, [11.0, 22.0])  # delta [1, 2]
    _checkpoint(seen_b, [30.0, 40.0])
    _checkpoint(target_b, [32.0, 44.0])  # delta [2, 4]

    result = task_vector_alignment(seen_a, target_a, seen_b, target_b)

    assert result["cosine"] == pytest.approx(1.0)
    assert result["elements"] == 2
