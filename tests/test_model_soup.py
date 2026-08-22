from pathlib import Path

import pytest

from vla_budget.model_soup import interpolate_checkpoints


def test_checkpoint_interpolation_and_identity_guard(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    safe_open = pytest.importorskip("safetensors").safe_open
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    save_file({"w": torch.tensor([0.0, 4.0])}, left / "model.safetensors")
    save_file({"w": torch.tensor([4.0, 0.0])}, right / "model.safetensors")
    (left / "config.json").write_text("{}", encoding="utf-8")

    output = interpolate_checkpoints(left, right, tmp_path / "out", partner_weight=0.25)
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        torch.testing.assert_close(checkpoint.get_tensor("w"), torch.tensor([1.0, 3.0]))
    assert (output / "config.json").is_file()
    assert interpolate_checkpoints(left, right, output, partner_weight=0.25) == output
    with pytest.raises(ValueError, match="different identity"):
        interpolate_checkpoints(left, right, output, partner_weight=0.2)
