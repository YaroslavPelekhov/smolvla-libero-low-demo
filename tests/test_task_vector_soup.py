import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from vla_budget.task_vector_soup import build_task_vector_checkpoint


def _checkpoint(path, value):
    path.mkdir(parents=True)
    safetensors.save_file(
        {"weight": torch.tensor([value], dtype=torch.float32)}, path / "model.safetensors"
    )
    (path / "config.json").write_text("{}\n", encoding="utf-8")


def test_task_vector_interpolation_preserves_anchor_base(tmp_path):
    paths = {name: tmp_path / name for name in ("seen_a", "target_a", "seen_b", "target_b")}
    _checkpoint(paths["seen_a"], 10.0)
    _checkpoint(paths["target_a"], 12.0)  # delta_a = 2
    _checkpoint(paths["seen_b"], 20.0)
    _checkpoint(paths["target_b"], 24.0)  # delta_b = 4
    output = tmp_path / "output"

    build_task_vector_checkpoint(
        seen_anchor=paths["seen_a"],
        target_anchor=paths["target_a"],
        seen_partner=paths["seen_b"],
        target_partner=paths["target_b"],
        output=output,
        partner_weight=0.25,
    )

    result = safetensors.load_file(output / "model.safetensors")["weight"].item()
    assert result == pytest.approx(12.5)  # 10 + 0.75*2 + 0.25*4
    manifest = json.loads((output / "task_vector_manifest.json").read_text(encoding="utf-8"))
    assert manifest["partner_weight"] == 0.25
    assert manifest["processor_source"] == str(paths["target_a"])
