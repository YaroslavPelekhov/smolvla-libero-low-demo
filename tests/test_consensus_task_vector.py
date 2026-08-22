import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from vla_budget.consensus_task_vector import build_consensus_checkpoint


def _checkpoint(path, values):
    path.mkdir(parents=True)
    safetensors.save_file(
        {"weight": torch.tensor(values, dtype=torch.float32)}, path / "model.safetensors"
    )
    (path / "config.json").write_text("{}\n", encoding="utf-8")


def test_consensus_merge_keeps_anchor_on_sign_conflict(tmp_path):
    paths = {name: tmp_path / name for name in ("seen_a", "target_a", "seen_b", "target_b")}
    _checkpoint(paths["seen_a"], [10.0, 10.0, 10.0])
    _checkpoint(paths["target_a"], [12.0, 8.0, 12.0])  # [2, -2, 2]
    _checkpoint(paths["seen_b"], [20.0, 20.0, 20.0])
    _checkpoint(paths["target_b"], [24.0, 24.0, 19.0])  # [4, 4, -1]
    output = tmp_path / "output"

    build_consensus_checkpoint(
        seen_anchor=paths["seen_a"],
        target_anchor=paths["target_a"],
        seen_partner=paths["seen_b"],
        target_partner=paths["target_b"],
        output=output,
        partner_weight=0.25,
    )

    result = safetensors.load_file(output / "model.safetensors")["weight"]
    assert result.tolist() == pytest.approx([12.5, 8.0, 12.0])
    manifest = json.loads((output / "consensus_vector_manifest.json").read_text())
    assert manifest["agreed_elements"] == 1
    assert manifest["active_elements"] == 3
