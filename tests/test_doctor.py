from __future__ import annotations

import hashlib
import json

from vla_budget import doctor
from vla_budget.protocol import manifest_digest


def test_protocol_check_uses_file_recorded_by_preregistration(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions_v2.yaml"
    predictions.write_text("hypothesis: corrected\n", encoding="utf-8")
    manifest = {"seen": {"total_tasks": 1, "total_episodes": 2}, "target": {"tasks": {"0": {}}}}
    manifest["sha256"] = manifest_digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    preregistration = {
        "predictions_file": predictions.name,
        "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
    }
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    monkeypatch.setattr(doctor, "resolve", lambda value: tmp_path / value)

    result = doctor._protocol_check(
        {"paths": {"manifest": manifest_path.name, "preregistration": preregistration_path.name}}
    )

    assert result["ok"] is True
    assert result["predictions_file"] == str(predictions)


def test_protocol_check_reports_missing_registered_predictions(tmp_path, monkeypatch):
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(
        json.dumps({"predictions_file": "missing.yaml", "sha256": "unused"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(doctor, "resolve", lambda value: tmp_path / value)

    result = doctor._protocol_check(
        {"paths": {"manifest": manifest_path.name, "preregistration": preregistration_path.name}}
    )

    assert result["ok"] is False
    assert "registered predictions file is missing" in result["error"]
