import json

import pytest

from vla_budget.verify_submission import SubmissionError, _check_preregistration


def test_preregistration_accepts_matching_digest(tmp_path):
    prediction = tmp_path / "predictions_v3.yaml"
    prediction.write_text("hypothesis: frozen\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "preregistration_v3.json").write_text(
        json.dumps({"predictions_file": prediction.name, "sha256": digest}), encoding="utf-8"
    )
    _check_preregistration(tmp_path, 3)


def test_preregistration_rejects_modified_prediction(tmp_path):
    prediction = tmp_path / "predictions_v3.yaml"
    prediction.write_text("hypothesis: changed\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "preregistration_v3.json").write_text(
        json.dumps({"predictions_file": prediction.name, "sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(SubmissionError, match="hash mismatch"):
        _check_preregistration(tmp_path, 3)
