# Submission checklist

The intended hand-in is the complete repository plus `output/pdf/vla_budget_report.pdf`.

- [x] Correct NVIDIA LeRobot v3 dataset and exact prompt-based task join.
- [x] Nested first-demo prefixes; no cross-target demonstrations.
- [x] Two training seeds and at least 20 simulator episodes per task/checkpoint.
- [x] Zero-shot and paired wrong-language control.
- [x] Naive baseline and content-addressed predictions before every follow-up.
- [x] Preregistered 1/2-demo ceiling extension on a fresh reset seed (72/72 cells, 1,440 outcomes).
- [x] Three real rollout failure contact sheets with falsifiable experiments.
- [x] Bonus B comparison against Robometer-4B and proxy-optimization audit.
- [x] Four-page report generated from committed CSV/JSON evidence.
- [x] Local and CI checks for formatting, unit tests, and evidence integrity.

Final local gate:

```powershell
python -m pip install -e ".[dev]"
ruff check src tests report
pytest -q
vla-verify --without-pdf
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
vla-verify
git status --short
```

`git status --short` must be empty in the submitted commit. The large dataset, model checkpoints,
and raw rollout videos remain outside Git; their hashes, commands, Boolean outcome vectors, and
selected real failure frames are retained in the repository.
