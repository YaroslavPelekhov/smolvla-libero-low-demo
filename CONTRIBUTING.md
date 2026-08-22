# Contributing

This repository treats experimental evidence as immutable input, not editable prose. Changes should
preserve the distinction between registered results, diagnostics, and excluded audits.

## Development workflow

1. Create a focused branch from `main`.
2. Install the supported environment with `python -m pip install -e ".[dev]"`.
3. Add or update tests for any protocol, aggregation, or verification change.
4. Run `powershell -ExecutionPolicy Bypass -File scripts/release_gate.ps1`.
5. Open a pull request describing the scientific estimand affected by the change.

## Evidence rules

- Never replace a failed or negative registered matrix with a favorable rerun.
- Never edit `per_run.csv` or paired statistics by hand; regenerate derived files from raw eval JSON.
- Put exploratory results under `artifacts/diagnostics/` and invalid runs under `artifacts/audit/`.
- Freeze hypotheses and stopping rules in a content-addressed prediction file before evaluation.
- Do not commit checkpoints, datasets, raw videos, credentials, or machine-local temporary files.

Pull requests that modify registered evidence must state whether the estimand changed and identify the
matching preregistration record. Numerical report changes must be traceable to committed CSV/JSON.
