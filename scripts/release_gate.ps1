$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $project '.venv\Scripts\python.exe'
$ruff = Join-Path $project '.venv\Scripts\ruff.exe'
$pytest = Join-Path $project '.venv\Scripts\pytest.exe'

if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Missing .venv. Run: py -3.11 -m venv .venv; .venv\Scripts\pip install -e ".[dev]"'
}

Set-Location $project

& $ruff check src tests report
if ($LASTEXITCODE -ne 0) { throw 'Ruff failed' }

& $pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Pytest failed' }

& $projectPython -m vla_budget.verify_submission --without-pdf
if ($LASTEXITCODE -ne 0) { throw 'Evidence verification failed' }

& powershell -ExecutionPolicy Bypass -File scripts\build_report.ps1
if ($LASTEXITCODE -ne 0) { throw 'Report build failed' }

& $projectPython -m vla_budget.verify_submission
if ($LASTEXITCODE -ne 0) { throw 'Final PDF verification failed' }

git diff --check
if ($LASTEXITCODE -ne 0) { throw 'Git whitespace validation failed' }

Write-Output 'release_gate=pass'
