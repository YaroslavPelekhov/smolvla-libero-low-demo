$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $project 'tmp\pdfs\vla_budget_report'
$finalPdf = Join-Path $project 'output\pdf\vla_budget_report.pdf'
$projectPython = Join-Path $project '.venv\Scripts\python.exe'
$pythonExe = if (Test-Path $projectPython) {
    $projectPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}
$codexPoppler = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
$pdfinfoExe = if (Test-Path (Join-Path $codexPoppler 'pdfinfo.exe')) {
    Join-Path $codexPoppler 'pdfinfo.exe'
} else {
    (Get-Command pdfinfo -ErrorAction Stop).Source
}
$pdftoppmExe = if (Test-Path (Join-Path $codexPoppler 'pdftoppm.exe')) {
    Join-Path $codexPoppler 'pdftoppm.exe'
} else {
    (Get-Command pdftoppm -ErrorAction Stop).Source
}

Set-Location $project
& $pythonExe -m vla_budget.verify_submission --without-pdf
if ($LASTEXITCODE -ne 0) { throw 'Submission evidence verification failed' }
& $pythonExe report/build_latex_assets.py --results artifacts/results_ceiling --bonus-results artifacts/results/reward_ranking.json --report-dir report
if ($LASTEXITCODE -ne 0) { throw 'Result asset generation failed' }
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $finalPdf) | Out-Null

xelatex -interaction=nonstopmode -halt-on-error "-output-directory=$buildDir" report/main.tex
if ($LASTEXITCODE -ne 0) { throw 'First XeLaTeX pass failed' }
xelatex -interaction=nonstopmode -halt-on-error "-output-directory=$buildDir" report/main.tex
if ($LASTEXITCODE -ne 0) { throw 'Second XeLaTeX pass failed' }

$logPath = Join-Path $buildDir 'main.log'
$fatalLayout = Get-Content -LiteralPath $logPath | Select-String -Pattern @(
    'Overfull \\hbox',
    'LaTeX Warning: Citation.*undefined',
    'LaTeX Warning: Reference.*undefined'
)
if ($fatalLayout) {
    throw "LaTeX layout or reference validation failed:`n$($fatalLayout -join "`n")"
}

$builtPdf = Join-Path $buildDir 'main.pdf'
$pageLine = & $pdfinfoExe $builtPdf | Select-String '^Pages:'
$pages = [int](($pageLine -split ':', 2)[1].Trim())
if ($pages -ne 4) {
    throw "Report must contain exactly four pages; got $pages"
}

Copy-Item -LiteralPath $builtPdf -Destination $finalPdf -Force
Get-ChildItem -LiteralPath $buildDir -Filter 'page-*.png' -ErrorAction SilentlyContinue |
    Remove-Item -Force
& $pdftoppmExe -png -r 150 $finalPdf (Join-Path $buildDir 'page')
if ($LASTEXITCODE -ne 0) { throw 'Poppler rendering failed' }
& $pythonExe -m vla_budget.verify_submission
if ($LASTEXITCODE -ne 0) { throw 'Final submission verification failed' }

Write-Output "report=$finalPdf"
Write-Output "pages=$pages"
Write-Output "renders=$buildDir\page-1.png ... page-4.png"
