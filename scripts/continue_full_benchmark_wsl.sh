#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
PROJECT=${VLA_PROJECT:-/mnt/c/Users/psgpe/Downloads/world}

printf '[%s] waiting for both 100k seen checkpoints\n' "$(date -Is)"
while [[ ! -f "$OUTPUT/train/seen/seed-17/COMPLETED" || ! -f "$OUTPUT/train/seen/seed-29/COMPLETED" ]]; do
  sleep 30
done

printf '[%s] seen checkpoints complete; starting confirmatory matrix\n' "$(date -Is)"
cd "$PROJECT"
VLA_PROJECT="$PROJECT" VLA_OUTPUT="$OUTPUT" VLA_EVAL_JOBS=4 bash scripts/run_benchmark_wsl.sh
