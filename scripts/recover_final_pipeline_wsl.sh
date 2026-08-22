#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${VLA_OUTPUT:-/home/root/outputs/vla-benchmark-final}
PROJECT=${VLA_PROJECT:-/mnt/c/Users/psgpe/Downloads/world}

exec >> "$OUTPUT/recovery_pipeline.log" 2>&1
cd "$PROJECT"
printf '[%s] restarting incomplete evaluation cells serially\n' "$(date -Is)"
VLA_PROJECT="$PROJECT" VLA_OUTPUT="$OUTPUT" VLA_EVAL_JOBS=1 \
  bash scripts/run_benchmark_wsl.sh
printf '[%s] benchmark recovered; starting postprocess\n' "$(date -Is)"
VLA_PROJECT="$PROJECT" VLA_OUTPUT="$OUTPUT" \
  bash scripts/continue_postprocess_wsl.sh
