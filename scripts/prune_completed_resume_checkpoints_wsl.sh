#!/usr/bin/env bash
set -euo pipefail

readonly ALLOWED_ROOT=/home/root/outputs/vla-ceiling-v6/train/baseline
readonly ROOT=${1:-$ALLOWED_ROOT}
resolved_root=$(realpath -e "$ROOT")
if [[ "$resolved_root" != "$ALLOWED_ROOT" ]]; then
  printf 'refusing unexpected root: %s\n' "$resolved_root" >&2
  exit 2
fi

removed=0
bytes=0
while IFS= read -r -d '' candidate; do
  target=$(realpath -e "$candidate")
  case "$target" in
    "$resolved_root"/*/checkpoints/008000) ;;
    *)
      printf 'refusing unsafe target: %s\n' "$target" >&2
      exit 3
      ;;
  esac
  cell=${target%/checkpoints/008000}
  [[ -f "$cell/COMPLETED" ]] || continue
  [[ -d "$cell/checkpoints/010000/pretrained_model" ]] || continue
  size=$(du -sb "$target" | cut -f1)
  rm -rf -- "$target"
  removed=$((removed + 1))
  bytes=$((bytes + size))
done < <(find "$resolved_root" -type d -path '*/checkpoints/008000' -print0)

printf 'removed_checkpoints=%d\n' "$removed"
awk -v bytes="$bytes" 'BEGIN {printf "reclaimed_gib=%.3f\n", bytes / 1073741824}'
