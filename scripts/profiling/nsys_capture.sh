#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_PREFIX COMMAND [ARG ...]" >&2
  exit 2
fi

output_prefix="$1"
shift
mkdir -p "$(dirname "${output_prefix}")"
exec nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output="${output_prefix}" \
  "$@"
