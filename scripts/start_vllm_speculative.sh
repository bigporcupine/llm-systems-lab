#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 TARGET_MODEL TARGET_REVISION PORT SPECULATIVE_CONFIG_JSON DTYPE" >&2
  exit 2
fi

target_model="$1"
target_revision="$2"
port="$3"
speculative_config="$4"
dtype="$5"

# Keep the version-specific speculative configuration explicit in the run
# artifact instead of hiding draft settings inside this launcher.
exec vllm serve "${target_model}" \
  --revision "${target_revision}" \
  --dtype "${dtype}" \
  --port "${port}" \
  --generation-config vllm \
  --speculative-config "${speculative_config}"
