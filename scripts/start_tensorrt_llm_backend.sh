#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MODEL_OR_ENGINE PORT" >&2
  exit 2
fi

model_or_engine="$1"
port="$2"
exec trtllm-serve "${model_or_engine}" --host 127.0.0.1 --port "${port}"
