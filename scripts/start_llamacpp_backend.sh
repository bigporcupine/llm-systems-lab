#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MODEL.gguf PORT" >&2
  exit 2
fi

model_path="$1"
port="$2"
exec llama-server --model "${model_path}" --host 127.0.0.1 --port "${port}" --ctx-size 8192 --n-gpu-layers 0
