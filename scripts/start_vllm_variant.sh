#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 MODEL REVISION DTYPE PORT [QUANTIZATION] [LOAD_FORMAT]" >&2
  exit 2
fi

model="$1"
revision="$2"
dtype="$3"
port="$4"
quantization="${5:-none}"
load_format="${6:-auto}"

command=(vllm serve "${model}" --revision "${revision}" --dtype "${dtype}" --port "${port}" --generation-config vllm)
if [[ "${quantization}" != "none" ]]; then
  command+=(--quantization "${quantization}")
fi
if [[ "${load_format}" != "auto" ]]; then
  command+=(--load-format "${load_format}")
fi
exec "${command[@]}"
