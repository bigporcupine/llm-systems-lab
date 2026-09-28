#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "Usage: $0 BASE_MODEL BASE_REVISION ADAPTER_NAME ADAPTER_PATH DTYPE PORT [QUANTIZATION]" >&2
  exit 2
fi

base_model="$1"
base_revision="$2"
adapter_name="$3"
adapter_path="$4"
dtype="$5"
port="$6"
quantization="${7:-none}"

command=(
  vllm serve "${base_model}"
  --revision "${base_revision}"
  --dtype "${dtype}"
  --port "${port}"
  --generation-config vllm
  --enable-lora
  --lora-modules "${adapter_name}=${adapter_path}"
)
if [[ "${quantization}" != "none" ]]; then
  command+=(--quantization "${quantization}")
fi

exec "${command[@]}"
