#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 transformers|vllm MODEL REVISION DTYPE PORT" >&2
  exit 2
fi

backend="$1"
model="$2"
revision="$3"
dtype="$4"
port="$5"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${repo_root}/.venv-${backend}"

case "${backend}" in
  transformers)
    exec "${venv_dir}/bin/python" -m llm_systems_lab.backends.transformers_server \
      --model "${model}" \
      --revision "${revision}" \
      --dtype "${dtype}" \
      --port "${port}"
    ;;
  vllm)
    exec "${venv_dir}/bin/vllm" serve "${model}" \
      --revision "${revision}" \
      --dtype "${dtype}" \
      --port "${port}" \
      --generation-config vllm
    ;;
  *)
    echo "Unknown backend: ${backend}" >&2
    exit 2
    ;;
esac

