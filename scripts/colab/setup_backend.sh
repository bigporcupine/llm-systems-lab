#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 transformers|vllm" >&2
  exit 2
fi

backend="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${repo_root}/.venv-${backend}"

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip wheel

case "${backend}" in
  transformers)
    "${venv_dir}/bin/python" -m pip install -e "${repo_root}[exp002-transformers]"
    ;;
  vllm)
    "${venv_dir}/bin/python" -m pip install vllm
    "${venv_dir}/bin/python" -m pip install -e "${repo_root}"
    ;;
  *)
    echo "Unknown backend: ${backend}" >&2
    exit 2
    ;;
esac

"${venv_dir}/bin/python" --version
echo "Prepared ${backend} environment at ${venv_dir}"

