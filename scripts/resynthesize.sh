#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${SPARC_ENV_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/sparc/venv}"
if [[ ! -x "$env_dir/bin/python" ]]; then
    echo 'Run bash scripts/setup_resynthesis.sh first.' >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
exec uv run --no-project --python "$env_dir/bin/python" -m sparc.resynthesize "$@"
