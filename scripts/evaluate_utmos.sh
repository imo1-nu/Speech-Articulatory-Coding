#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${SPARC_ENV_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/sparc/venv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$repo_dir/.cache/numba}"
cd "$repo_dir"
exec uv run --no-project --python "$env_dir/bin/python" -m sparc.evaluate_utmos "$@"
