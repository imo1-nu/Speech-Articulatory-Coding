#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${SPARC_ENV_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/sparc/venv}"
export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.cache/matplotlib}"
cd "$repo_dir"
exec uv run --no-project --python "$env_dir/bin/python" -m sparc.cross_utterance_ppg "$@"
