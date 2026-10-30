#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
env_dir="${SPARC_ENV_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/sparc/venv}"
export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
export MPLCONFIGDIR="$repo_dir/.cache/matplotlib"
export PYTHONPATH="$repo_dir/.cache/feedback-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$env_dir/bin/python" -m experiments.feedback.control_run "$@"
