#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# Keep Python on a local disk: the repository's share may not support symlinks.
env_dir="${SPARC_ENV_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/sparc/venv}"
if [[ ! -x "$env_dir/bin/python" ]]; then
    uv venv --managed-python --python 3.12 "$env_dir"
fi

uv pip install --python "$env_dir/bin/python" \
    --index-strategy unsafe-best-match -r requirements-resynthesis.txt

echo 'Ready: bash scripts/resynthesize.sh sample_audio/sample1.wav'
