#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv .venv --python python3

# RunPod workspaces may not support the filesystem operations used by uv's
# default cache and clone modes. Keep the cache on local temporary storage and
# copy packages into the environment.
PROJECT_UV_CACHE="${TMPDIR:-/tmp}/sft-tool-calling-uv-cache"
mkdir -p "$PROJECT_UV_CACHE"
UV_CACHE_DIR="$PROJECT_UV_CACHE" UV_LINK_MODE=copy \
    uv pip install --python .venv/bin/python --torch-backend=auto \
        -r requirements.txt

echo "Virtual environment created and dependencies installed."
