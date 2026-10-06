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
    uv pip install --python .venv/bin/python -r requirements.txt

# FlashAttention's build imports Torch, so install it only after the main
# environment is complete and disable build isolation. Limit parallel compiler
# jobs to avoid exhausting RunPod system RAM.
UV_CACHE_DIR="$PROJECT_UV_CACHE" UV_LINK_MODE=copy \
    uv pip install --python .venv/bin/python packaging ninja
MAX_JOBS="${MAX_JOBS:-4}" UV_CACHE_DIR="$PROJECT_UV_CACHE" UV_LINK_MODE=copy \
    uv pip install --python .venv/bin/python \
    "flash-attn>=2.8,<3" --no-build-isolation

.venv/bin/python -c "import flash_attn; print('FlashAttention:', flash_attn.__version__)"

echo "Virtual environment created and dependencies installed."
