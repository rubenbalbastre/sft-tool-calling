#!/usr/bin/env bash
set -euo pipefail

# Transfer the outputs/ and data/ directories between this computer and RunPod.
#
# Usage:
#   ./scripts/transfer-runpod-artifacts.sh --upload [local-directory]
#   ./scripts/transfer-runpod-artifacts.sh --download [local-directory]
#
# Examples:
#   ./scripts/transfer-runpod-artifacts.sh --upload
#   ./scripts/transfer-runpod-artifacts.sh --download ./runpod-artifacts
#
# The local directory defaults to ./runpod-artifacts.

usage() {
    cat <<EOF
Usage: $0 --upload|--download [local-directory]

  --upload    Copy local outputs/ and data/ to RunPod.
  --download  Copy outputs/ and data/ from RunPod to this computer.

The local directory defaults to ./runpod-artifacts.

Examples:
  $0 --upload
  $0 --download ./runpod-artifacts
EOF
}

DIRECTION="${1:-}"
LOCAL_ROOT="${2:-./runpod-artifacts}"

RUNPOD_HOST=root@194.68.245.122
PORT=22026
REMOTE_ROOT=/root/sft-tool-calling
SSH_KEY="$HOME/.ssh/id_ed25519"

case "$DIRECTION" in
    --upload)
        for folder in outputs data; do
            if [[ ! -d "$LOCAL_ROOT/$folder" ]]; then
                echo "Missing $LOCAL_ROOT/$folder" >&2
                exit 1
            fi
        done

        echo "Uploading $LOCAL_ROOT/{outputs,data} to $RUNPOD_HOST:$REMOTE_ROOT/"
        ssh -p "$PORT" -i "$SSH_KEY" "$RUNPOD_HOST" "mkdir -p '$REMOTE_ROOT'"
        scp -r -P "$PORT" -i "$SSH_KEY" \
            "$LOCAL_ROOT/outputs" \
            "$LOCAL_ROOT/data" \
            "${RUNPOD_HOST}:${REMOTE_ROOT}/"
        ;;
    --download)
        mkdir -p "$LOCAL_ROOT"
        echo "Downloading $RUNPOD_HOST:$REMOTE_ROOT/{outputs,data} to $LOCAL_ROOT/"
        scp -r -P "$PORT" -i "$SSH_KEY" \
            "${RUNPOD_HOST}:${REMOTE_ROOT}/outputs" \
            "${RUNPOD_HOST}:${REMOTE_ROOT}/data" \
            "$LOCAL_ROOT/"
        ;;
    --help|-h)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
