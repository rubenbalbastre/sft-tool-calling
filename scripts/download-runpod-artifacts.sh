#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-./runpod-artifacts}"
mkdir -p "$DESTINATION"

RUNPOD_HOST=root@213.173.105.13
PORT=24168

scp -r -P "$PORT" -i "$HOME/.ssh/id_ed25519" \
    $RUNPOD_HOST:/root/sft-tool-calling/outputs \
    $RUNPOD_HOST:/root/sft-tool-calling/data \
    "$DESTINATION"
