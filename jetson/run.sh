#!/usr/bin/env bash
# Starts the LAN voice-chat server on the Jetson. Run from the deployed folder (~/voicechat-jetson).
set -euo pipefail
ROOT="${VC_ROOT:-$HOME/voicechat-jetson}"
cd "$ROOT"
export PRIME_ONNX="${PRIME_ONNX:-$HOME/primetts_v2/primetts_v21_3voice.onnx}"
export VC_ASR_DIR="${VC_ASR_DIR:-$HOME/xasr-eval/base}"
export VC_PORT="${VC_PORT:-8443}"
exec venv/bin/python app/server.py
