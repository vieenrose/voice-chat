#!/usr/bin/env bash
# Starts the LAN voice-chat server on the Jetson. Run from the deployed folder (~/voicechat-jetson).
set -euo pipefail
ROOT="${VC_ROOT:-$HOME/voicechat-jetson}"
cd "$ROOT"
export PRIME_ONNX="${PRIME_ONNX:-$HOME/primetts_v2/primetts_v21_3voice.onnx}"
export VC_ASR_DIR="${VC_ASR_DIR:-$HOME/xasr-new/480}"
export VC_PORT="${VC_PORT:-8443}"
export VC_SMART_TURN="${VC_SMART_TURN:-1}"  # Smart Turn on: turn_check values are logged to server/rec/*.log for tuning
export VC_LLM="${VC_LLM:-Ternary-Bonsai-2-4B-PTQ1_0.gguf}"  # 4B ternary; set VC_LLM=Ternary-Bonsai-2-2B-PTQ1_0.gguf for the 2B
exec venv/bin/python app/server.py
