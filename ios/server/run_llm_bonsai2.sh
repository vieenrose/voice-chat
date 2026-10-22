#!/usr/bin/env bash
# Bonsai v2 (2B, Hadamard-rotated ternary) behind an OpenAI-compatible endpoint on 127.0.0.1:11435.
# mlx-vlm >= 0.7.8 loads it (text only here); mlx-lm 0.32 does not. Own venv, own process.
set -euo pipefail
ROOT=/Volumes/DevSSD/voice-chat-ios
export HF_HOME="$ROOT/hf"
exec "$ROOT/venv-bonsai2/bin/python" -m mlx_vlm.server \
  --model prism-ml/Ternary-Bonsai-2-2B-mlx-2bit --host 127.0.0.1 --port "${LLM_PORT:-11435}"
