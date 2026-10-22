#!/usr/bin/env bash
# speech-to-speech 1.0.0 on an 8 GB Apple Silicon Mac. Every model is sized so the whole
# pipeline stays resident: once the weights spill to swap, each stage reloads them and a
# turn takes over a minute instead of a few seconds.
#   STT: Qwen3-ASR 0.6B 4-bit (MLX, via launch.py)   LLM: Bonsai 4B ternary (mlx-lm), or with
#   LLM=bonsai2 Bonsai v2 2B (run_llm_bonsai2.sh serves it on :11435; mlx-lm cannot load it)
#   TTS: Qwen3-TTS 0.6B CustomVoice 6-bit (mlx-audio)
# Run from the external SSD root: /Volumes/DevSSD/voice-chat-ios
set -euo pipefail

ROOT=/Volumes/DevSSD/voice-chat-ios
export HF_HOME="$ROOT/hf"
export HF_HUB_OFFLINE=0            # cached weights are reused; Smart Turn is fetched once
export S2S_ASR_BACKEND="${S2S_ASR_BACKEND:-mlx}"

if [[ "${LLM:-bonsai4}" == bonsai2 ]]; then
  LLM_ARGS=(--llm_backend chat-completions --responses_api_base_url http://127.0.0.1:11435/v1
            --responses_api_api_key none --model_name prism-ml/Ternary-Bonsai-2-2B-mlx-2bit)
else
  LLM_ARGS=(--llm_backend mlx-lm --model_name prism-ml/Ternary-Bonsai-4B-mlx-2bit)
fi

cd "$ROOT/server"
exec "$ROOT/venv-s2s1/bin/python" "$ROOT/server/launch.py" serve \
  --stt qwen3-asr \
  --qwen3_asr_model_name mlx-community/Qwen3-ASR-0.6B-4bit \
  --qwen3_asr_language zh \
  --no_enable_live_transcription \
  --init_chat_prompt "${CHAT_PROMPT:-你是友善的語音助理。請用繁體中文，以一兩句簡短口語回答，不要用 Markdown、列點或表情符號。}" \
  "${LLM_ARGS[@]}" \
  --tts qwen3 \
  --qwen3_tts_model_name mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8765}" \
  --log_level "${LOG_LEVEL:-info}"
