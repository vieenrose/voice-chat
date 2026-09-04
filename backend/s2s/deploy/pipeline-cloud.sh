#!/usr/bin/env bash
# Cloud-answering pipeline: Qwen3-ASR transcribes (STT only, local, in-process
# via transformers -- see s2s/stt_qwen3_asr.py), a hosted LLM on OpenCode Go
# answers and calls the tools.
#
# No local llama-server is needed for either the answering model (that is
# OpenCode Go, a network call) or STT (Qwen3-ASR loads through transformers
# directly inside this same process, on first use). Verified by starting this
# with nothing listening on :11435 at all -- it comes up clean.
#
# --responses_api_base_url / --model_name / --responses_api_api_key below are
# therefore never actually contacted: they only satisfy `--llm_backend
# chat-completions`'s argument dataclass, which our own get_llm_handler
# override (s2s/serve.py._get_llm_handler) replaces outright. LLM_API_BASE /
# LLM_MODEL_ID / LLM_API_KEY are what agent/qwen_harness.py._endpoint() reads
# for the real tool-calling call, and that is the OpenCode Go endpoint.
#
# LLM_API_KEY is deliberately left unset here: paste it into the 語言模型 card
# in the UI (POST /v1/llm-config), which holds it in this process's environment
# only -- never logged, never written to disk. Every turn will fail with "抱歉，
# API 金鑰無效" until a key is set; that is the correct behaviour, not a bug.
export LLM_API_BASE="${LLM_API_BASE:-https://opencode.ai/zen/go/v1}"
# muse-spark-1.3-contributor was tried here and reverted: it is a /v1/responses
# model (see agent/qwen_harness.py's _RESPONSES_API_PREFIXES), and this script
# still only speaks chat/completions. mimo-v2.5 is confirmed chat/completions
# per OpenCode Go's own docs.
export LLM_MODEL_ID="${LLM_MODEL_ID:-mimo-v2.5}"
# Carried over from the local-E4B default: a conservative sampling temperature
# for reciting an exact value (an extension, a date) rather than paraphrasing
# it -- see pipeline.sh for the measurement this was pinned from. Not yet
# re-measured against a hosted OpenCode Go model.
export LLM_AGENT_TEMP="${LLM_AGENT_TEMP:-0.2}"
cd /home/user/voice-chat/backend
exec python3 -m s2s.serve --mode realtime \
  --ws_host 127.0.0.1 --ws_port 8765 \
  --stt none \
  --llm_backend chat-completions \
  --model_name gemma-4-e4b-qat \
  --responses_api_base_url http://127.0.0.1:11435/v1 \
  --responses_api_api_key "none" \
  --responses_api_audio_content_type input_audio \
  --responses_api_stream \
  --min_silence_ms 700 \
  --tts qwen3 --qwen3_tts_backend ggml --qwen3_tts_language zh \
  --qwen3_tts_speaker Vivian \
  --qwen3_tts_instruct "用台灣人的口音說話，語氣親切自然，像在跟朋友聊天" \
  --qwen3_tts_gguf_talker_path /tmp/qwen3_tts/talker_cv_q8.gguf \
  --qwen3_tts_gguf_codec_path  /tmp/qwen3_tts/codec.gguf \
  --log_level info
