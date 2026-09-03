#!/usr/bin/env bash
# STT variant: Paraformer transcribes, the model gets text. Kept because it is the
# only path that produces a user transcript -- see the note in the default script.
export LLM_MODEL_ID=gemma-4-e4b-qat
export LLM_API_BASE=http://127.0.0.1:11435/v1
cd /home/user/voice-chat/backend
exec python3 -m s2s.serve --mode realtime \
  --ws_host 127.0.0.1 --ws_port 8765 \
  --stt paraformer --language zh \
  --model_name gemma-4-e4b-qat \
  --responses_api_base_url http://127.0.0.1:11435/v1 \
  --responses_api_api_key "none" \
  --tts qwen3 \
  --qwen3_tts_speaker Vivian \
  --qwen3_tts_instruct "用台灣人的口音說話，語氣親切自然，像在跟朋友聊天" \
  --qwen3_tts_backend ggml \
  --qwen3_tts_gguf_talker_path /tmp/qwen3_tts/talker_cv_q8.gguf \
  --qwen3_tts_gguf_codec_path /tmp/qwen3_tts/codec.gguf \
  --qwen3_tts_language zh \
  --enable_live_transcription \
  --log_level info
