#!/usr/bin/env bash
# Native-audio variant: no STT stage at all. Each completed VAD segment goes
# straight to Gemma as input_audio.
#
# Follows HF's gemma4-12b-macos example. Two consequences worth stating:
#
#  * S2S_USE_UPSTREAM_LLM=1 is required. --stt none delivers audio on
#    GenerateResponseRequest.audio, and only the framework's own
#    chat-completions stage reads it; our Qwen-Agent stage is text-only. So this
#    variant runs WITHOUT the three tools -- no web_search, get_weather or
#    get_current_datetime -- exactly like the upstream example.
#  * --responses_api_audio_content_type input_audio is the payload shape current
#    llama.cpp accepts; audio_url is refused.
export S2S_USE_UPSTREAM_LLM=1
export LLM_MODEL_ID=gemma-4-e4b-qat
export LLM_API_BASE=http://127.0.0.1:11435/v1
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
  --qwen3_tts_gguf_talker_path /tmp/qwen3_tts/talker_cv_q8.gguf \
  --qwen3_tts_gguf_codec_path  /tmp/qwen3_tts/codec.gguf \
  --log_level info
