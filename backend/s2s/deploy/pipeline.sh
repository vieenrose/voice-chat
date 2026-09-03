#!/usr/bin/env bash
# DEFAULT pipeline: Gemma 4 native speech input, no external STT stage.
# Native audio input WITH the tools: no STT stage, and the turn still runs through
# our own loop, so web_search / get_weather / get_current_datetime stay available.
#
# This is the difference from pipeline-audio.sh, which follows HF's example and
# hands audio to the framework's own chat-completions stage -- that path reads
# request.audio but knows nothing about our tools. Ours carries both: the spoken
# turn becomes an input_audio content part and the tool schemas ride along.
# Measured on Gemma 4 E4B, a tool routes from speech alone in 0.2-0.6 s.
#
# Needs llm-audio.sh (llama-server with --mmproj) rather than llm.sh.
#
# Known gap: with --stt none there is no transcript, so the user's own bubble in
# the UI stays empty and the debug export has no rawStt. The assistant side is
# unaffected.
# --min_silence_ms 700, not the 300 of HF's example. Measured from an exported
# session, 300 ms ended turns mid-question on Mandarin: VAD segments came out at
# 0.75-2.04 s (mean 1.44) and the captions show exactly where they were cut --
# 喂，你好 became 喂， 你 and 請問今天幾月幾號 became 請問今天幾月幾. The model then
# answered the fragment, so a date question got a Chongqing weather report.
# Mandarin has intra-sentence pauses longer than 300 ms; this costs 400 ms of
# endpointing after the user stops and buys whole utterances.
export LLM_MODEL_ID=gemma-4-e4b-qat
# Reciting an extension back is the one thing this demo must not get wrong, and at
# the default 0.7 it did: the tool returned 分機 1102 every time while the spoken
# answer said 3567 and 5522 -- 1 turn in 3. At 0.2, 8 runs produced no wrong number.
# The model still paraphrases freely; it just stops improvising digits.
export LLM_AGENT_TEMP="${LLM_AGENT_TEMP:-0.2}"
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
  --qwen3_tts_speaker Vivian \
  --qwen3_tts_instruct "用台灣人的口音說話，語氣親切自然，像在跟朋友聊天" \
  --qwen3_tts_gguf_talker_path /tmp/qwen3_tts/talker_cv_q8.gguf \
  --qwen3_tts_gguf_codec_path  /tmp/qwen3_tts/codec.gguf \
  --log_level info
