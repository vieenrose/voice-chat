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
# Picked on measured latency AND measured accuracy, in that order of surprise --
# reproduce both with
#   python -m s2s.checks.model_latency --max-input-price 0.40
#   python -m s2s.checks.accuracy --model deepseek-v4-flash
# deepseek-v4-flash answers in 1.2-2.2s to first delta against 5.1-6.0s for
# muse-spark-1.3-contributor (the previous pick) and 7.9s for mimo-v2.5, and it
# is the only candidate that also held up on accuracy: 10/11 on the varied-shape
# suite, its one miss being a harness bug that hits every model (see
# agent/qwen_harness.py._has_word -- a digits-only answer is discarded).
#
# muse-spark-1.2-contributor was measured faster still on paper and is cheaper
# ($0.10/$0.20 per Mtok against $0.22/$0.66), and was briefly the default here.
# It is NOT usable: it intermittently emits its pre-tool preamble and then stops
# -- no tool call, no answer, so the user hears 「好的，我查一下。」 and then
# silence. Reproduced in-process on the negation case (6 of 8 attempts) and over
# the live pipeline on both weather and web_search in s2s.checks.exhaustive.
# Cheap and fast is worth nothing if the turn does not finish.
#
# An earlier note here said muse-spark was reverted for being a /v1/responses
# model while this script "only speaks chat/completions". That is no longer
# true -- agent/native_loop.py._run_turn_responses_api serves the whole
# _RESPONSES_API_PREFIXES family. deepseek-v4-flash is chat/completions either
# way. gpt-5.6-luna, by contrast, is in the catalogue but 400s on every turn.
export LLM_MODEL_ID="${LLM_MODEL_ID:-deepseek-v4-flash}"
# Carried over from the local-E4B default: a conservative sampling temperature
# for reciting an exact value (an extension, a date) rather than paraphrasing
# it -- see pipeline.sh for the measurement this was pinned from.
#
# Since re-measured against a hosted model, and it matters far more than a
# sampling knob usually does: s2s.checks.accuracy scored the SAME model 6/11 at
# the harness's own 0.7 default against 10/11 at this 0.2, because the extra
# temperature made it stop after its pre-tool preamble instead of going on to
# call the tool. Whatever else changes here, this pin earns its keep.
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
