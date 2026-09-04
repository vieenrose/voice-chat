#!/usr/bin/env bash
# Gemma 4 E2B with the multimodal projector loaded, dedicated to captioning
# (s2s/caption_gemma.py). A separate process and port from llm-audio.sh's E4B, so
# a slow or stalled caption call can only make the caption late -- it cannot
# block the turn the conversation actually runs on.
#
# E2B rather than E4B: a caption is a transcription, not an answer, so the
# smaller model's weaker reasoning doesn't cost it anything here, and it leaves
# more of the 12 GB card for the E4B server already resident on it.
#
# -c 4096: a caption call carries at most a few seconds of audio and asks for a
# short transcript back, nowhere near what a multi-turn conversation needs.
# --reasoning off: reasoning and the answer share one max_tokens budget, and
# asked to "transcribe" cold (no tool schema, no system prompt) E2B reliably
# spent the whole budget narrating a "Thinking Process..." plan and returned
# empty content (finish_reason "length", 0 transcript chars) -- measured before
# this flag was added. The caller (s2s/caption_gemma.py) also constrains the
# reply to {"transcript": "..."} JSON, which independently stops E2B restating
# the instruction and burying the transcript mid-paragraph in quotes.
exec /home/user/llama.cpp/build/bin/llama-server \
  -m "${LLM_PATH_GEMMA_E2B:-/home/user/llms/gemma-qat-e2b/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf}" \
  --mmproj "${LLM_PATH_GEMMA_E2B_MMPROJ:-/home/user/llms/gemma-qat-e2b/mmproj-F16.gguf}" \
  --spec-draft-model "${LLM_PATH_GEMMA_E2B_MTP:-/home/user/llms/gemma-qat-e2b/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf}" \
  --spec-type draft-mtp --spec-draft-n-max "${LLM_MTP_DRAFT_N:-3}" \
  --host 127.0.0.1 --port "${LLM_CAPTION_PORT:-11436}" \
  -c 4096 -np 1 -fa on --alias gemma-4-e2b-caption \
  --n-gpu-layers 99 --jinja --reasoning-format deepseek --reasoning off
