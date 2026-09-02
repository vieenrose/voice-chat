#!/usr/bin/env bash
# Gemma 4 E4B, QAT release at UD-Q4_K_XL, on llama.cpp serving the OpenAI API.
#
# --jinja enables native tool calls; without it the agent falls back to
# qwen-agent's prompt dialect, which is measurably worse at emitting a call.
#
# The MTP (NextN) head ships as its own file in this repo rather than inside the
# weights, so it is passed as the draft model: --spec-draft-model plus
# --spec-type draft-mtp. That is worth 106 tok/s against 79 without it.
exec /home/user/llama.cpp/build/bin/llama-server \
  -m "${LLM_PATH_GEMMA:-/home/user/llms/gemma-qat/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf}" \
  --spec-draft-model "${LLM_PATH_GEMMA_MTP:-/home/user/llms/gemma-qat/MTP/mtp-gemma-4-E4B-it-Q8_0.gguf}" \
  --spec-type draft-mtp --spec-draft-n-max "${LLM_MTP_DRAFT_N:-3}" \
  --host 127.0.0.1 --port "${LLM_PORT:-11435}" \
  -c "${LLM_CTX:-8192}" --alias gemma-4-e4b-qat \
  --n-gpu-layers 99 --jinja --reasoning-format deepseek
