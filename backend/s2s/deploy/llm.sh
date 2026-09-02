#!/usr/bin/env bash
# Gemma 4 E4B (Q4_K_M) on llama.cpp, serving the OpenAI API on :11435.
#
# --jinja is what enables native tool calls: without it the model's own chat
# template is not applied and the agent falls back to qwen-agent's prompt dialect,
# which is measurably worse at emitting a well-formed call.
#
# No --spec-type: this repo ships the NextN/MTP head as a separate file
# (MTP/mtp-gemma-4-E4B-it-Q8_0.gguf), not inside these weights, so there is
# nothing here for self-speculative decoding to draft from.
exec /home/user/llama.cpp/build/bin/llama-server \
  -m "${LLM_PATH_GEMMA:-/home/user/llms/gemma/gemma-4-E4B-it-Q4_K_M.gguf}" \
  --host 127.0.0.1 --port "${LLM_PORT:-11435}" \
  -c "${LLM_CTX:-8192}" --alias gemma-4-e4b \
  --n-gpu-layers 99 --jinja --reasoning-format deepseek
