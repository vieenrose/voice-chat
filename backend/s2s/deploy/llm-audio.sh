#!/usr/bin/env bash
# Gemma 4 E4B with the multimodal projector loaded, so /v1/chat/completions
# accepts native audio. Same weights as llm.sh plus --mmproj.
#
# Mirrors HF's gemma4-12b-macos example (which uses -hf to fetch model+projector
# together); here both files are already on disk. -np 1 and -fa on are from that
# example. The "audio input is in experimental stage" warning comes from
# llama.cpp and is expected.
exec /home/user/llama.cpp/build/bin/llama-server \
  -m "${LLM_PATH_GEMMA:-/home/user/llms/gemma-qat/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf}" \
  --mmproj "${LLM_PATH_GEMMA_MMPROJ:-/home/user/llms/gemma-qat/mmproj-F16.gguf}" \
  --spec-draft-model "${LLM_PATH_GEMMA_MTP:-/home/user/llms/gemma-qat/MTP/mtp-gemma-4-E4B-it-Q8_0.gguf}" \
  --spec-type draft-mtp --spec-draft-n-max "${LLM_MTP_DRAFT_N:-3}" \
  --host 127.0.0.1 --port "${LLM_PORT:-11435}" \
  -c "${LLM_CTX:-16384}" -np 1 -fa on --alias gemma-4-e4b-qat \
  --n-gpu-layers 99 --jinja --reasoning-format deepseek
