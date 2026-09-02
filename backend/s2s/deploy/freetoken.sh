#!/usr/bin/env bash
# NOT the default, and its weights are no longer on disk: re-download
# nvidia/Qwen3.6-35B-A3B-NVFP4 (23.4 GB) first. Kept because the four flags below
# are the entire difference between this working and not, and each cost a
# debugging session to find.
# nvcc here is 12.4 while torch ships cu130; the mismatch guard is what its own
# error message points at. Needed at RUNTIME too, not just at install time.
export FREETOKEN_ALLOW_CUDA_MISMATCH=1
# flashinfer's JIT passed --compress-mode=size, which nvcc 12.4 rejects; that flag
# is patched out of the installed flashinfer instead of shimming nvcc, because
# flashinfer derives CUDA_HOME from `which nvcc` and a shim dir breaks its include
# paths (/usr/include/c++/13/cmath: math.h not found).
export CUDA_HOME=/usr
# Qwen3.6-35B-A3B (NVFP4) on FreeToken, serving the OpenAI API on :11435 --
# the same port llama-server used, so nothing downstream changes.
#
# --attention-backend triton \
  --moe-backend offload is the point: 35B of weights will not fit in 12 GB, but
# only ~3B are active per token (the A3B), so the routed experts live in host RAM
# and FreeToken pulls the ones each token needs onto the GPU through an LRU cache.
# --memory-ratio leaves room for the speech stages, which already hold ~3.5 GB.
exec /home/user/ft-venv/bin/ft serve \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen3.6-35b-a3b \
  --host 127.0.0.1 --port 11435 \
  --attention-backend triton \
  --moe-backend offload \
  --moe-cpu-threads 16 \
  # qwen3_coder, not qwen: this model emits the XML-ish dialect
  #   <tool_call><function=get_weather><parameter=location>台北</parameter></function></tool_call>
  # while the `qwen` parser expects <tool_call>{"name":...}</tool_call>. With the
  # wrong parser the call stays in content, tool_calls comes back null, and the
  # turn ends as "I could not find an answer".
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --memory-ratio 0.62 \
  --max-seq-len-override 8192
