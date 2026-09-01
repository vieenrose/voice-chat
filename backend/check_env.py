#!/usr/bin/env python3
"""Environment self-check: does this interpreter have what the default stack needs?

backend/requirements.txt used to be wrong in both directions — pins for packages the
repo never imports, and nothing at all for sherpa-onnx / qwen-agent / jieba / lxml /
faster-qwen3-tts / qwentts-cpp-python, which the STT+TTS+agent import ladders require.
That failure mode was silent: every ladder has a try/except, so the service booted
"ok" with mock/degraded backends.

Exit code is non-zero when a REQUIRED import is missing, so this can gate a deploy:
    python3 backend/check_env.py && python3 backend/app.py --port 8000

Usage:
    python3 backend/check_env.py            # required + optional
    python3 backend/check_env.py --quiet    # only print problems
"""
import argparse
import importlib
import sys

# (import name, distribution, why) — REQUIRED = needed by the default configuration
# documented in README.md (X-ASR + llama-server + Qwen-Agent + Qwen3-TTS).
REQUIRED = [
    ("fastapi", "fastapi", "HTTP/WS server"),
    ("uvicorn", "uvicorn[standard]", "ASGI server"),
    ("httpx", "httpx", "llama-server + SearXNG + wttr clients"),
    ("numpy", "numpy", "PCM buffers"),
    ("psutil", "psutil", "llama-server port ownership, /health RSS"),
    ("loguru", "loguru", "logging"),
    ("lxml", "lxml", "web_search scraping / page enrichment"),
    ("jieba", "jieba", "CJK tokenization for relevance + query reformulation"),
    ("sherpa_onnx", "sherpa-onnx", "STT: X-ASR streaming recognizer"),
    ("qwen_agent", "qwen-agent", "agent harness (default): function calling"),
    ("faster_qwen3_tts", "faster-qwen3-tts", "TTS: Qwen3-TTS GGML runtime"),
    ("qwentts_cpp", "qwentts-cpp-python", "TTS: GGML backend wheel (cu124 index)"),
    ("websockets", "websockets", "test/e2e clients"),
]

OPTIONAL = [
    ("smolagents", "smolagents", "fallback agent harness"),
    ("pydantic_ai", "pydantic-ai-slim", "fallback agent harness"),
    ("funasr", "funasr", "STT fallback (Paraformer)"),
    ("faster_whisper", "faster-whisper", "STT fallback"),
    ("kokoro_onnx", "kokoro-onnx", "TTS fallback (Kokoro ONNX)"),
    ("onnxruntime", "onnxruntime(-gpu)", "TTS fallbacks' inference provider"),
    ("duckduckgo_search", "duckduckgo_search", "web_search fallback backend"),
    ("torch", "torch", "ML fallback adapters (not needed by the default stack)"),
]

EXTERNAL = {
    # Not pip-installed: mounted / cloned separately. Listed so a failing check is
    # explained rather than mysterious.
    "llama-server": "llama.cpp build (LLAMA_SERVER_BIN, default /home/user/llama.cpp/build/bin/llama-server)",
    "searx (webapp)": "official SearXNG (SEARXNG_URL, default http://localhost:8888)",
    "/tmp/qwen3_tts/*.gguf": "TTS GGUF weights (or TTS_MODEL_DIR)",
    "/tmp/XASR/…/chunk-160ms-model": "X-ASR ONNX weights (or STT_MODEL_DIR)",
    "/tmp/llms/*.gguf": "LLM GGUF weights (or LLM_PATH_*)",
}


def check(pairs, quiet):
    missing = []
    for mod, dist, why in pairs:
        try:
            importlib.import_module(mod)
            if not quiet:
                print(f"  ok    {mod:<20} ({dist})")
        except Exception as e:
            missing.append((mod, dist, why, type(e).__name__, str(e)[:120]))
            print(f"  MISS  {mod:<20} ({dist}) — {why}")
            if not quiet:
                print(f"          {type(e).__name__}: {str(e)[:120]}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    print("\nRequired for the documented default stack:")
    req_missing = check(REQUIRED, args.quiet)
    print("\nOptional (fallback adapters):")
    opt_missing = check(OPTIONAL, args.quiet)

    print("\nProvided outside pip (mounted volumes / separate services):")
    for k, v in EXTERNAL.items():
        print(f"  - {k}: {v}")

    if opt_missing:
        print(f"\n{len(opt_missing)} optional import(s) unavailable — the corresponding"
              " fallback chain will be skipped at runtime.")
    if req_missing:
        print(f"\nFAIL: {len(req_missing)} required import(s) missing. Install them before"
              " starting the backend, or the STT/TTS/agent ladders will silently degrade:")
        print("    python3 -m pip install -r backend/requirements.txt")
        return 1
    print("\nOK: environment satisfies the default stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
