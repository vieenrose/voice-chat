# Voice Chat — full-duplex streaming speech-to-speech with tools

**X-ASR (STT) → Qwen3.5-4B-MTP (LLM + native tools) → Qwen3-TTS 0.6B Q8 (streaming TTS)**

A real-model, no-mock full-duplex voice chat demo with web-search tool calling,
multi-turn history (bilingual zh/en), and public HTTPS access via Tailscale funnel.

```
STT (X-ASR sherpa 160ms streaming, zh+en)
      │  PCM 16k (mic) / text_input
      ▼
LLM (Qwen3.5-4B-MTP Q4_K_M, llama-server, native OpenAI tools)
      │  tool_calls: web_search / get_current_datetime
      ▼
SearXNG (self-hosted :8888) + wttr.in weather + Wikipedia fallback
      │  tool_result injected
      ▼
TTS (Qwen3-TTS-12Hz-0.6B-CustomVoice Q8 GGUF, faster_qwen3_tts cu124,
     TRUE token streaming, TTFA ~20ms, silence-compressed prosody)
      │  PCM 24k chunks over WS
      ▼
Browser (Svelte 5, AudioWorklet, pre-roll jitter buffer 0.6s, barge-in)
```

## ✨ Features

- **Full duplex**: talk while the assistant speaks; barge-in cancels TTS instantly.
- **Real models only** (`mock=false` everywhere):
  - STT: `GilgameshWind/X-ASR-zh-en` — sherpa-onnx Zipformer, 160 ms streaming, zh+en, CUDA.
  - LLM: `unsloth/Qwen3.5-4B-MTP-GGUF` Q4_K_M — llama-server, native `tools=[web_search, get_current_datetime]`.
  - TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` Q8 — `qwentts-cpp-python==0.3.1+cu124` GGML CUDA, true streaming.
  - Fallbacks: Kokoro-1.0 (82M ONNX) → MOSS-TTS-Nano-100M ONNX (CUDA EP) → VoxCPM2.
- **Tool calling**: self-hosted SearXNG (`!bing` engine), entity-first zh/en query crafting,
  **wttr.in** direct weather source (no API key, zh+en, today/tomorrow), Wikipedia fallback.
- **Date-aware agent**: today/tomorrow weekday/date injected into every prompt, plus a `get_current_datetime` tool (IANA timezone).
- **Latency discipline** (measured on RTX 3060): LLM TTFT ~100–400 ms, TTS TTFB ~0.4–1.2 s,
  E2E ≈ 1 s (plain) / ~3 s (search), whole-sentence TTS chunks, 0 mid-sentence pauses
  (sentence-only flush — Ling/Qwen3 stream char-level tokens, so an old 8-token flush would chop every sentence).
- **Remote access**: public HTTPS via Tailscale funnel, valid cert, iPhone-tested.
  Demo: `https://training-machine.tailf63b31.ts.net`

## 🚀 Quick start (local)

```bash
# 1) LLM (llama-server, CUDA)
/home/user/llama.cpp/build/bin/llama-server \
  -m /tmp/llms/Qwen3.5-4B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 11435 -c 8192 --alias qwen3.5-4b-mtp --n-gpu-layers 99 --jinja

# 2) Backend (FastAPI + WS)
cd backend && python3 app.py --port 8000        # health: http://127.0.0.1:8000/health

# 3) Frontend (dev) 
cd frontend && npm install && npm run dev       # http://localhost:5173

# 3b) Frontend (prod, single port) — backend serves ./frontend/dist automatically
cd frontend && npm run build
```

Model files expected on disk (mount/volumes in Docker):
- LLM: `/tmp/llms/Qwen3.5-4B-Q4_K_M.gguf` (or `--llm` override)
- TTS: `/tmp/qwen3_tts/talker_cv_q8.gguf` + `/tmp/qwen3_tts/codec.gguf`
- STT: `/tmp/XASR/deployment/models/chunk-160ms-model`
- SearXNG: `:8888` (self-hosted, `!bing` engine enabled)

## 🐳 Docker

```bash
docker compose up -d --build          # backend + llama-server (CUDA)
# UI served on http://localhost:8000 (FastAPI serves ./frontend/dist)
```
Volumes: mount your model dirs at `/models/llm`, `/models/tts`, `/models/stt` (paths are configurable via env,
see `docker-compose.yml`). CUDA users: `--gpus all` + nvidia-container-toolkit.

## 🔌 API

| Endpoint | Description |
|---|---|
| `GET /health` | models loaded, RSS MB, mock flag |
| `WS /ws/chat?session_id=…` | full-duplex audio / text chat (binary PCM in, JSON events out: `stt_final`, `llm_token`, `tool_call`, `tool_result`, `tts_chunk`, `tts_end`, `latency`) |
| `POST /api/chat` | JSON chat (non-stream) |
| `GET /api/search?q=…` | SearXNG + wttr.in + factoid search |
| `GET /stats` | counters + latency history |

## 🧩 Troubleshooting notes

- **Mid-sentence pauses**: every TTS showed them until the *true* root cause was fixed —
  Ling/Qwen3.5 stream **one character per token**, and three code paths flushed TTS at
  `>=8 tokens ending in ' '` → 8-character chops. Fixed to flush at sentence-end only
  (`app.py direct_tts`, `pipeline/stream_chat_interleaved`) + a 300-token safety cap.
- **Tool XML**: Ling can emit `<tool_call>web_search…` XML as content — parsed & stripped
  (`_strip_tool_xml`) so tool queries never get spoken.
- **Multi-tool requests**: keep each OpenAI tool schema well-formed — a malformed
  `"required":[…]` caused llama-server HTTP 500 `type must be string, but is object`.
- **AI components**: Silero VAD · X-ASR · Qwen3.5-4B-MTP · Qwen3-TTS 0.6B — all real models.

## 📜 License / models

Code: Apache-2.0 intent. Models:
Qwen3.5-4B-MTP (Apache-2.0), Qwen3-TTS (Apache-2.0), X-ASR (Apache-2.0),
Kokoro-1.0 (Apache-2.0), MOSS-TTS-Nano (Apache-2.0), SearXNG (AGPL-3.0).