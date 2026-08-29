# Voice Chat — Streaming Speech-to-Speech with Tools

**X-ASR → Granite-4.2-3B (smolagents) → Qwen3-TTS** — real models, no mocks, low latency.

A full-duplex voice chat demo: speak, get interrupted, search the web, and hear the answer. Bilingual (zh/en), multi-turn, tool-aware, and available at `https://training-machine.tailf63b31.ts.net` via Tailscale Funnel.

```
Mic 16k ──► VAD (Silero) ──► STT (X-ASR 160ms, sherpa) ──► LLM (Granite-4.2-3B, smolagents) ──► TTS (Qwen3-TTS Q8, 24k) ──► Speaker
                    ▲                    │  tools: web_search · get_current_datetime                    │
                    └────────────────────┴─── SearXNG :8888 + wttr.in ──────────────────────────────────┘
```

## Stack

| Layer | Model / Service | Notes |
|---|---|---|
| **VAD** | Silero VAD | Turn-taking, barge-in |
| **STT** | `GilgameshWind/X-ASR-zh-en` | sherpa-onnx Zipformer, 160ms streaming, zh+en 16k |
| **LLM** | `ibm-granite/granite-4.2-3b-GGUF` Q4_K_M | llama-server `:11435`, native tools, `smolagents` ToolCallingAgent |
| **TTS** | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` Q8_0 | `qwentts-cpp-python` GGML CUDA, true streaming, 24k |
| **Search** | SearXNG `:8888` + wttr.in | General + news categories, honest scoring, no curated mocks |
| **Agent** | `smolagents` 1.26 | ToolCallingAgent, max 3 steps, auto tool_choice |

All components are real — `mock=false` everywhere. Fallbacks: Kokoro-1.0 → MOSS-TTS-Nano.

## Features

- **Full-duplex** — barge-in cancels TTS instantly, pre-roll 0.6s jitter buffer
- **Tool-aware** — `web_search` (SearXNG + wttr.in weather) and `get_current_datetime` (IANA tz) via native function calling
- **Bilingual** — zh/en auto-detected, zh-TW/zh-CN/en SearXNG routing
- **Low latency** — Svelte 5 + AudioWorklet + binary WS, whole-sentence TTS flush

Measured on RTX 3060: STT 300ms partial · LLM TTFT 100–400ms · TTS TTFB 0.4–1.2s · E2E ~1s (plain) / ~3s (search).

## Quick Start

```bash
# 1) LLM
/home/user/llama.cpp/build/bin/llama-server \
  -m /tmp/llms/granite-4.2-3b-Q4_K_M.gguf \
  --host 127.0.0.1 --port 11435 -c 8192 --alias granite-4.2-3b --n-gpu-layers 99 --jinja

# 2) Backend
cd backend && python3 app.py --port 8000  # http://127.0.0.1:8000/health

# 3) Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173
# prod: npm run build — backend serves frontend/dist at /
```

Required files (mount as volumes in Docker):
- LLM: `/tmp/llms/granite-4.2-3b-Q4_K_M.gguf`
- TTS: `/tmp/qwen3_tts/talker_cv_q8.gguf` + `codec.gguf`
- STT: `/tmp/XASR/deployment/models/chunk-160ms-model`

## Docker

```bash
docker compose up -d --build
# UI at http://localhost:8000
```

See `docker-compose.yml` for volume mounts (`/models/llm`, `/models/tts`, `/models/stt`).

## API

| Endpoint | Description |
|---|---|
| `GET /health` | models, RSS, mock flag, SearXNG status |
| `WS /ws/chat?session_id=…` | Binary PCM in → `stt_final` / `llm_token` / `tool_call` / `tool_result` / `tts_chunk` / `tts_end` / `latency` out |
| `POST /api/chat` | JSON chat (non-stream) |
| `GET /api/search?q=…` | Honest SearXNG search (no curated mocks) |
| `GET /stats` | Counters & latencies |

## Notes

- **Honest search** — No hard-coded mocks. SearXNG relevance <0.34 falls through to DDG → lite scrape → `mock-offline` only. `wttr.in` is a real weather API, not a mock.
- **Clean UI** — Minimal, single-column on mobile, dark/light theme, pulse-free audio.

## License

Code: Apache-2.0. Models: Granite-4.2 (Apache-2.0), Qwen3-TTS (Apache-2.0), X-ASR (Apache-2.0), SearXNG (AGPL-3.0).
