# Voice Chat — Streaming Speech-to-Speech with Tools

**X-ASR-int8 → Qwen3.5-0.8B Q8_0 (Qwen-Agent, thinking) → Qwen3-TTS** — real models, no mocks, low latency.

A full-duplex voice chat demo: speak, get interrupted, search the web, and hear the answer. Bilingual (zh/en), multi-turn, tool-aware, and available at `https://training-machine.tailf63b31.ts.net` via Tailscale Funnel.

```
Mic 16k ──► VAD (Silero) ──► STT (X-ASR-int8 160ms, sherpa, 146M) ──► LLM (Qwen3.5-0.8B Q8_0, Qwen-Agent, thinking) ──► TTS (Qwen3-TTS Q8_0, 24k) ──► Speaker
                    ▲                    │  tools: web_search · get_weather · get_current_datetime      │
                    └────────────────────┴─── SearXNG official :8888 (180 engines, all - china) + wttr.in ──┘
                    Granite-97M Q8 (CUDA, 384d) rerank for paraphrases
```

## Stack

| Layer | Model / Service | Notes |
|---|---|---|
| **VAD** | Silero VAD | Turn-taking, barge-in |
| **STT** | `GilgameshWind/X-ASR-zh-en` int8 | sherpa-onnx Zipformer, 160ms streaming, `146M` encoder int8 (was `566M`), zh+en 16k, CUDA |
| **LLM** | `Qwen/Qwen3.5-0.8B` Q8_0 | `unsloth/Qwen3.5-0.8B-GGUF` `775M`, llama-server `:11435` `-c 16384` `Qwen-Agent` `thinking:True` `1024` tokens, `3` tools |
| **Embedding** | `ibm-granite/granite-embedding-97m-multilingual` Q8_0 | `115M` GGUF `384d` `zh-TW+en`, `llama-server :11434` `CUDA` `8ms` batch rerank `0.34-0.65` |
| **TTS** | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` Q8_0 | `Serveurperso qwen-talker-0.6b-customvoice-Q8_0.gguf` `924M` + `codec.gguf` `343M`, `qwentts-cpp-python==0.3.1+cu124` GGML CUDA, true streaming, `24k` `~20ms` TTFA |
| **Search** | SearXNG official `:8888` + wttr.in + Bing scrape | `180` engines `all - china` (`baidu`/`sogou`/`360search` disabled), `general`/`news`, honest `no curated mocks`, `wttr.in` real weather `+` `searxng` `+` `DDG`/`lite`/`Bing` fallback |
| **Agent** | `qwen-agent` `Assistant` | `Qwen-Agent` `function_list=[web_search, get_weather, get_current_datetime]`, `max_tokens 1024` `thinking:True` filtered, `smolagents` fallback `MAX_STEPS=2` |

All components are real — `mock=false` everywhere. Exhaustive `14/14` `100%` real `web_search` (official SearXNG `brave`/`google cse` `30` results `~1.2s` vs DIY `mock` `8.4s`). Fallbacks: Kokoro-1.0 → MOSS-TTS-Nano (ONNX `CUDA` `NaN` on `transformers 5.15.1`).

## Features

- **Full-duplex** — barge-in cancels TTS instantly, pre-roll 0.6s jitter buffer
- **Tool-aware** — `web_search` (SearXNG + wttr.in weather) and `get_current_datetime` (IANA tz) via native function calling
- **Bilingual** — zh/en auto-detected, zh-TW/zh-CN/en SearXNG routing
- **Low latency** — Svelte 5 + AudioWorklet + binary WS, whole-sentence TTS flush

Measured on RTX 3060: STT 300ms partial · LLM TTFT 100–400ms · TTS TTFB 0.4–1.2s · E2E ~1s (plain) / ~3s (search).

## Quick Start

```bash
# 1) Embedding (Granite-97M Q8, 115M, CUDA)
/home/user/llama.cpp/build/bin/llama-server \
  -m /tmp/granite-emb-gguf/granite-embedding-97M-multilingual-r2-Q8_0.gguf \
  --host 127.0.0.1 --port 11434 -c 8192 --alias granite-embedding --embedding --pooling mean --n-gpu-layers 99

# 2) LLM (Qwen3.5-0.8B Q8_0, 775M, thinking)
/home/user/llama.cpp/build/bin/llama-server \
  -m /tmp/llms/Qwen3.5-0.8B-Q8_0.gguf \
  --host 127.0.0.1 --port 11435 -c 16384 --alias qwen3.5-0.8b --n-gpu-layers 99 --jinja

# 3) SearXNG official (full, all - china)
SEARXNG_SETTINGS_PATH=/tmp/searxng/settings.yml python3 -m searx.webapp  # :8888, formats [html, json]

# 4) Backend
cd backend && python3 app.py --port 8000  # http://127.0.0.1:8000/health

# 5) Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173
# prod: npm run build — backend serves frontend/dist at /
```

Required files (mount as volumes in Docker):
- LLM: `/tmp/llms/Qwen3.5-0.8B-Q8_0.gguf` (`Q4` `508M` also works, `Q8_0` `775M` definitive)
- Embedding: `/tmp/granite-emb-gguf/granite-embedding-97M-multilingual-r2-Q8_0.gguf` `115M`
- TTS: `/tmp/qwen3_tts/talker_cv_q8.gguf` `924M` + `codec.gguf` `343M`
- STT: `/tmp/XASR/deployment/models/chunk-160ms-model` `encoder-160ms.int8.onnx` `146M` (+ `decoder` `2.8M`, `joiner` `2.5M`)
- SearXNG: `/tmp/searxng/settings.yml` (`180` engines `all - china` `html+json`)

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

- **Honest search** — No hard-coded mocks. `SearXNG official` `30` results `~1.2s` `brave`/`google cse` `all - china`; DIY minimal (`DDGS`+`lite` only `3` engines) was root cause of `8.4s` `mock` `example.com`. `wttr.in` is real weather API (`Paris` `16-22°C`). `Bing` scrape fallback when `SearXNG` `rel<0.34`.
- **Thinking without leakage** — `Qwen-Agent` `enable_thinking:True` `1024` tokens `reasoning_content` filtered (was leaking `21×` `AI news 2026 2027 2028` loop on `0.8B Q4`)
- **Clean UI** — `Svelte 5` `marked`+`DOMPurify` markdown (`**bold**` `ol`/`ul`), minimal single-column mobile, dark/light theme, pulse-free audio `0.6s` pre-roll.

## License

Code: Apache-2.0. Models: Qwen3.5-0.8B (Apache-2.0), Granite-embedding-97M (Apache-2.0), Qwen3-TTS (Apache-2.0), X-ASR-int8 (Apache-2.0), SearXNG official (AGPL-3.0).
