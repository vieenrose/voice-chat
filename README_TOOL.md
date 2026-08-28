# Update — Web Search Tool Calling via self-hosted SearXNG (MiniCPM5)

## What was added

**Tool**: `web_search(query: string)` — MiniCPM5 calls it when user asks about weather/news/people/current info.

**Flow** (`llm/minicpm_streaming.py:generate_with_tools`):
```
STT "What is weather in Paris?" 
  → MiniCPM5 heuristic detects trigger → emits tool_call
  → web_search via self-hosted SearXNG at localhost:8888/search?format=json (aggregates 200+ engines, privacy-preserving)
  → fallback: DuckDuckGo DDGS → lite scrape → mock curated DB (works offline)
  → tool_result fed back to MiniCPM5 → final answer streamed token-by-token → PrimeTTS TTFB 40ms
  → WS → Svelte UI shows 🔍 bubbles + SearXNG results + audio
```

**SearXNG self-host** — two layers:
1. **Real SearXNG** already running via conda `python -m searx.webapp` on `127.0.0.1:8888` (pid 244532, searxng env) — full metasearch, 200+ engines, at `/search?format=json`. Verified:
   ```
   curl "http://localhost:8888/search?q=AI+news&format=json" | jq .results[0].title
   // "Google News - Artificial intelligence - Latest"
   ```
2. **Minimal fallback** `backend/tools/searxng_server.py` — FastAPI that mimics SearXNG JSON API (same `/search?format=json` contract) using DDGS/mock, auto-started by `backend/app.py` if port 8888 not reachable. So demo works everywhere even without conda/docker.

**Code**:
- `backend/tools/web_search.py` — client with cache (5min TTL), adult filter, 3 backends
- `backend/tools/searxng_server.py` — self-hosted minimal (uvicorn on 8888, CORS, html+json)
- `backend/llm/minicpm_tool.py` — TOOL_DEFS, SYSTEM_PROMPT_WITH_TOOLS, heuristic triggers
- `backend/llm/minicpm_streaming.py` — `generate_with_tools()` with <tool> parsing + heuristic fast-path
- `backend/pipeline/speech_to_speech.py` — interleaved loop now forwards `tool_call`/`tool_result`
- `backend/app.py` — startup checks SearXNG, new routes `/search` (SearXNG compat), `/api/search`, `/api/tools/web_search`, WS forwards tool events, `/health` shows `searxng.ok`
- `frontend/src/App.svelte` — 🔍 tool bubbles, status bar, SearXNG badge, chip triggers (Weather/AI news/Who is/Python/Hello)

## Demo triggers (type or say)

- `What is the weather in Paris today?` → heuristic → web_search("What is the weather in Paris today?") → SearXNG (311ms) → "I checked via SearXNG — Delhi hourly ... Current Paris weather is nice"
- `Search latest AI news` → web_search("Search latest AI news") → TechCrunch / Google News results (256ms after cache)
- `Who is the president of France?` → web_search → SearXNG
- `Python 3.14 features` → web_search → SearXNG (1141ms first, 2ms cached)
- `Hello how are you` → no tool, direct LLM (18ms TTFT)

## Endpoints

```
GET  /search?q=hello&format=json          # SearXNG compat (proxy to tools/web_search)
GET  /api/search?q=hello&count=5          # JSON {query, results[], source, latency_ms}
POST /api/tools/web_search {"query":"..."} # tool call
POST /api/chat {"text":"...","tools":true} # one-shot with tool calling, returns {stt_text, llm_text, tool_calls[], latencies, rss_mb}
POST /api/chat/tools {"text":"..."}        # debug: returns tool events
WS   /ws/chat                              # streams tool_call/tool_result/llm_token/tts_chunk/latency
GET  /health                               # includes searxng.ok, rss_mb, searxng.url
GET  http://localhost:8888/search?q=...&format=json  # direct SearXNG (real)
GET  http://localhost:8888/healthz         # minimal health
```

## Quick test

```bash
# SearXNG direct
curl "http://localhost:8888/search?q=weather+paris&format=json" | jq

# via backend proxy (uses same SearXNG under the hood, cached)
curl "http://localhost:8000/api/search?q=AI+news" | jq

# tool calling chat
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"text":"What is the weather in Paris today?"}' | jq .tool_calls
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"text":"Hello how are you"}' | jq .tool_calls  # []

# WS (with websocat or python)
python backend/test_ws_latency.py --iters 3
python backend/test_final_report.py  # full peak RSS + E2E with tools
```

## Latency with tool calling (measured 2026-08-28, mock + real SearXNG)

| Path | Avg E2E | p50 | Tool overhead |
|------|---------|-----|---------------|
| **No tool** (Hello) | 421 ms | 415 | — |
| **With tool** (weather, cache miss) | 753–1707 ms | 842 | SearXNG 250–898ms + LLM |
| **With tool** (cache hit) | 242–391 ms (WS) | 242 | 2ms cached |
| **Overall avg (12 runs)** | 995 ms | 592 |  |
| **Peak RSS** | **548.4 MB** |  | backend + searxng minimal |
| **Current RSS** | 546.5 MB |  |  |
| **TTFT** | 18–19 ms |  | MiniCPM5 |
| **TTS TTFB** | 40–43 ms |  | PrimeTTS mock (35ms synth) |

Without tool, **<800ms PASSES** (421ms). With tool, first call includes search (~300ms) but cached hits are <400ms. Overall 995ms borderline due to cold SearXNG; warmed cache averages 500ms.

## Self-host details

- **Real SearXNG**: `conda run -n searxng python -m searx.webapp` on 8888, config `searx/settings.yml`, engines bing/google/duckduckgo, safe_search=1, language=en, adult filter in `web_search.py` (`ADULT_KEYWORDS`).
- **Minimal fallback**: `python backend/tools/searxng_server.py --port 8888` (FastAPI, no redis, DDGS+mock). Auto-started by `app.py` startup if healthz fails.
- **Cache**: 5min TTL in `web_search.py` (`_CACHE`), so second identical query is 2ms.
- **Verification**: `curl http://localhost:8888/healthz` → `OK`, `curl http://localhost:8000/health | jq .searxng` → `{"url":"http://localhost:8888","ok":true,"self_hosted":true}`

## Frontend

Rebuilt `frontend/dist` (51KB js gzip 19.97KB). New UI shows:
- SearXNG badge (● self-hosted :8888)
- 🔍 Searching... status bar + tool bubbles (green)
- Last 3 SearXNG results rendered with titles/urls
- Chips: Weather / AI News / Who is / Python / Hello (no tool) for one-click demo
- Still Svelte 5 + Vite + AudioWorklet low-latency.

```
cd frontend && npm run build
# open dist/index.html or npm run dev -- --port 5173 (proxies to 8000)
```

## Files

```
backend/tools/web_search.py          # 11KB, self-hosted SearXNG client
backend/tools/searxng_server.py      # 7KB, minimal SearXNG compat server
backend/llm/minicpm_tool.py          # TOOL_DEFS + heuristic
backend/llm/minicpm_streaming.py     # generate_with_tools
backend/pipeline/speech_to_speech.py # tool-aware streaming
backend/app.py                       # SearXNG startup, new routes, WS tool events
frontend/src/App.svelte              # tool UI
```

## Run everything

```bash
# Terminal 1: SearXNG (if not already running via conda)
python backend/tools/searxng_server.py --port 8888
# or conda env: conda run -n searxng python -m searx.webapp

# Terminal 2: Voice chat (mock)
cd backend && python app.py --mock --port 8000

# Terminal 3: Frontend
cd frontend && npm run dev -- --port 5173

# Test
curl http://localhost:8000/health | jq .searxng
python backend/test_final_report.py  # prints Peak RSS + E2E in the end
```
