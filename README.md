# Voice Chat — Streaming Speech-to-Speech with Tools

**X-ASR-int8 → Qwen3.5-2B Q4 (Qwen-Agent, thinking) → Qwen3-TTS** — real models, no mocks, low latency.

A full-duplex voice chat demo: speak, get interrupted (by voice *or* text, either direction), search the web, and hear the answer. **Traditional Chinese (Taiwan, zh-TW) by default** — speech input and output default to zh-TW regardless of what language the question was asked in, with English mixed in only for proper nouns/terms that don't translate well. Multi-turn, tool-aware, and available at `https://training-machine.tailf63b31.ts.net` via Tailscale Funnel.

```
Mic 16k ──► Endpoint detect (sherpa-onnx) ──► STT (X-ASR-int8 160ms, 146M) ──► LLM (Qwen3.5-2B Q4, Qwen-Agent, thinking) ──► TTS (Qwen3-TTS Q8_0, 24k) ──► Speaker
                    ▲                    │  tools: web_search · get_weather · get_current_datetime      │
                    └────────────────────┴─── SearXNG official :8888 (180 engines, all - china) + wttr.in ──┘
                    Granite-97M Q8 (CUDA, 384d) rerank for paraphrases
```

## Stack

| Layer | Model / Service | Notes |
|---|---|---|
| **Turn-taking** | sherpa-onnx rule-based endpointing | Trailing-silence rules (`rule1`/`rule2`/`rule3`) drive `stt_final`; no separate VAD model in the loop |
| **STT** | `GilgameshWind/X-ASR-zh-en` int8 | sherpa-onnx Zipformer, 160ms streaming, `146M` encoder int8 (was `566M`), zh+en 16k, CUDA |
| **LLM** | `Qwen/Qwen3.5-2B` Q4_K_M (default) | `unsloth/Qwen3.5-2B-GGUF` `1.3G`, llama-server `:11435` `-c 16384` `Qwen-Agent` `thinking:True`, `3` tools — switchable at runtime between `0.8B Q8_0` / `2B Q4_K_M` / `4B Q4_K_M` via the UI or `POST /api/model` |
| **Embedding** | `ibm-granite/granite-embedding-97m-multilingual` Q8_0 | `115M` GGUF `384d` `zh-TW+en`, `llama-server :11434` `CUDA` `8ms` batch rerank `0.34-0.65` |
| **TTS** | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` Q8_0 | `qwen-talker-0.6b-customvoice-Q8_0.gguf` `924M` + `codec.gguf` `343M`, `qwentts-cpp-python==0.3.1+cu124` GGML CUDA, true streaming, `24k` `~20ms` TTFA |
| **Search** | SearXNG official `:8888` + wttr.in + Bing scrape | `180` engines `all - china` (`baidu`/`sogou`/`360search` disabled), `general`/`news`, honest `no curated mocks`, `wttr.in` real weather `+` `searxng` `+` `DDG`/`lite`/`Bing` fallback |
| **Agent** | `qwen-agent` `Assistant` | `Qwen-Agent` `function_list=[web_search, get_weather, get_current_datetime]`, `thinking:True` filtered, `smolagents`/`PydanticAI` fallback harnesses |

All components are real — `mock=false` everywhere. Fallbacks: Kokoro-1.0 → MOSS-TTS-Nano (ONNX `CUDA` `NaN` on `transformers 5.15.1`).

**Every import ladder now ends in a mock adapter** (`stt/mock_streaming.py`, `tts/mock_streaming.py`). Before this, the last rung was another hard third-party import, so `app.py --mock` could not start on a machine without the model libraries — the ladder's `try/except ImportError` swallowed everything *except* its own final failure. `/health` reports `"mock"` when a rung is reached.

## Measured (RTX 3060, live stack, `backend/test_e2e_report.py`)

| Metric | Value | Note |
|---|---|---|
| STT content accuracy | **100 %**, CER 0.000 | `asr_example.wav`, 5/5 runs |
| First-audio latency, plain turn | **0.36–2.0 s** | |
| First-audio latency, **tool** turn | **1.0–3.6 s** | bounded by the search round-trip; was 6.7 s before tool answers streamed (see Notes) |
| LLM TTFT (p50) | **0.05–0.58 s** | was 1.38 s while the thinking pass sat in `message.content`; see Notes. The range is GPU contention with the TTS process on a 3060, not a code path |
| TTS TTFB (p50) | 0.34–0.88 s | was 1.34–1.93 s before the reasoning-format fix |
| E2E (p50 / max) | 1.3–2.7 s / 7–19 s | E2E = whole answer synthesized, so it scales with answer length; was 3.2 s / max 22.2 s at the start of this round |
| Barge-in correctness | **3/3** | text→text, voice→voice, voice→text; speech→first transcript 2.32–2.36 s. The voice→voice case only *needs* an interrupt if turn 1 is still in flight, and the check now says which case it measured (see Notes) |
| Tool-call selection | **83 % mean, 71–100 % across 6 passes** | two `--tool-repeats 3` runs at the default random seed: 71/86/71 and 71/100/100. Was 62 % (57–71 %) before the guards. The recurring miss is a news/weather query answered from memory |
| RSS | ~3.0 GB idle, ~3.3 GB peak | backend process |

**Caveats kept honest:** the 7-query set is small, and it is *this demo's* set — 5/7 is not a claim about general tool-calling quality either way. One pass is not a measurement: the same query set scored 57 %, 57 %, 71 % before the guards landed, and 71 %, 86 %, 71 % after them, so every number here is quoted as a spread — including the ones after the fixes, where the same build measured 71 %, 86 %, 71 %, then 71 %, 100 %, 100 %. Earlier revisions of this README quoted `14/14 100 %`.

**An A/B that changed a default (3 passes each, random seed):**

| Config | Tool accuracy | LLM TTFT p50 | E2E p50 | What the user heard |
|---|---|---|---|---|
| thinking in `content` (was) | 81 % (71/86/86) | 1384 ms | 3856 ms | answers, plus the scratchpad: *"…但根据规则，必须调用工具。所以步骤应该是先调用 get_current_datetime…"* |
| thinking off (`LLM_AGENT_THINKING=0`) | **71 %** (71/71/71) | 56 ms | 1638 ms | fast, clean prose — and invented facts: `get_weather` never called, `"東京天氣晴朗，平均溫度約25°C"`; a greeting answered with an invented date |
| thinking on + `--reasoning-format deepseek` (now) | 83 % (71/86/71, 71/100/100) | **53–576 ms** | **1306–2722 ms** | no scratchpad (thinking lands in `reasoning_content`, which was always ignored) |

Turning thinking off was the tempting fix — 7× faster first token — and it is *worse*, because its failures are fabrications rather than misses. The durable fix was server-side: llama-server's `--reasoning-format deepseek` (probed for, not assumed: `llm_manager._supports_llama_flag`) moves the deliberation out of `message.content` entirely.

**Two guards worth knowing about, because both were found by measuring, not by reading:**

- **A tool the model only *named*.** The answer came back as `現在的時間是 [get_current_datetime]。` — the model "called" the tool by printing its name and stopped, and the demo spoke a bracketed identifier. `agent/qwen_harness.py` detects the un-executed reference, runs the tool for real (the tool's own events make it visible in the UI), and either splices the value in place or hands the result back for one corrective turn; `pipeline._is_tool_artifact` separately refuses to speak such drafts.
- **A question the model answered from memory.** "Who is the president of France?" scored no tool call at all. `requires_fresh_facts()` recognises incumbency/winner questions from the *question alone* and invokes `web_search` before the model speaks (a real call, whose events reach the client), instead of hoping the router notices.
- **A fabricated clock.** On the voice path the model said `現在是下午 3 點 25 分` when it was 13:25, with no tool call at all. `fabricates_time_without_tool()` fires only when the *question* is about now **and** the *answer* states a time/date, forces the real `get_current_datetime` call, and rewrites the sentence deterministically (`format_clock_zh`) — asking the model to redo it came back with the same wrong time, because its own wrong claim was already in context. Measured effect of the three guards together: **62 % → 100 %** on the benchmark set.

## Pronunciation (why speech came out partly wrong, and what actually fixed it)

"Sometimes it mis-pronounces part of the sentence" has five candidate causes — the TTS model, the
streaming chunker, the text handed to it, the sample-rate path, and the *evaluator* (the ASR
mis-hearing). They need different fixes, so this is now measured instead of argued:

```bash
python3 backend/test_tts_asr_roundtrip.py --tts qwen3 --repeats 3 --modes full,stream   # CER per category
python3 backend/test_tts_asr_roundtrip.py --tts audio8 --raw-text                       # another engine / before-state
python3 backend/compare_tts_reports.py /tmp/a.json /tmp/b.json                          # item-by-item + noise band
```

The harness varies one factor at a time (one-shot vs three chunk sizes, repeats, two resamplers
24 k→16 k, the production paced-streaming ASR path vs a one-shot pass) and localizes every
mismatch against the audio's own chunk-boundary times. Eval-only dep: `zhconv` (see
`backend/requirements-eval.txt`) — X-ASR answers in simplified Chinese, so comparing it against a
ground-truth traditional reference scored a *correctly spoken* sentence at 42 % CER until both
sides were folded first.

**Calibration first: the ASR's own floor is 10.5 % CER** (paced path) / 5.3 % (one-shot) on
`asr_example.wav`, a human recording with a verified transcript. Every number below is TTS→audio→ASR,
so categories near the floor mean the TTS is fine. Read them as `cer − floor`.

38 texts × 3 repeats, streaming (`chunk_frames=12`), same engine, only the text front-end changed:

| Category | written text as-is | through `tts/spoken_text.py` | reading |
|---|---|---|---|
| **markdown** (`**…**`, `` `…` ``, links, tables) | **65.0 %** | **15.5 %** | real fix, far outside noise |
| mixed-script (`IBM 的 quantum …paper`) | 58.1 % | 55.1 % | the model's limit, not text shaping |
| numbers (`34°C`, `68%`, dates) | 20.0 % | 20.1 % | CER ignores numerals *by design* — see below |
| proper_zh_en | 8.9 % | 15.8 % | **identical input text** → this spread *is* the noise band |
| plain_zh / names / short / long / pure_en | 4.3 / 9.3 / 2.3 / 4.2 / 21.2 % | unchanged | clean text passes through untouched (asserted in tests) |
| corpus mean | 23.3 % | **17.1 %** | |

**Cause 1 — written language reaching an acoustic model.** Chat models emit markdown; nothing
stripped it, so `**颱風路徑北移**` went to the TTS verbatim and came back as `誰因此將就嗎？要回來…`
(65 % → 15.5 % on that category). `backend/tts/spoken_text.py` is now the one front-end, called
through `pipeline.prepare_tts_text()` from both TTS entry points (`synth_and_emit`, `app.tts_chunks`).
It only rewrites *notation* (markup, `°C`→`度`, `68%`→`百分之68`, `3/4`→`4分之3`, emoji/URLs) — never
content, and it returns the rule names it applied so a log line can explain why the spoken text
differs from the chat bubble (which still shows the original). `tests/test_spoken_text.py` includes a
guard rail that fails if a corpus sentence ever becomes a rule.

**Cause 2 — a unit read as a letter.** `最高溫度 34°C` was spoken `三十四度 C`, i.e. the letter "C"
apart, in **3/3 repeats**; normalized it is `三十四度` in 3/3. CER cannot see this (numerals are
stripped on both sides on purpose, so that "34 °C" said as "三十四度" is not counted as an error) —
which is exactly why the fix is verified from the transcripts, not from the metric.

**Cause 3 (investigated, then rejected): the streaming chunker.** Raw text showed a real chunk
penalty — stream@12 frames 36.7 % vs one-shot 22.9 % — and 47–57 % of mismatch positions landed
within 0.45 s of a chunk boundary. But after the text fix the penalty is gone (stream@12 24.5 % ≈
one-shot 24.6 %) while the boundary correlation barely moved (0.49 → 0.47). Conclusion: what looked
like the chunker cutting audio was mostly small chunks *plus markup* derailing the model; no
audio-cutting bug was confirmed, so no chunker change was made.

**Noise, stated as a number.** Categories whose text the front-end does not touch can only move by
noise, and one of them moved 8.9 → 15.8 % (n=4). Per-repeat corpus means spread 2.1 pp (raw) and
17.2 pp (normalized, driven by a single repeat-derailment item with CER > 200 %). So: trust
item-level deltas and per-repeat medians (`compare_tts_reports.py` prints both); treat any
category delta under ~7 pp at n=4 as a tie. `--repeats 3` is the minimum, not the ideal.

**Two engines evaluated and NOT adopted** (the pronunciation complaint made a model swap the
obvious hypothesis — it did not survive measurement):

- **MOSS-TTS-Realtime GGUF** (`BricksDisplay/MOSS-TTS-Realtime-GGUF`): the GGUFs are loadable in
  spirit — backbone is stock llama.cpp `qwen3`, audio side is `codec.cpp`'s `codec_lm` — but the
  *Realtime generation loop* (17 codebooks: `cb-0` text, `cb-1..16` RVQ) lives in `llama.rn`'s
  `rn-tts.cpp`, i.e. Node. `codec.cpp` has the MOSS codec and TTSD smoke tests, and a `tts-cli`,
  but no Realtime loop; there is no PyPI package. Adopting the GGUFs means writing that loop
  against a C API. The upstream PyTorch path does work from Python
  (`MossTTSRealtimeInference`, prefill/step, streaming via `session.push_text()/drain()`) but needs
  **11.8 GB of weights** (4.68 GB backbone + 7.10 GB MOSS-Audio-Tokenizer) and its own
  `transformers==5.0.0`/`torch==2.9.1` pins — this repo pins 5.15/2.11 for the STT+LLM, so it can
  only come in as a separate-venv sidecar, not an in-process adapter.
- **Audio8 TTS Preview 0.1B ONNX INT8** (`Audio8/audio8-TTS-0.1B-ONNX-INT8`, 0.45 GB, CPU-only,
  11 languages): adapter written and working end to end (`backend/tts/audio8_onnx_streaming.py`,
  `--tts audio8`), and *worse* on the axis that mattered — plain Chinese **29.1 % CER vs Qwen3-TTS'
  9.2 %**, long sentences 29.8 % vs 3.2 % — while also being far too slow: 30 ms/frame AR +
  22–35 ms/frame codec decode + **2.9 s prompt prefill per call** (the exported graph is
  one-token and replays the 110-frame reference voice every time), measured TTFA 5.2–6.8 s against
  Qwen3-TTS's 0.2–0.5 s. Upstream's `stream()` re-decodes `stream_context_frames + chunk_frames`
  per emitted chunk, which is what produced RTF 9–12 at `chunk_frames=10`; both facts are recorded
  in the adapter's docstring. Left available (`TTS_PREFER`/`AUDIO8_MODEL_DIR`), not made the default.

## Security & access control

| Control | Default | How to change |
|---|---|---|
| `VOICE_CHAT_TOKEN` (or `--token`) | unset = **open dev mode** | When set, `/api/*`, `/ws/chat`, `/search`, `/stats` require `X-Auth-Token`, `Authorization: Bearer`, or `?token=`. `/health` stays readable so UI polling and container healthchecks work; its usage history needs `?verbose=1` + token. |
| Model switching | **loopback-only** when no token is configured | Any token holder. It restarts `llama-server` (silencing every session) and can terminate the process holding the LLM port — no longer callable by whoever happens to reach the port. `_kill_unowned_listener` now refuses to kill anything that is not `llama-server`. |
| CORS | `ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`, **no credentials** | `ALLOWED_ORIGINS=<csv>`; `ALLOW_CREDENTIALS=1` only takes effect with an explicit origin list (wildcard + credentials is refused and logged). Previously the app sent `Access-Control-Allow-Origin: <any caller>` *with* `Allow-Credentials: true`. |
| `/search?format=html` | all output HTML-escaped, only `http(s)` becomes a link | — (was reflected XSS: the query and internet-sourced titles/URLs were interpolated raw) |
| WebSocket handshake | rejected (HTTP 403, before `accept()`) unless the `Origin` is same-origin, loopback, or listed in `ALLOWED_ORIGINS` | `WS_ALLOW_ANY_ORIGIN=1`. **CORS does not cover WebSockets** — browsers only check `fetch()` — so until this existed any page on the internet could open `ws://your-host/ws/chat` from a visitor's browser and get a live mic-to-speaker session with tool calls. Clients that send no `Origin` (native apps, `test_e2e_report.py`, `curl`) are unaffected |

If you publish this port (the README's own Tailscale Funnel note), set `VOICE_CHAT_TOKEN` first — the backend logs a warning at boot when bound to a non-loopback host without one.

## Configuration

| Env | Default | Used by |
|---|---|---|
| `VOICE_CHAT_TOKEN` | — | auth middleware (`--token` alias) |
| `ALLOWED_ORIGINS` / `ALLOW_CREDENTIALS` | dev origins / off | CORS |
| `SEARXNG_URL` | `http://localhost:8888` | `app.py` **and** `tools/web_search.py` (previously only `app.py`, so under Docker `/health` said SearXNG was fine while every search bypassed it) |
| `LLM_API_BASE`, `LLM_PORT`, `LLM_CTX`, `LLM_DEFAULT_MODEL_ID`, `LLM_PATH_*`, `LLAMA_SERVER_BIN` | see `llm_manager.py` | text-LLM subprocess + harnesses |
| `LLM_SEED` | `-1` (random) | `--seed` passed to llama-server; set it to make a benchmark run reproducible |
| `LLM_AGENT_TEMP`, `LLM_AGENT_TOP_P` | `0.7`, `0.9` | sampling for the **agent** turn (tool-call routing is temperature-sensitive — A/B it with `--tool-repeats`) |
| `LLM_AGENT_SEED` | unset | per-request seed via qwen-agent; `4711` reproduces a run exactly |
| `LLM_AGENT_THINKING` | `1` | the model's thinking pass on agent turns. `0` is ~7× faster to first token but fabricates instead of searching (measured above) |
| `LLM_AGENT_NO_TOOL_SMALLTALK` | `1` | adds "greetings get a conversational reply, not a tool call" to the agent system prompt — the 2 B router otherwise answered *"how are you?"* by calling `get_current_datetime` |
| `LLM_REASONING_FORMAT` | `deepseek` | llama-server `--reasoning-format`, so thinking goes to `reasoning_content` and can never be spoken. `none` = old behaviour (and the flag is skipped automatically on builds that lack it) |
| `WS_ALLOW_ANY_ORIGIN` | off | opt out of the WebSocket Origin gate |
| `VOICE_TZ` | `Asia/Taipei` | timezone the datetime tool and the clock repair speak |
| `EMBED_API_BASE`, `EMBED_MODEL` | `:11434` | semantic rerank |
| `TTS_MODEL_DIR` / `STT_MODEL_DIR` | `/tmp/qwen3_tts`, `/tmp/XASR/…` | model locations (Docker mounts) |


## Features

- **Full-duplex barge-in, either direction** — speaking over an in-progress reply cancels it immediately, whether that reply came from voice or a typed message, and vice versa. STT runs continuously in the background instead of blocking on the current turn, so a new utterance is recognized *while* the assistant is still talking. Every reply is tagged with a monotonic `turn_id`; the client drops any stray audio from a turn that lost a race with its own cancellation, and a shared lock serializes the (voice-triggered, text-triggered, and explicit button/WS) cancellation paths so they can't interleave.
- **Tool-aware** — `web_search` (SearXNG + wttr.in weather) and `get_current_datetime` (IANA tz) via native function calling
- **zh-TW by default, English mixed in** — LLM prompts (both harness system messages and a per-turn reinforcement hint, `llm/ling_streaming.py:LANG_HINT`) unconditionally default every reply to Traditional Chinese (Taiwan usage), regardless of the input's language — English is kept only for untranslatable terms/proper nouns. TTS segments text by script and passes the correct explicit language per segment (`chinese`/`english`) instead of a single whole-utterance `"auto"` guess, so a name or term quoted mid-sentence doesn't get the wrong pronunciation. Default voice, UI text, and quick-demo chips are all zh-TW; the frontend's display-layer OpenCC conversion uses the phrase-aware `s2twp` preset (Taiwan vocabulary, not just Traditional glyphs) as a display-only safety net.
- **Low latency** — Svelte 5 + AudioWorklet + binary WS, whole-sentence TTS flush, 0.6s pre-roll jitter buffer

Measured on RTX 3060 (see the Measured table above for the current numbers — the E2E figure here was first-audio, not whole-answer).

## Quick Start

```bash
# 1) Embedding (Granite-97M Q8, 115M, CUDA)
/home/user/llama.cpp/build/bin/llama-server \
  -m /tmp/granite-emb-gguf/granite-embedding-97M-multilingual-r2-Q8_0.gguf \
  --host 127.0.0.1 --port 11434 -c 8192 --alias granite-embedding --embedding --pooling mean --n-gpu-layers 99

# 2) LLM (Qwen3.5-2B Q4_K_M, 1.3G, thinking) — OPTIONAL: the backend (step 4) now
# spawns and owns this itself if nothing's listening on :11435 yet (see
# backend/llm_manager.py), or adopts it if you start it manually first, as below.
/home/user/llama.cpp/build/bin/llama-server \
  -m /tmp/llms/Qwen3.5-2B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 11435 -c 16384 --alias qwen3.5-2b --n-gpu-layers 99 --jinja

# 3) SearXNG official (full, all - china)
SEARXNG_SETTINGS_PATH=/tmp/searxng/settings.yml python3 -m searx.webapp  # :8888, formats [html, json]

# 4) Backend
cd backend && python3 app.py --port 8000  # http://127.0.0.1:8000/health

# 5) Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173
# prod: npm run build — backend serves frontend/dist at /
```

Required files (mount as volumes in Docker):
- LLM: `/tmp/llms/Qwen3.5-2B-Q4_K_M.gguf` (`1.3G`; smaller `Qwen3.5-0.8B` variants also work, `2B Q4` is the definitive default)
- Embedding: `/tmp/granite-emb-gguf/granite-embedding-97M-multilingual-r2-Q8_0.gguf` `115M`
- TTS: `/tmp/qwen3_tts/talker_cv_q8.gguf` `924M` + `codec.gguf` `343M`
- STT: `/tmp/XASR/deployment/models/chunk-160ms-model` `encoder-160ms.int8.onnx` `146M` (+ `decoder` `2.8M`, `joiner` `2.5M`)
- SearXNG: `/tmp/searxng/settings.yml` (`180` engines `all - china` `html+json`)

## Docker

```bash
docker compose up -d             # searxng + granite-embedding + llm + voice-chat (CUDA)
```

The image mounts model weights as volumes and talks to `llama-server` over the `llm`
service (`LLM_API_BASE` defaults to `http://llm:8080/v1`, which is a compose service name —
a bare `docker run` without that service now degrades with a logged reason instead of
answering 500).

**Building the light image (~300 MB, no model stack, boots in mock mode):**

```bash
docker build --target smoke --build-arg INSTALL_MODE=light -t voice-chat-smoke .
docker run --rm -p 8010:8000 voice-chat-smoke            # /health reports all-mock
```

This is the same `Dockerfile`, same base image, same apt layer, same COPYs — only
`requirements-light.txt` and the boot flags differ. It is what CI builds (`.github/workflows/ci.yml`,
`container` job) because it needs no GPU and no weights, and it is the check that caught
real breakage in this repo: a `COPY` with two sources and a file destination (silently
shipped one of the two files), a hard `import torch` on the boot path, adapters that fail
at *construction* rather than import, and the `/api/chat` 500 above. Verified here with
podman 5.7 on `python:3.13-slim`: build 68 s → 1.14 GB image, boot 12 s, `/health` →
`{"stt":"mock","llm":"mock","tts":"mock"}` at 76 MB RSS, `/api/chat` → 200, 0 raw
`<script>` tags from the escaped search page.

**Still not covered by any of it:** the `INSTALL_MODE=full` image (torch cu128 + five model
stacks), CUDA driver interaction, and the model stack inside a container — those need a GPU
host. `.dockerignore` keeps the build context at ~600 MB instead of shipping weights,
`node_modules` and `.git` into every build.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | models, RSS, mock flag, SearXNG status, `auth` mode. Public (lite). `?verbose=1` adds VMS + the per-utterance latency history and **requires a token** when one is configured |
| `WS /ws/chat?session_id=…[&token=…]` | Binary PCM in (`0x01` audio / `0x02` flush / `0x03` barge-in) or JSON (`start`/`audio_chunk`/`stop`/`barge_in`/`text_input`) → `stt_partial` / `stt_final` / `llm_token` / `llm_reset` / `tool_call` / `tool_result` / `tts_start` / `tts_chunk` / `tts_end` / `latency` / `barge_in`, every assistant event tagged with `turn_id`. Omitting `session_id` now mints a unique one (previously every such client shared one slot and could cancel each other's replies) |
| `POST /api/chat` | JSON chat (non-stream); optional `voice` is validated and applied **per call** (`audio_b64` or `text`, `tools` toggle) |
| `GET /api/search?q=…` | Honest SearXNG search (no curated mocks) |
| `GET /stats` | Counters & latency percentiles (bounded history) |
| `GET /api/model` | Current + available text-LLM models (id, label, loaded, file present) |
| `POST /api/model {"model_id":"…"}` | Switch the loaded text-LLM model — stops and restarts the llama-server (~1-2s observed), also selectable from the UI's Model card. **Loopback-only unless a token is set** |

`llm_reset` is new: the agent can stream answer text and then have to withdraw it (an XML tool call appeared mid-stream, or a tool step followed). The event tells the client to discard the partial bubble — the text was never spoken.

## Validating a change

```bash
# 1) Pure-logic regression tests — no GPU, no models, no network (also run in CI)
python3 -m unittest discover -s backend/tests -v      # 124 tests

# 2) Does this interpreter have what the default stack needs? (non-zero exit = missing)
python3 backend/check_env.py

# 3) Against a LIVE deployment (real models): STT accuracy/CER, tool-call selection,
#    latency percentiles, peak RSS, and all three barge-in directions
python3 backend/test_e2e_report.py --server http://127.0.0.1:8000

# 3b) Pronunciation: synthesize -> re-transcribe -> CER per category, with the ASR's own
#     error floor, three ASR/resample cross-checks and chunk-boundary correlation.
#     Needs GPU models + `pip install --target $EVAL_LIB_DIR -r backend/requirements-eval.txt`
python3 backend/test_tts_asr_roundtrip.py --tts qwen3 --repeats 3 --modes full,stream
python3 backend/compare_tts_reports.py before.json after.json    # item-by-item + noise band

# 4) Will the pins actually install in the Docker image? (wheels for cp313/x86_64,
#    including the --find-links CUDA TTS wheel; no build needed to find out)
python3 backend/check_requirements.py            # --arch aarch64 / --python-version 312 / --offline

# 5) Lint — the enforced gate is E4, E7, E9, F, B (zero findings, including the
#    271 pre-existing style sites that were swept: E701/E702/E401 by autopep8,
#    E722/E741/E731/E402 by hand; see ruff.toml)
ruff check backend --config ruff.toml

# 6) The container path. (a) does the whole pin set resolve on the image's interpreter?
sudo podman run --rm --network=host -v $PWD/backend:/wb:ro -w /wb python:3.13-slim \
  pip install --dry-run --no-deps --ignore-installed --target /tmp/x -r requirements.txt
# 7) (b) build the real Dockerfile, light variant, and boot it (this is the CI job)
docker build --target smoke --build-arg INSTALL_MODE=light -t voice-chat-smoke .
docker run -d --name vc -p 8010:8000 voice-chat-smoke
curl -s localhost:8010/health | grep -o '"stt":"mock"'
curl -s -X POST localhost:8010/api/chat -H 'content-type: application/json' -d '{"text":"hi"}'
docker rm -f vc
```

Both were run on 2026-08-31 (podman 5.7, `python:3.13-slim`): (6) resolves all 31 pins
("Would install …"), including the `+cu124` TTS wheel from the `--find-links` index; (7)
builds in ~70 s to a 1.14 GB image that serves `/health` → `{"stt":"mock","llm":"mock","tts":"mock"}`
at 76 MB RSS, answers `/api/chat` with 200 + audio, and escapes the search page. Building
the light image found four real bugs in one sitting, which is why it is now a CI job.

What is still **not** covered: `INSTALL_MODE=full` (torch cu128 + five model stacks), CUDA
driver interaction inside a container, and running the real models in there — all need a
GPU host. `test_requirements_light_is_a_subset_of_requirements` keeps
`requirements-light.txt` from drifting into a dependency set production does not use, and
`TestShipFilesAreValid` parses `docker-compose.yml` and every workflow file (the shipped
`ci.yml` had an unquoted `: ` in a step name, i.e. GitHub would never have run it) and
asserts the Dockerfile's default `LLM_API_BASE` names a service that exists in compose.

Latency/accuracy runs are only comparable with a fixed seed, because tool routing flips
between runs otherwise:

```bash
LLM_AGENT_SEED=4711 python3 backend/app.py --port 8000 &
python3 backend/test_e2e_report.py --tool-repeats 3        # reports the spread, not one coin flip
```

`backend/tests/` exists because the three bugs that actually shipped this month (CJK relevance scoring, truncated tool-call JSON, wrong-language TTS segments) were all in pure functions that nothing tested.

## Notes

- **The model was narrating its own planning to the user.** With thinking on, llama-server's
   default `--reasoning-format auto` leaves the deliberation in `message.content`, and the
   agent path speaks `content`, so "What time is it right now?" answered *"…但根据规则，必須調用工具。
   所以步驟應該是先调用 get_current_datetime…"* — in Simplified Chinese, about the assistant's own
   procedure. Fixed at the source (`llm_manager._reasoning_args`, `--reasoning-format deepseek`,
   probed for via `--help` because an unsupported flag stops llama-server from starting at all),
   with `_strip_thinking()` as a backstop on the final answer and
   `TestThinkingTextNeverGetsSpoken` holding the line. Streaming deltas cannot be filtered this way
   (you only know a block was thinking after it ends, by which point it has been spoken) — which is
   exactly why the server-side separation is the fix and not a text scrubber.
- **An unreachable `LLM_API_BASE` is now a degraded turn, not a 500.** The adapter constructor
   deliberately stays optimistic ("will retry per request") so a llama-server that comes up later is
   picked up, but the retry raised `httpx.ConnectError` straight out of the endpoint — inside the
   container, where the image default names a compose service, every `/api/chat` was an HTML 500 with
   a traceback. Now: `LingStreaming._reachable()` pre-flights (cached 1 s, per instance so a model
   switch reprobes) and degrades **before** the harness/legacy branch (it was first placed after that
   early `return`, so the light install — the one most likely to have a dead endpoint — still raised);
   `--mock` degrades to the canned answers it promised, a real deployment hears which endpoint is
   missing, and `/api/chat` returns a JSON 503 for anything that still escapes.
- **The voice→voice barge-in check now asserts its own precondition.** It streamed an interruption
   after the first `tts_chunk` and required a `barge_in` event — fine at 3.9 s turns, wrong at 1.3 s:
   STT needs ~2.3 s of paced audio before a barge-in can fire, so by then turn 1 had legitimately
   finished and a 3× latency win scored as FAIL. The check starts at the first transcript instead, and
   records `first_turn_in_flight` / `first_turn_done_before_barge` so the report says which situation it
   measured instead of silently asserting one. Real interruption is still exercised by voice→text (2.3 s
   reaction, zero stray chunks).
- **Small talk stopped calling tools.** `Qwen3.5-2B` answered "Hello, how are you today?" by calling
   `get_current_datetime` and replying "今天好，星期二上午 2 點 52 分，天氣晴朗。" — 1/3 correct on a
   greeting. One sentence in the agent system prompt (`LLM_AGENT_NO_TOOL_SMALLTALK=1`) fixed it.
- **Adapter interfaces unified, leftovers removed.** `tts.voices` was a bound method on four adapters
   and a list on two others (and the `KeyError` paths called `self.voices()`, so a list-style adapter
   would raise `TypeError` while trying to report an unknown voice); the dead `direct_tts(history=…)`
   parameter, `hf_official`'s empty `stt_partial` filler (the UI paints partials as live captions), and
   a class docstring teaching callers to use the removed `stream_chat` are gone. `TestTtsAdapterVoiceInterface`
   and the dead-code test keep them gone.

- **Tool answers now stream** — `qwen_harness` reads qwen-agent's partial-yield generator
  and emits `llm_delta` per step, so a `web_search` turn produces first audio in ~1 s
  instead of waiting for the whole agent loop (was 6.7 s). If a step turns out to be a
  tool call rather than the answer, the harness emits a reset and the consumer drops
  what it buffered (`llm_reset`, handled in `run_turn`, `direct_tts`, and the UI).
- **Multi-turn reaches the agent as real messages** — previously a digest of
  `role: content[:120]` from the last 4 turns, which is why referential follow-ups
  ("BBC headlines", "and tomorrow?") lost their referent.
- **STT decodes off the event loop** — each 20 ms frame used to run sherpa's native
  `decode_stream` inline, stalling the loop (LLM streaming, WS sends, every other
  session) once per frame. Each burst now runs in `asyncio.to_thread`.
- **One decoder stream per connection** — `recognizer` is shared (safe), `stream` was a
  single process-wide object, so two tabs interleaved their transcripts.
- **Per-call TTS voice** — `synthesize_streaming(text, voice=…)`; `set_voice()` still
  exists but mutates a process-wide default on an instance shared by all sessions, and a
  mistyped voice name used to be swallowed by `except KeyError: pass`.
- **Nothing is keyed to a demo question any more** — `_clean_leakage` used to special-case
  the string `President of France`, and both speech paths suppressed a question echo with
  `startswith("who is the president")`. The echo rule is now a content comparison
  (`is_echo_of_prompt`, shared by the voice and text paths so they cannot drift), and
  `test_live_paths_hold_no_benchmark_query_literals` fails the suite if a benchmark
  question reappears as a string constant in a live module.
- **Spoken text is de-markdowned** — a search-grounded answer arrived as
  `法國總統是**愛德華·馬克龍**`; `ling_streaming._speakable()` collapses `**b**`/`__b__` and
  `[text](url)`, and `*`/`` ` `` are dropped from streamed deltas one character at a time.
- **`--mock` really boots anywhere now** — `import torch` in `main()` was only used for
  `cuda.is_available()` yet made an 800 MB+ dependency a boot requirement, and the
  constructor (not the import ladder) was where adapters actually failed. Both fixed;
  `test_mock_boot_needs_no_ml_libraries` re-runs it with every ML import blocked.
- **`HF_OFFICIAL` pipeline reached parity** — `pipeline/hf_official.py` used to accept
  `barge_in_event`/`barge_in_lock`/`on_new_voice_turn` and ignore all three, iterate STT
  inline (nothing recognised while a reply played), emit no `turn_id`, and compute
  `stt_ms`/`llm_start`/`e2e_start` and throw them away. It now runs the same
  pump + per-turn-task + `turn_id` + barge-in contract as the default pipeline
  (`test_review_fixes.TestHFOfficialPipelineParity`, 5 tests with fake STT/LLM/TTS).
- **Bounded state** — latency history is a 1000-entry ring, the search cache is an
  LRU-capped TTL map. Both were unbounded before.

- **Switchable LLM size** — `backend/llm_manager.py` owns the llama-server subprocess for the text LLM (`qwen3.5-0.8b-q8` / `qwen3.5-2b-q4` / `qwen3.5-4b-q4`, see `MODEL_REGISTRY`) and can stop+restart it with a different GGUF on request. Only one size is loaded at a time — VRAM headroom on a 12GB card is tight enough with embedding+TTS+STT already resident that keeping all three warm simultaneously risked OOMing something else. On startup it adopts an already-running server on the configured port (this project's traditional manual-start workflow) rather than duplicating it; the first switch afterwards takes real ownership. **Bare-metal only** — `docker-compose.yml` runs the LLM in its own fixed sibling container (`llm:8080`) with none of `llm_manager`'s expected local binary/GGUF paths present in the `voice-chat` container, so a switch attempt there fails loudly (a clear error, not silent corruption) rather than working; Dockerizing the switcher would need the llama-server binary and all three GGUFs bundled into (or reachable from) that same container.
- **Barge-in architecture** — `stream_chat_interleaved` (`backend/pipeline/speech_to_speech.py`) runs STT via a background pump task instead of blocking on the current turn's LLM/TTS, so new speech is recognized while a reply is still playing; a fresh utterance cancels the in-flight turn and starts a new one. A shared `asyncio.Lock` serializes that voice-triggered cancellation against the WS-level `do_barge_in` (button, `barge_in` message, or a new `text_input`) so the two paths can't race each other's `set()`/`clear()` on the shared cancellation event, and an `on_new_voice_turn` callback lets a fresh voice utterance also supersede an in-flight `text_input` reply (not just the reverse). Every response event carries a monotonic `turn_id`; the frontend drops any event from a turn lower than the highest it has seen (or explicitly blacklisted as just-interrupted), which correctly lets the *start* of a new reply through instead of dropping it under a fixed timing window.
- **Honest search** — No hard-coded mocks. `SearXNG official` `30` results `~1.2s` `brave`/`google cse` `all - china`; DIY minimal (`DDGS`+`lite` only `3` engines) was root cause of `8.4s` `mock` `example.com`. `wttr.in` is real weather API (`Paris` `16-22°C`). `Bing` scrape fallback when `SearXNG` `rel<0.34`.
- **Thinking without leakage** — `Qwen-Agent` `enable_thinking:True`, `reasoning_content` filtered from both TTS and chat history (was leaking a reasoning-loop on smaller quantizations).
- **Clean UI** — `Svelte 5` `marked`+`DOMPurify` markdown (`**bold**` `ol`/`ul`), minimal single-column mobile, dark/light theme, pulse-free audio `0.6s` pre-roll.

## License

Code: Apache-2.0. Models: Qwen3.5-2B (Apache-2.0), Granite-embedding-97M (Apache-2.0), Qwen3-TTS (Apache-2.0), X-ASR-int8 (Apache-2.0), SearXNG official (AGPL-3.0).
