# Voice Chat — Streaming Speech-to-Speech with Tools

**X-ASR-int8 → Qwen3.5-2B UD-Q8_K_XL+MTP (Qwen-Agent) → Qwen3-TTS.** Real models, no mocks, low latency.

A full-duplex voice chat demo: speak, interrupt it mid-sentence by voice *or* text, search the
web, and hear the answer. **Traditional Chinese (Taiwan) by default** — input and output are
zh-TW regardless of the question's language, with English kept only for proper nouns and terms
that don't translate well.

```
Mic 16k ─► endpoint detect ─► STT (X-ASR int8, 160ms) ─► LLM (Qwen3.5, Qwen-Agent) ─► TTS (Qwen3-TTS) ─► Speaker
                  ▲                                       │  web_search · get_weather · get_current_datetime
                  └───────────────────────────────────────┴─► SearXNG :8888 + wttr.in
```

## Stack

| Layer | Model / Service | Notes |
|---|---|---|
| **Turn-taking** | sherpa-onnx rule-based endpointing | trailing-silence rules drive `stt_final`; no separate VAD |
| **STT** | `GilgameshWind/X-ASR-zh-en` int8 | Zipformer, 160 ms streaming, 146 M encoder, zh+en 16 k, CUDA |
| **LLM** | `Qwen/Qwen3.5-2B` UD-Q8_K_XL + MTP (default) | llama-server `:11435`, `-c 16384`, thinking on, 3 tools. Switchable at runtime — see below |
| **Embedding** | `granite-embedding-97m-multilingual` Q8_0 | 384 d, zh-TW+en, `:11434`, semantic rerank for paraphrases |
| **TTS** | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` Q8_0 | GGML CUDA via `qwentts-cpp-python`, true streaming, 24 k, ~20 ms TTFA |
| **Search** | SearXNG `:8888` + wttr.in + Bing scrape | 180 engines, real results only — no curated mocks |
| **Agent** | `qwen-agent` `Assistant` | `smolagents` / `PydanticAI` fallback harnesses |

Every import ladder ends in a mock adapter, so `app.py --mock` starts without any model
libraries installed; `/health` reports `mock` when a rung is reached.

### Switchable models

`qwen3.5-2b-q4` (default), `qwen3.5-4b-q4` and `bonsai-8b-ternary`, selectable from the UI's
Model card or `POST /api/model`. Only one is loaded at a time — VRAM on a 12 GB card is tight
with embedding + TTS + STT resident. **Bare-metal only**: the compose setup runs the LLM in a
fixed sibling container.

**Bonsai 8B is ternary (1.58-bit)** — 8.2 B parameters in a 2.18 GB file. Measured on the same
zh corpus as the entries above it scores PPL 16.518, against the 2B Q8's 16.738 and the 4B's
13.027, at 118 tok/s and 3/4 on the UI prompts with all four tools routed correctly. So a
ternary 8B lands roughly where a 2 B Q8 does: the compression is real, and so is the quality
cost per parameter. The file size also oversells the footprint — 2.18 GB of weights still took
7.1 GB of VRAM at 16 k context.

It needs Prism's llama.cpp fork, which is why registry entries may carry a `"bin"` of their own:
their `Q2_0` reuses upstream's `GGML_TYPE_Q2_0` type id with a different block layout, so
mainline refuses the file outright. Build the `prism` branch with `-DGGML_CUDA=ON` (their README
says CPU/Metal only — out of date, the CUDA kernels including `mmvq` are there) and point
`LLAMA_SERVER_BIN_PRISM` at it. Set `TMPDIR` to a real disk first: nvcc's intermediates hit the
`/tmp` quota otherwise. Note the HF repo also ships a plain `*-Q2_0.gguf` in their *legacy*
group-128 layout that fails even on the fork; `PQ2_0` is the current one.

**Two models that were evaluated and dropped**, so the same ground isn't retrodden:

*Ling 3.0 tiny* (`bailingmoe3` MoE, 128 experts / 8 active) works — it needed its own tool-call
dialect, since Qwen-Agent expects a JSON call body while Ling's chat template asks for
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>`, and with that adapter it routed
all three tools correctly. At IQ4_XS it scored the same 3/4 on the four UI prompts as 2B. It was
still dropped: slower (148 tok/s vs 172), much heavier on VRAM (6.9 GB vs ~3.5), and it drifted
into Simplified Chinese more than Qwen does (`湿度` for 濕度), which is the one thing this demo
is specifically not supposed to do. Its adapter and tests were removed with it — recoverable
from git if it's ever worth revisiting. One finding outlived it: **IQ4_XS beat MXFP4** on the
same weights (148 vs 108 tok/s, 6.9 vs 7.9 GB VRAM, identical score), despite MXFP4 being the
format designed for MoE experts. Measure the quant; don't reason about it.

*0.8B.* Decode runs 236 tok/s at 0.8B against 172 tok/s at 2B — 0.97 s vs 1.31 s for a
220-token answer. That 0.34 s falls inside a
turn where the user already waits 1.0–3.6 s for first audio, which search and TTS dominate, so
nobody can hear it. What it costs is reliability: every 0.8B build tested narrated its own
planning aloud, echoed the agent framework's tool template, quoted the prompt back, or repeated
itself — each failure needing another filter. Unquantized f16 scored the same as q8, so this is
capacity rather than quantization, and no amount of filtering fixes it.

## Quick Start

```bash
# 1) Embedding
llama-server -m granite-embedding-97M-multilingual-r2-Q8_0.gguf \
  --host 127.0.0.1 --port 11434 -c 8192 --alias granite-embedding --embedding --pooling mean --n-gpu-layers 99

# 2) LLM — optional: the backend spawns and owns this itself if :11435 is free,
#    or adopts it if you start it first (backend/llm_manager.py)
llama-server -m /home/user/llms/mtp/Qwen3.5-2B-UD-Q8_K_XL.gguf \
  --host 127.0.0.1 --port 11435 -c 16384 --alias qwen3.5-2b --n-gpu-layers 99 --jinja \
  --reasoning-format deepseek --spec-type draft-mtp --spec-draft-n-max 3

# 3) Search
SEARXNG_SETTINGS_PATH=/tmp/searxng/settings.yml python3 -m searx.webapp   # :8888

# 4) Backend
cd backend && python3 app.py --port 8000        # http://127.0.0.1:8000/health

# 5) Frontend
cd frontend && npm install && npm run dev       # http://localhost:5173
# prod: npm run build — the backend serves frontend/dist at /
```

Model files (mount as volumes under Docker):

- LLM `/home/user/llms/mtp/Qwen3.5-2B-UD-Q8_K_XL.gguf` · Embedding `/tmp/granite-emb-gguf/…Q8_0.gguf`
- TTS `/tmp/qwen3_tts/talker_cv_q8.gguf` + `codec.gguf` · STT `/tmp/XASR/deployment/models/chunk-160ms-model/`
- SearXNG `/tmp/searxng/settings.yml`

> **`/tmp` is tmpfs on this machine — it is RAM, and it enforces a quota.** The paths above are
> the dev defaults, but a multi-GB write there fails partway with `Disk quota exceeded` even
> though `df` shows free space. For anything new, put weights on a real disk and point the
> matching `LLM_PATH_*` at them.

```bash
docker compose up -d --build     # UI at http://localhost:8000
```

## Features

- **Full-duplex barge-in, either direction.** Speaking over a reply cancels it, whether that
  reply came from voice or text, and vice versa. STT runs on a background pump rather than
  blocking on the current turn, so new speech is recognized *while* the assistant is talking.
  Every reply carries a monotonic `turn_id`; the client drops events from a superseded turn, and
  a shared lock serializes the voice, text and explicit cancellation paths.
- **Tool-aware.** `web_search`, `get_weather` and `get_current_datetime` via native function
  calling, with pre-flight routing for questions whose form requires a lookup (below).
- **zh-TW first.** Default voice, UI text and demo prompts are all Traditional Chinese; the
  display layer uses OpenCC's phrase-aware `s2twp` as a safety net.
- **Thinking shown, never spoken.** Deliberation streams to a collapsible panel and is kept out
  of both the audio and the chat transcript.
- **Low latency.** Svelte 5 + AudioWorklet + binary WS, whole-sentence TTS flush, 0.6 s pre-roll
  jitter buffer.

## Measured (RTX 3060, live stack)

| Metric | Value |
|---|---|
| STT accuracy | 100 %, CER 0.000 (`asr_example.wav`, 5 runs) |
| First audio — plain turn | 0.4–2.0 s |
| First audio — tool turn | 1.5–7.0 s (dominated by the search round-trip, not decode) |
| LLM TTFT (p50) | 0.05–0.58 s |
| TTS TTFB (p50) | 0.34–0.88 s |
| E2E (p50 / max) | 1.3–2.7 s / 7–19 s (whole answer synthesized, so it scales with length) |
| Barge-in | 3/3 — text→text, voice→voice, voice→text |
| Tool-call selection | 83 % mean, 71–100 % over 6 passes |
| RSS | ~3.0 GB idle, ~3.3 GB peak |

Tool routing is temperature-sensitive, so a single pass is not a measurement — every number
above is a spread. Fix `LLM_AGENT_SEED` to compare builds.

**Thinking on vs. off** (3 passes each) — turning it off is 7× faster to first token and
*worse*, because its failures are fabrications rather than misses:

| Config | Tool accuracy | TTFT p50 | What the user heard |
|---|---|---|---|
| thinking in `content` | 81 % | 1384 ms | answers, plus the scratchpad read aloud |
| thinking off | 71 % | 56 ms | fast, clean, and invented: weather never looked up |
| thinking + `--reasoning-format deepseek` | **83 %** | **53–576 ms** | answers only |

**Choosing the quantization.** Ranked by perplexity, not by the UI matrix (see below for why):

Perplexity over 194 kB of non-repeating zh Wikipedia prose — a deterministic measure of how far
each quantization moved the model, run with `llama-perplexity -c 2048`:

| Build | 2B PPL | 4B PPL | 2B tok/s |
|---|---|---|---|
| **UD-Q8_K_XL + MTP** (current) | **16.738** | **13.027** | 122 |
| UD-Q4_K_XL | 17.080 | 13.173 | 168 |
| Q4_K_M | 17.091 | — | 172 |
| IQ4_XS | — | — | 175 |

Q8 is the most faithful at both sizes (~2 % lower), and UD-Q4_K_XL edges plain Q4_K_M — which
is what Unsloth Dynamic claims: spend the extra bits on the quantization-sensitive layers
rather than lifting every layer uniformly. MTP recovers most of what Q8 costs in speed
(100 → 122 tok/s on 2B) without touching the precision the quantization was chosen for.

**The four-prompt matrix cannot rank these builds, and earlier revisions of this file wrongly
treated it as if it could.** Two runs of the *same weights* differing only by MTP — a
throughput feature that cannot change what the model writes — scored 6/8 and 4/8. Across four
near-equivalent configurations the scores were 4, 5, 6 and 4 out of 8. At n=8 and temperature
0.7 that spread is the noise floor. It is a smoke test for "does a turn work end to end", not
a quality metric. Use perplexity to compare weights, and fix `LLM_AGENT_SEED` with several
passes if you want the matrix to mean anything.

**MTP.** The registered weights carry NextN layers, so the model drafts its own next tokens and
the target model verifies them; accepted drafts are the tokens that would have been generated
anyway. Measured +20 % on 4B with byte-identical greedy output at temperature 0 — over two
prompts, so treat "lossless" as well-founded in theory and lightly checked in practice.
`LLM_MTP=0` disables it; `--spec-type` is probed, so a build without it still starts.

If you want a turn to feel faster, the thinking budget and the search round-trip are where the
seconds are — not the model size. First audio is 1.0–3.6 s on a tool turn, of which decode is a
fraction.

## Design notes

Things worth knowing before changing this code — each exists because the obvious version
misbehaved in a way that reached the user.

**Nothing the machine says to itself is an answer.** Reasoning reaches the client as its own
`reasoning` event and never as speech. Three other kinds of text look like an answer and are
not: our own system prompt replayed back (matched by shingle overlap against the prompt text,
so truncated or paraphrased replays are caught too), qwen-agent's Simplified-Chinese tool
narration (`工具"…"被调用时使用了以下参数：…`), and the model's own checklist scaffolding
(`Evaluate the Input:`, `User Question: "…"`). A standalone sentence-length all-English span is
also treated as non-answer, since answers here are zh-TW with English only *inside* a Chinese
sentence. Detection is deliberately shape-based rather than a list of observed phrases.

Because a single token is too small a unit to classify, a sentence recognized as reasoning only
after it has streamed is retracted from the transcript as well as withheld from speech
(`llm.ling_streaming.retract_span`). Both paths accumulate `text_so_far` locally — adopting the
harness's cumulative copy would restore what was just removed. If nothing survives the filter,
the turn falls back to a spoken "no answer" line rather than showing scaffolding in silence.

**The language instruction is stated once, in the system prompt**, and never appended to the
user's turn. Glued onto the transcript it became part of what the model was asked: it was copied
verbatim into `web_search` queries and read back aloud as the answer.

**Some questions are routed by their form, not the model's judgement.** The current date and a
current office holder are no more present in the weights than tomorrow's weather is, so
`required_tool_for_request()` invokes the tool before the model speaks. The Chinese patterns
match ordinary word order (`總統是誰`, 誰 after the office noun) as well as verb-first `誰是` —
matching only the latter meant this demo's own `法國現在的總統是誰？` chip was answered from
memory. Lookup intent is tested before the clock so `現在` doesn't turn that into a time query.
`_preflight_enabled()` turns the routing off to measure the bare model.

Three repair guards catch answers that assert things nothing verified: a tool the model only
*named* (`現在的時間是 [get_current_datetime]。`), a fabricated clock, and a fabricated forecast.
Each runs the real tool — its own events, so the repair is visible in the UI — and either
splices the value in or takes one corrective turn. A pre-flighted tool counts as a tool that
ran; treating it otherwise made the guards call everything a second time.

**Nothing is said twice.** The harness streams its answer deltas live, then reconciles them
against the authoritative final text and speaks only the remainder — a common-prefix
comparison. It was comparing the raw streamed text against a `final_text` that had already
been whitespace-collapsed, so an answer opening with a newline (most of them) broke the
prefix at character 0 and the "remainder" became the entire answer. Every streamed reply was
emitted twice: once with its line breaks, once flattened onto one line. This survived a long
time because the audio was fine — `SpokenGuard` refuses to speak a repeat — so only the
transcript showed it, and it was read as small-model repetition. It is on every model and
every quantization because it is arithmetic. Both sides are normalized before comparing now
(`tests/test_stream_replay.py`).

**Markdown is not speech.** Chat models emit `**bold**`, and handing that to an acoustic model
produced 65 % CER on those sentences. `tts/spoken_text.py` is the single front-end for both TTS
entry points; it rewrites notation only (markup, `°C`→`度`, `68%`→`百分之68`, emoji, URLs), never
content, and returns the rules it applied so a log line can explain why the spoken text differs
from the chat bubble. Text is then segmented by script and synthesized with an explicit
per-segment language, so a term quoted mid-sentence isn't pronounced as Chinese.

**Search is honest.** No curated results. Chinese queries are tokenized with jieba before
relevance scoring — a regex treats an unsegmented phrase as one token, which scored every
Chinese query at zero and silently disabled the fallback chain. Fallback backends race
concurrently instead of each waiting out its own timeout.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `VOICE_CHAT_TOKEN` | — | auth token (`--token` alias) |
| `ALLOWED_ORIGINS` / `ALLOW_CREDENTIALS` | dev origins / off | CORS |
| `WS_ALLOW_ANY_ORIGIN` | off | opt out of the WebSocket Origin gate |
| `SEARXNG_URL` | `http://localhost:8888` | search backend |
| `LLM_API_BASE`, `LLM_PORT`, `LLM_CTX`, `LLM_DEFAULT_MODEL_ID`, `LLM_PATH_*` | see `llm_manager.py` | LLM subprocess |
| `LLM_MTP` / `LLM_MTP_DRAFT_N` | `1` / `3` | self-speculative decoding on MTP weights; `0` disables |
| `LLM_SEED` / `LLM_AGENT_SEED` | random / unset | reproducible runs |
| `LLM_AGENT_TEMP`, `LLM_AGENT_TOP_P` | `0.7`, `0.9` | agent-turn sampling |
| `LLM_AGENT_THINKING` | `1` | thinking pass on agent turns |
| `LLM_AGENT_NO_TOOL_SMALLTALK` | `1` | greetings get a reply, not a tool call |
| `LLM_REASONING_FORMAT` | `deepseek` | keeps thinking out of `message.content` |
| `VOICE_TZ` | `Asia/Taipei` | timezone spoken by the datetime tool |
| `EMBED_API_BASE`, `EMBED_MODEL` | `:11434` | semantic rerank |
| `TTS_MODEL_DIR`, `STT_MODEL_DIR` | `/tmp/…` | model locations (Docker mounts) |

## API

| Endpoint | Description |
|---|---|
| `GET /health` | models, RSS, mock flag, SearXNG status, auth mode. Public; `?verbose=1` adds latency history and needs a token |
| `WS /ws/chat?session_id=…` | binary PCM (`0x01` audio / `0x02` flush / `0x03` barge-in) or JSON (`start`/`audio_chunk`/`stop`/`barge_in`/`text_input`) → `stt_partial`/`stt_final`/`llm_token`/`llm_reset`/`reasoning`/`tool_call`/`tool_result`/`tts_start`/`tts_chunk`/`tts_end`/`latency`/`barge_in`, each tagged with `turn_id` |
| `POST /api/chat` | JSON chat (non-streaming) |
| `GET /api/search?q=…` | SearXNG search |
| `GET /stats` | counters and latency percentiles |
| `GET` / `POST /api/model` | list / switch the loaded LLM (~1–2 s). Loopback-only unless a token is set |

`llm_reset` tells the client to discard a partial bubble: the agent can stream text and then
withdraw it when a tool call appears mid-stream. Nothing withdrawn was ever spoken.

## Security

| Control | Default |
|---|---|
| `VOICE_CHAT_TOKEN` | unset = open dev mode. When set, `/api/*`, `/ws/chat`, `/search`, `/stats` require `X-Auth-Token`, `Authorization: Bearer`, or `?token=` |
| Model switching | loopback-only when no token is configured — it restarts llama-server and silences every session |
| CORS | explicit dev-origin list, no credentials. Wildcard + credentials is refused and logged |
| WebSocket handshake | rejected with 403 before `accept()` unless the Origin is same-origin, loopback, or allow-listed. **CORS does not cover WebSockets** — without this, any page could open a live mic session from a visitor's browser |
| `/search?format=html` | all output escaped; only `http(s)` becomes a link |

If you publish this port, set `VOICE_CHAT_TOKEN` first — the backend warns at boot when bound to
a non-loopback host without one.

## Validating a change

```bash
python3 -m unittest discover -s backend/tests -v     # pure logic, no GPU/network (CI)
python3 backend/check_env.py                         # is this interpreter complete?
python3 backend/check_requirements.py                # will the pins install in the image?
ruff check backend --config ruff.toml                # gate: E4, E7, E9, F, B

# against a LIVE deployment (real models)
python3 backend/test_e2e_report.py --server http://127.0.0.1:8000
python3 backend/test_tts_asr_roundtrip.py --tts qwen3 --repeats 3 --modes full,stream
python3 backend/compare_tts_reports.py before.json after.json

# the container path (this is the CI job)
docker build --target smoke --build-arg INSTALL_MODE=light -t voice-chat-smoke .
```

`backend/tests/` exists because the bugs that actually shipped — CJK relevance scoring,
truncated tool-call JSON, wrong-language TTS segments — were all in pure functions that nothing
tested. `INSTALL_MODE=full` and in-container CUDA are still uncovered; both need a GPU host.

Pronunciation is measured rather than argued: `test_tts_asr_roundtrip.py` synthesizes, re-
transcribes and reports CER per category against the ASR's own error floor (10.5 % paced /
5.3 % one-shot), varying one factor at a time. Read results as `cer − floor`, and treat any
category delta under ~7 pp at n=4 as a tie.

If you write your own UI-level sweep, two things are easy to get wrong: **assert on what the UI
renders** (mirror `handleServerMessage` — a version that accumulated `token` while the frontend
renders `text_so_far` reported 12/12 green against a transcript no user would ever see), and
**check the answer bubble separately from the audio**, since reasoning kept out of speech can
still be sitting in the transcript.

## License

Code Apache-2.0. Models: Qwen3.5, Qwen3-TTS, Granite-embedding-97M, X-ASR-int8 (all
Apache-2.0); SearXNG AGPL-3.0.
