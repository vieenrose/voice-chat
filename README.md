# Voice Chat — Streaming Speech-to-Speech with Tools, in zh-TW

**A fully local zh-TW voice agent on HuggingFace `speech-to-speech`.**
Silero VAD + Smart Turn → Paraformer → Qwen3.5 4B (Qwen-Agent, real tools) → Qwen3-TTS,
speaking the OpenAI Realtime protocol. No hosted API, no mocks, no cloud.

Speak, interrupt it mid-sentence, search the web, hear the answer.
**Traditional Chinese (Taiwan) throughout** — replies are zh-TW whatever language the question
was asked in, with English kept only for proper nouns and untranslatable terms.

```mermaid
flowchart LR
    Mic["🎤 Mic"] -->|OpenAI Realtime<br/>WebSocket / WebRTC| RT["Realtime server<br/>:8765"]
    RT --> VAD["Silero VAD v5<br/>+ Smart Turn v3.2"]
    VAD --> STT["STT<br/>Paraformer · zh"]
    STT --> AG["LLM stage<br/>Qwen-Agent harness"]
    AG <--> LLM["llama-server :11435<br/>Qwen3.5 4B Q8_K_XL"]
    AG -->|native tool call| T{{"web_search · get_weather<br/>get_current_datetime"}}
    T --> SX["SearXNG :8888<br/>+ wttr.in"]
    SX -.results.-> AG
    AG -->|sentence| TTS["Qwen3-TTS<br/>GGML · 24k"]
    TTS --> SPK["🔊 Speaker"]
    VAD -.->|speech_started| CS(["CancelScope<br/>cancel + flush"])
    CS -.-> AG
    CS -.-> TTS

    classDef m fill:#1f2430,stroke:#7c5cff,color:#eee
    classDef s fill:#12281c,stroke:#3f9e6a,color:#eee
    classDef f fill:#2a2033,stroke:#c07cff,color:#eee
    class STT,LLM,TTS,AG m
    class SX,T s
    class RT,VAD,CS f
```

## Architecture

Orchestration is [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech)
(the pip package `speech-to-speech`). It owns the pipeline — VAD, endpointing, STT, TTS, the
transport and cancellation — with each stage a `BaseHandler` in its own thread, joined by queues.

**Every component is local.** HuggingFace supplies the weights and the framework; nothing calls a
hosted API. The LLM runs on llama.cpp, TTS on GGML through `faster_qwen3_tts`, STT on FunASR, VAD
and Smart Turn on local ONNX. (`transformers` has no `pipeline("speech-to-speech")` task — the
framework is the standalone package.)

Components stay **native to the framework** wherever one exists: Paraformer for Chinese STT, the
stock Qwen3-TTS handler pointed at GGUF weights on disk, Silero VAD, Smart Turn. The LLM stage is
the one deliberate exception.

### The LLM stage

`backend/s2s/qwen_agent_handler.py` replaces what `s2s_pipeline.get_llm_handler` returns, so the
turn is driven by **Qwen-Agent** while every other stage is upstream's. Substituting that one
factory function is the whole integration — there is no fork to rebase.

Two constraints shape it:

- **Chunks must be sentence-sized.** `LMOutputProcessor` forwards them to TTS 1:1 with no
  splitting. Flushing gates on content rather than length — one ideograph, or two latin letters —
  because 「好。」 is a whole sentence in Chinese while a bare `1.` in a list is not.
- **Tools run server-side.** The Realtime protocol expects the *client* to execute a tool and post
  `function_call_output` back, which cannot work here: `web_search` talks to SearXNG on 127.0.0.1
  and the clock and weather tools are backend resources. Qwen-Agent's call → observe loop stays
  intact and the browser receives ordinary assistant text.

### Tool calls are native, not prompted

`use_raw_api` passes `tools=` to the server, so llama-server (`--jinja`) generates each call under
the model's own chat template with grammar constraints: it cannot emit malformed JSON, and the
call cannot be crowded out by whatever else the model is generating.

This is the alternative to qwen-agent's default, which injects a `<tool_call>{json}</tool_call>`
template into the system prompt and regex-parses the model's free text. Measured on Qwen3.5 4B:

| | weather turns | all five prompt shapes |
|---|---|---|
| **native tools** | **9/9** | **0 failures, 0 reasoning leaks** |
| prompt dialect | 8/9 | — |

Under the prompt dialect the failure mode is the model emitting neither a tool call nor an answer,
having spent its budget thinking; the user then hears 「抱歉，我找不到相關的答案」, which claims a
lookup found nothing when none ran. `max_tokens` is 2048 — room for thinking, a tool call and the
answer — which is safe because native calls cannot be crowded out.

`reasoning_budget_tokens` is not set: llama-server ignores it as a per-request field. The server
flag `--reasoning-budget` does work and is deliberately unused, because capping thinking takes the
tool-call decision with it (5/12 weather turns at 200).

### Barge-in

The framework's `CancelScope` handles it structurally rather than per-call-site: a generation
counter that `BaseHandler.should_process_input` checks, so every stage inherits cancellation. On
`speech_started` the send loop cancels, flushes the queues, and drops output tagged with a
superseded generation.

Smart Turn v3.2 adds semantic endpointing: an ambiguous pause starts STT and the LLM
speculatively, and if the user resumes, the turn reopens as a new revision and the superseded
work is discarded before it reaches the speaker.

Measured: a 525-character answer (~40 s of speech) cut **1.29 s** after the user started talking —
`response.done status=cancelled reason=turn_detected` — which is Silero's detection window.

Once TTS output outruns playback, a response can finish server-side while audio is still queued in
the browser. Interrupting then has nothing to cancel, so the client flushes its own scheduled
audio (`Playback.flush()`).

### The harness

`backend/agent/qwen_harness.py` declares three tools, a system prompt, and the agent loop.
Nothing inspects the model's answer to decide what it *should* have done, and no rule matches
question phrasing: the model calls its own tools.

What remains is filtering, in `backend/llm/ling_streaming.py`, and it is shape-based rather than a
list of observed phrases:

- **Nothing the machine says to itself is spoken.** Reasoning goes to `reasoning_content` via
  `--reasoning-format deepseek` and reaches the client as its own event. Four other things look
  like answers and are not: the system prompt replayed back (matched by shingle overlap, so
  paraphrases are caught), qwen-agent's Simplified-Chinese tool narration, checklist scaffolding
  (`Evaluate the Input:`), and any standalone sentence-length all-English span — answers here are
  zh-TW with English only *inside* a Chinese sentence.
- **A sentence recognized as reasoning after streaming is retracted** from the transcript as well
  as withheld from speech (`retract_span`), since a single token is too small a unit to classify.
- **Nothing is said twice.** The harness streams deltas live, then reconciles against the
  authoritative final text and speaks only the remainder — comparing normalized forms, and testing
  containment first, because the final text can be a filtered *subset* of what streamed.
- **A short answer is an answer.** Content is detected by one ideograph or a three-letter latin
  word, never by length: 台北 answers 台灣的首都是哪裡？ in two characters, as do 是的 and 五.
- **Markdown is not speech.** `tts/spoken_text.py` rewrites notation only — markup, `°C`→`度`,
  `68%`→`百分之68`, emoji, URLs — never content, and reports which rules fired. Text is then
  segmented by script and synthesized with an explicit per-segment language.
- **Search is honest.** No curated results. Chinese queries are tokenized with jieba before
  relevance scoring, and fallback backends race concurrently rather than each waiting out its own
  timeout.

The language instruction is stated once, in the system prompt, never appended to the user's turn.

## Stack

| Layer | Model / Service | Notes |
|---|---|---|
| **Orchestrator** | `speech-to-speech` 0.2.12 | OpenAI Realtime server on `:8765`, WebSocket + WebRTC |
| **Turn-taking** | Silero VAD v5 + Smart Turn v3.2 | local ONNX; speculative turns with revisions |
| **STT** | Paraformer (FunASR) | Chinese-oriented; `pip install "speech-to-speech[paraformer]"` |
| **LLM** | `Qwen3.5-4B-UD-Q8_K_XL` | llama-server `:11435`, MTP on, thinking on, 3 tools |
| **Agent** | `qwen-agent` `Assistant` | custom s2s LLM stage, native tool calls |
| **TTS** | `Qwen3-TTS-12Hz CustomVoice` Q8_0 | stock s2s handler on GGML CUDA, ~20-80 ms TTFA |
| **Search** | SearXNG `:8888` + wttr.in | 180 engines, real results only |
| **Embedding** | `granite-embedding-97m-multilingual` Q8_0 | 384 d, `:11434`, semantic rerank |

One model, deliberately: `Qwen3.5 4B UD-Q8_K_XL` is the reference everything is validated against.
Chosen on perplexity over 194 kB of non-repeating zh-TW Wikipedia prose (4B Q8 13.027 against
Q4_K_XL 13.173) — a deterministic measure, unlike a handful of UI prompts whose spread is noise.
The framework also accepts `--llm_backend transformers`, which would run int8 safetensors
in-process instead of llama.cpp.

## Quick Start

```bash
# 1) Search backend (tools)
SEARXNG_SETTINGS_PATH=/tmp/searxng/settings.yml python3 -m searx.webapp   # :8888

# 2) LLM
llama-server -m /home/user/llms/mtp/Qwen3.5-4B-UD-Q8_K_XL.gguf \
  --host 127.0.0.1 --port 11435 -c 8192 --alias qwen3.5-4b \
  --n-gpu-layers 99 --jinja --reasoning-format deepseek

# 3) The speech-to-speech pipeline, with Qwen-Agent as its LLM stage
cd backend && python3 -m s2s.serve --mode realtime \
  --ws_host 127.0.0.1 --ws_port 8765 \
  --stt paraformer --language zh \
  --model_name qwen3.5-4b \
  --responses_api_base_url http://127.0.0.1:11435/v1 --responses_api_api_key none \
  --tts qwen3 --qwen3_tts_backend ggml --qwen3_tts_language zh \
  --qwen3_tts_gguf_talker_path /tmp/qwen3_tts/talker_cv_q8.gguf \
  --qwen3_tts_gguf_codec_path  /tmp/qwen3_tts/codec.gguf \
  --enable_live_transcription
```

Then talk to it — any OpenAI Realtime client works, including the framework's own:

```bash
speech-to-speech talk --url ws://127.0.0.1:8765/v1/realtime
```

Or open the web client:

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173
```

### Reaching it from another machine

The client derives its endpoint from the page's own origin, so serving the UI elsewhere means
the WebSocket has to be reachable there too. Two cases:

- **Page on `http://`** (dev, or the same box) — the client uses `ws://<host>:8765` and the
  pipeline needs `--ws_host 0.0.0.0` to accept it.
- **Page on `https://`** — a browser blocks `ws://` from an https page as mixed content, and
  the pipeline speaks plain ws, so something must terminate TLS. The client uses
  `wss://<host>:8443`, and Tailscale can be that terminator:

  ```bash
  tailscale serve --bg --https 8443 http://127.0.0.1:8765
  ```

  This is **tailnet-only** even when the page itself is published through Funnel, and it lets
  the pipeline stay bound to `127.0.0.1`, which is the point: the Realtime endpoint has **no
  authentication**, so anyone who can reach it can drive the LLM and the GPU. Never Funnel it
  or bind it to a public interface.

Either default is only a default — the endpoint is editable in the UI.

Press **連線**, then **開始說話**. It speaks the Realtime protocol straight to `:8765` — no
backend of its own — and shows live partial transcripts, the reply as it is spoken, and a
cancelled badge on any turn you interrupt.

**⬇ 記錄** in the header downloads the whole session as one `.json` — the protocol trace both
ways, the transcript, client-only events (barge-in flushes, mic errors), and the audio/latency
counters. Attach it to a bug report instead of describing what happened. Audio payloads are
elided rather than stored, so a 300-chunk turn exports at ~48 KB instead of ~3.5 MB, and mic
frames are counted rather than listed (they arrive every 20 ms and would bury everything else).

> The server runs **one session at a time** by default (`--num_pipelines 1`); further
> connections are rejected with "all 1 pipeline slots in use". An open browser tab holds that
> slot, so close it before running the checks below. Each extra pipeline has its own STT/TTS
> handlers and costs VRAM.

Or check it all without a microphone:

```bash
cd backend  && python3 -m s2s.checks.turn "台灣的首都是哪裡？"  # one turn, server side
cd backend  && python3 -m s2s.checks.bargein                    # interrupt a long answer
cd frontend && node checks/turn.mjs                             # a voice turn through the UI's client
cd frontend && node checks/bargein.mjs                          # barge-in through the UI's client
cd frontend && node checks/log.mjs                              # the debug export (no server needed)
```

Dependencies beyond `requirements.txt`:

```bash
pip install "speech-to-speech[paraformer]"    # Paraformer STT through FunASR
```

Weights are pulled from the Hub on first use (Silero VAD, Smart Turn v3.2, Paraformer) except
the two already on disk: LLM `/home/user/llms/mtp/Qwen3.5-4B-UD-Q8_K_XL.gguf`, TTS
`/tmp/qwen3_tts/talker_cv_q8.gguf` + `codec.gguf`. Embedding
`/tmp/granite-emb-gguf/…Q8_0.gguf` · SearXNG `/tmp/searxng/settings.yml`.

> **`/tmp` here is tmpfs — it is RAM, and it enforces a quota.** A multi-GB write fails partway
> with `Disk quota exceeded` even though `df` shows free space. Put new weights on a real disk
> and point the matching `LLM_PATH_*` at them.

### The frontend

`frontend/` is a Realtime client and nothing else: `src/lib/realtime.js` (protocol),
`src/lib/audio.js` (mic capture, playback queue), `src/App.svelte` (UI). It holds no session
state of its own and calls no HTTP API — the server owns turn-taking, so the UI only renders
what the protocol reports.

It implements the protocol details under [API](#api), plus three of its own:

- **Barge-in flushes the browser.** Cancelling server-side stops new audio, but chunks already
  scheduled in the `AudioContext` keep playing, so `Playback.flush()` stops the scheduled
  sources rather than only the queue.
- **Audio rates are pinned.** Capture pins an `AudioContext` to 16 kHz so the browser does the
  resampling from whatever the device offers; playback requests 24 kHz, the rate Qwen3-TTS
  produces, which the server would otherwise downsample.
- **The transcript is converted for display.** Paraformer emits Simplified, so `lib/zhtw.js`
  applies OpenCC S2T at the **character** level (`to: 'tw'`), because a transcript must stay
  invariant at the phonetic level — it is a record of what was said, and S2T character mappings
  are homophonous (臺北/台北 both *tái-běi*). The `twp` preset is avoided: it substitutes
  regional vocabulary and so changes pronunciation, turning 用鼠标点击 into 用滑鼠點選
  (*shǔbiāo diǎnjī* → *huáshǔ diǎnxuǎn*), a sentence the user never said. The dictionaries load
  lazily, so the main bundle stays ~53 kB and nothing on the audio path waits for them.

**⬇ 記錄** in the header exports the session as one `.json`: the protocol trace both ways, the
transcript (with the raw STT text alongside the converted form), client-side events such as
barge-in flushes and microphone errors, and the audio counters. Audio payloads are elided and
mic frames counted rather than listed, so a 300-chunk turn exports at ~48 kB instead of ~3.5 MB.

## Features

- **Barge-in on speech detection**, not on end-of-utterance — the distinction is the whole
  feature on a long answer. Handled by the framework's `CancelScope`, with the client flushing
  its own queued audio.
- **Real tools**, called natively by the model: `web_search`, `get_weather`,
  `get_current_datetime`. No curated results, no fabricated forecasts.
- **zh-TW throughout.** Replies default to Traditional Chinese whatever the input language; the
  user's transcript is converted with OpenCC for display.
- **Thinking never spoken.** Deliberation stays in `reasoning_content`, out of both the audio and
  the transcript.
- **Live partial transcripts** while you speak, and a debug export of the whole session.

## Measured (RTX 3060, live stack)

End to end over the Realtime protocol, fully local, cold client each time
(`s2s/checks/turn.py`):

| Prompt | First audio | Total |
|---|---|---|
| 你好 | 1.4 s | 2.0 s |
| 現在幾點？ (clock tool) | 3.6 s | 4.6 s |
| 帮我查一下今天那个台北的天气 (weather tool) | 9.7 s | 15.6 s |

Per-stage p50 over the five prompt shapes, measured at the harness
(`ttft` is the model's first token, so it excludes TTS):

| Shape | ttft p50 | total p50 | Answer |
|---|---|---|---|
| chat | 1.25 s | 1.44 s | 13 chars |
| plain | 1.34 s | 1.59 s | 20 chars |
| clock | 2.73 s | 3.48 s | 41 chars |
| weather | 6.50 s | 9.11 s | 150 chars |
| search | 7.34 s | 10.57 s | 225 chars |

Qwen3-TTS time-to-first-audio is 0.02–0.08 s at RTF 0.2–0.25, so a tool turn's latency is the
lookup round-trip, not synthesis.

**Barge-in: a 525-character answer (~40 s of speech) cancelled 1.29 s after the user started
speaking** (`status=cancelled reason=turn_detected`), which is Silero's detection window.

Tool routing and reasoning leaks: 0 failures and 0 leaks over all five shapes.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `S2S_LLM_API_BASE` / `S2S_LLM_MODEL_NAME` | `…:11435/v1` / `qwen3.5-4b` | which llama-server the LLM stage talks to |
| `S2S_USE_UPSTREAM_LLM` | unset | `1` runs the stock s2s LLM stage instead of Qwen-Agent |
| `QWEN_AGENT_USE_RAW_API` | `true` | native tool calls; `false` uses qwen-agent's prompt dialect |
| `LLM_AGENT_MAX_TOKENS` | `2048` | per-turn generation cap (thinking + tool call + answer) |
| `LLM_AGENT_THINKING` | `1` | thinking pass on agent turns |
| `LLM_AGENT_TEMP` / `LLM_AGENT_TOP_P` | `0.7` / `0.9` | agent-turn sampling |
| `LLM_AGENT_SEED` | unset | reproducible agent turns |
| `LLM_AGENT_NO_TOOL_SMALLTALK` | `1` | greetings get a reply, not a tool call |
| `VOICE_TZ` | `Asia/Taipei` | timezone `get_current_datetime` reports when the model passes none |
| `SEARXNG_URL` | `http://localhost:8888` | search backend for `web_search` |
| `LLM_PATH_4B`, `LLM_PORT`, `LLM_CTX` | see `llm_manager.py` | weights and llama-server binding |
| `LLM_MTP` / `LLM_MTP_DRAFT_N` | `1` / `3` | self-speculative decoding on MTP weights |
| `EMBED_API_BASE`, `EMBED_MODEL` | `:11434` | semantic rerank of search results |
| `TTS_MODEL_DIR` | `/tmp/qwen3_tts` | TTS GGUF location (Docker mount) |

## API

The interface is the OpenAI Realtime protocol on `:8765` — any client implementing the core
event set works, including the framework's own `speech-to-speech talk`.

**Client → server:** `session.update`, `input_audio_buffer.append`,
`conversation.item.create`, `response.create`, `response.cancel`.

**Server → client:** `input_audio_buffer.speech_started` / `speech_stopped`,
`conversation.item.input_audio_transcription.delta` / `.completed`,
`response.output_audio.delta`, `response.output_audio_transcript.done`, `response.done`.

Three details decide whether a client behaves correctly:

- **`input_audio_transcription.delta` is cumulative.** Each carries the whole transcript so far,
  not the increment, so partials must *replace*. Appending yields 「欢迎欢迎大家来欢迎大家来体验…」.
- **A voice turn never emits `response.created`** — the server sends it only for an explicit
  `response.create`. End-of-speech is what signals that a reply is coming.
- **With an audio response the assistant text arrives on
  `response.output_audio_transcript.done`**, once per spoken chunk, so those accumulate rather
  than replace.

Input audio is base64 PCM16 at the server's 16 kHz pipeline rate, and the input format must not
be declared: `AudioPCM.rate` is `Literal[24000]` in the schema, so sending the real 16 kHz
rejects the entire `session.update`. Output declares 24000, the rate Qwen3-TTS produces.

One session at a time by default (`--num_pipelines 1`); further connections are rejected. Each
extra pipeline has its own STT/TTS handlers and costs VRAM.

## Security

**The Realtime endpoint has no authentication.** Anyone who can reach `:8765` can drive the LLM
and the GPU. So:

| Control | Practice |
|---|---|
| Binding | `--ws_host 127.0.0.1`; expose it with `tailscale serve`, which stays tailnet-only |
| Never | Funnel it, or bind it to a public interface |
| TLS | required for microphone access from another machine — browsers only grant `getUserMedia` in a secure context |
| SearXNG | loopback only; it is reachable from the tool, not from the client |

## Validating a change

```bash
python3 -m unittest discover -s backend/tests   # pure logic, no GPU or network
ruff check backend --config ruff.toml           # gate: E4, E7, E9, F, B

# against a LIVE pipeline on :8765 (close the browser tab first -- one session slot)
cd backend  && python3 -m s2s.checks.turn "台灣的首都是哪裡？"
cd backend  && python3 -m s2s.checks.bargein
cd frontend && node checks/turn.mjs             # a voice turn through the UI's own client
cd frontend && node checks/bargein.mjs
cd frontend && node checks/log.mjs              # debug export; no server needed
cd frontend && node checks/zhtw.mjs             # transcript conversion; no server needed
```

`backend/tests/` covers the pure functions where shipped bugs actually lived — CJK relevance
scoring, tool-call JSON, short-answer filtering, stream reconciliation.

Writing a UI-level sweep? **Assert on what the UI renders.** A harness that accumulates raw
tokens while the client renders reconciled text will report green against a transcript no user
sees. And **check the answer bubble separately from the audio**: text kept out of speech can
still reach the transcript.

## License

Code Apache-2.0. Models: Qwen3.5, Qwen3-TTS, Granite-embedding-97M, Paraformer, Silero VAD,
Smart Turn v3.2 (Apache-2.0/MIT); SearXNG AGPL-3.0.
Orchestration: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech).
