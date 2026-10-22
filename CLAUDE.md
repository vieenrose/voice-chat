# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Paths below are relative to the repo root unless a `cd` is shown.

**Services (GPU host, weights under `/home/user/llms`, `/tmp/qwen3_tts`, etc.)** — launch flags live in the deploy scripts, not in env files:
```bash
SEARXNG_SETTINGS_PATH=/tmp/searxng/settings.yml python3 -m searx.webapp   # SearXNG :8888
backend/s2s/deploy/llm-audio.sh          # Gemma 4 E4B llama-server :11435 (needs --mmproj)
backend/s2s/deploy/llm-e2b-caption.sh    # Gemma 4 E2B caption llama-server :11436
backend/s2s/deploy/pipeline.sh           # speech-to-speech Realtime server :8765
cd frontend && npm install && npm run dev   # UI on :5173
```
`pipeline-stt.sh`, `pipeline-audio.sh` (no tools), `llm.sh` (text only) and `freetoken.sh` are variants; see the README table before using them.

**Tests and lint** (CI runs the first two; see `.github/workflows/ci.yml`):
```bash
ruff check backend --config ruff.toml                    # must stay at zero findings; do not add ignores
python3 -m unittest discover -s backend/tests -v         # run from repo root, as CI does
cd backend && python3 -m pytest tests/ -q                # alternative; README says unittest can't import tests/ on some layouts
cd backend && python3 -m pytest tests/test_tool_guard.py -q          # one file; add ::test_name for one test
```
The unit tests need the CPU subset CI installs (`fastapi uvicorn httpx websockets numpy scipy psutil loguru jieba lxml soundfile`); without them most test modules fail to import. Some zh-TW tests depend on the `zhconv` converter used in `s2s/tts_text.py`.

**Live checks** (need the full stack; close any browser tab first, the server holds one session slot):
```bash
cd backend && python3 -m s2s.checks.exhaustive          # --quick skips spoken turns
cd backend && python3 -m s2s.checks.turn "台灣的首都是哪裡？"
cd backend && python3 -m s2s.checks.bargein
cd frontend && node checks/turn.mjs    # also bargein.mjs; log/zhtw/endpoints.mjs need no server
```

**Container smoke:** `docker build --target smoke --build-arg INSTALL_MODE=light -t voice-chat-smoke .` — the light image has no model stack and uses mock STT/TTS adapters.

## Architecture

The live voice path is `speech-to-speech` 0.2.12 (pip package, owns VAD, endpointing, TTS, the OpenAI Realtime transport on :8765, and barge-in) with **one stage replaced**. `backend/s2s/serve.py` swaps `get_llm_handler` for `s2s/agent_handler.py`; nothing else in the framework is forked.

Turn flow:
1. Silero VAD + Smart Turn segment the mic audio. `--stt none`: there is no speech-to-text stage on the answering path.
2. `s2s/agent_handler.py` (`AgentLanguageModelHandler`) turns the segment into an `input_audio` content part and records finished turns into the framework's `Chat` itself, since nothing else does for `--stt none`. Only the last three spoken turns keep their audio; barged-in turns are never recorded.
3. `agent/native_loop.py` runs the call → observe → answer loop against llama-server `:11435` (`/v1/chat/completions`, native tool calls). It is bounded by `LLM_AGENT_MAX_STEPS` (default 3). The tool implementations live in `agent/harness.py` (`web_search`, `get_weather`, `get_current_datetime`); the `search_contacts` tool is defined in `agent/qwen_harness.py`, and which set reaches the native loop is worth checking before assuming. Arguments are checked by `agent/tool_guard.py` (`validate_args`) and results are fenced by `sanitize_tool_output` before re-entering the prompt. Tools execute server-side; the Realtime client-tool protocol is deliberately unused.
4. Answer text goes through `agent/` filters and `s2s/tts_text.py` (zh-TW rules), then sentence-sized chunks to Qwen3-TTS (`tts/`, GGML on CUDA, 24 kHz output).
5. Barge-in: VAD `speech_started` triggers the framework's CancelScope, which cancels the LLM and flushes TTS.

**Caption (second model, never on the answer path):** `s2s/caption_gemma.py` (`GemmaCaptionStream`) gets a copy of the same PCM via `AudioHandler.append_pcm` and re-sends the whole utterance to llama-server `:11436` every `S2S_CAPTION_GEMMA_INTERVAL` seconds, constrained by a JSON schema to `{"transcript": ...}`. It publishes *partial* transcripts only. `S2S_CAPTION_BACKEND=xasr` selects the older engine (`s2s/caption.py`); `S2S_CAPTION=0` disables captions.

**Other pieces:**
- `backend/app.py` is a separate FastAPI app (`/health`, `/search`, `/api/search`, `/api/chat`, `/ws/chat`, model switching via `llm_manager.py`, which owns a llama-server subprocess). The browser voice path does not use it; it talks Realtime to :8765.
- `backend/pipeline/` and `backend/llm/`, `backend/stt/`, `backend/tts/` contain adapter ladders: each `try: import heavy_dep` probes an optional dependency and falls through to a mock. The unused imports in those folders are deliberate; ruff ignores F401 there.
- `backend/tools/contact_db.py` searches `tools/directory.csv` (name, pinyin, extension, department). Extensions are read back by `spoken_ext`, and the agent's temperature is pinned low (`LLM_AGENT_TEMP=0.2` in `pipeline.sh`) so digits are not improvised.
- `backend/s2s/checks/` are end-to-end probes against a live server; `backend/tests/` are CPU-only unit tests; `frontend/checks/*.mjs` drive the UI's own client.
- Frontend (`frontend/src/`): Svelte 5 + Vite. `lib/realtime.js` is the protocol client, `lib/audio.js` capture and playback (`public/audio-processor.js` worklet), `lib/endpoints.js` builds the URLs from page origin, `lib/zhtw.js` converts Simplified output to Traditional for display. It holds no session state.

## Conventions and gotchas

- **Output is Traditional Chinese (Taiwan)** end to end. Simplified glyphs are fed to Qwen3-TTS on purpose, and `zhtw.js` / the backend converter fix what the user reads. Do not "fix" the TTS input to Traditional.
- **Launch parameters belong in `backend/s2s/deploy/*.sh`**, not in the README or env defaults; the README only points at them.
- **Realtime protocol details that bite:** `input_audio_transcription.delta` is cumulative (replace, don't append); input audio is 16 kHz and must not be declared in `session.update` (the schema pins 24000); with audio output, assistant text arrives in `response.output_audio_transcript.done`, once per chunk, and accumulates.
- **The Realtime endpoint has no authentication.** Keep `--ws_host 127.0.0.1`; expose via `tailscale serve`, never Funnel or a public bind.
- **`/tmp` is tmpfs (RAM with a quota).** Weights and anything large must go on a real disk, not `/tmp`.
- **Only one pipeline session at a time** (`--num_pipelines 1`). A stale browser tab or a running `s2s.checks` probe blocks the next connection.
- Lint is held at zero. Several rules were added because they caught real bugs (bare `except:` swallowing `asyncio.CancelledError`, `zip` without `strict`); fix findings rather than extending the ignore list.
- `docker-compose.yml` is an older stack (granite embedding, Qwen3.5-2B on :11435) and does not match the Gemma deploy scripts; check before using it.
