# Tool calling — how it actually works today

_(Replaces the original MiniCPM5-era note, which described a `<tool>`-XML parser, a
heuristic search trigger and a "mock curated DB" — all three were deliberately removed
in `a15b9bd` / `42e7855` / `2068d84`. Nothing here is hard-coded per outlet, city or
demo question.)_

## Flow

```
stt_final / text_input
  -> LingStreaming.generate_chat_with_tools(history, prompt + LANG_HINT)
       -> agent/qwen_harness.run_agent_task(task, queue, history)   # Qwen-Agent Assistant
            - tools: web_search | get_weather | get_current_datetime
                -> tools/web_search.web_search_sync() on a persistent search loop
                   wttr.in -> SearXNG aggregate -> embedding rerank -> (bing | lite | DDGS, raced)
            - emits tool_call / tool_result / llm_delta into the queue
       -> consumer relays events; llm_delta becomes llm_token, so TTS starts while the
         agent is still generating (a tool turn used to produce no audio for ~7 s)
```

Harness selection is a ladder: `qwen_harness` -> `harness` (smolagents) -> `pydantic_harness`.
All three expose `run_agent_task(task, event_q, history=None)` and `reset_agent()`, share
`agent/_shared.py` (per-call event routing via `contextvars`, one lock per shared agent
instance) and are rebuilt after `POST /api/model` so they pick up the new model alias.

## Tools

| Tool | Backing | Notes |
|---|---|---|
| `web_search(query, count)` | SearXNG (`SEARXNG_URL`) + wttr.in + scraping fallbacks | 5 min TTL LRU cache, revalidated by relevance on read |
| `get_weather(location, date)` | wttr.in `format=j1` (+ SearXNG for context) | `today` / `tomorrow` / `day_after_tomorrow` -> day index |
| `get_current_datetime(timezone)` | `zoneinfo` | returns today / tomorrow / yesterday with weekdays |

## Guards on what the model claims (found by measuring, not reading)

| Failure observed | Detector | Repair |
|---|---|---|
| `現在的時間是 [get_current_datetime]。` — tool "called" by printing its name, nothing executed | `detect_unexecuted_tool()` | run the named tool (its own events fire, so the UI shows it), then splice the value or hand it back for one corrective turn; `pipeline._is_tool_artifact()` separately refuses to speak the draft |
| `現在是下午 3 點 25 分` (it was 13:25) — a stated time with no tool behind it | `fabricates_time_without_tool()` — needs a *now* question **and** a time claim in the answer | force the real `get_current_datetime` call and rewrite the sentence with `format_clock_zh()`; a model redo reproduced the same wrong time, so this path is deterministic |

Both are opt-out-free and cheap: the clock path adds one local tool call and no LLM
round-trip. Measured effect on routing: **62 % → 86 %** on the 7-query set.

## Reproducing a run

`LLM_AGENT_SEED` (per-request, forwarded by qwen-agent) and `LLM_SEED` (llama-server
process-wide) exist because tool routing at `temperature 0.7` scored 57 %, 57 %, 71 % on
identical inputs. `test_e2e_report.py --tool-repeats N` reports the spread. `LLM_AGENT_TEMP`
/ `LLM_AGENT_TOP_P` expose the agent's sampling so routing can be A/B-ed against the metric
instead of argued about.

## Streaming the answer (why tool turns got faster)

qwen-agent's `FnCallAgent._run` yields `response + partial_output` on **every** LLM
chunk, so the final step's `content` is available incrementally. `qwen_harness` emits
`llm_delta` events as it grows, per **step**:

- content grows -> `llm_delta` delta
- the step's tail becomes a tool call / tool result, or a new step starts -> `llm_delta
  {reset: true}`, i.e. "what I streamed was not the answer, drop it", and streaming is
  re-armed for the next step
- `...` / `<arg_` / `<|` appears in content -> permanent abandon for that turn

`llm_reset` is forwarded to the client, which clears its partial bubble. Measured effect:
first audio on a `web_search` turn went from **6.7 s to 1.0-1.4 s**.

## Search quality mechanics (no cheating, so quality has to be engineered)

- **`repair_truncated_json_query`** — qwen-agent's own `extract_fn()` strips the last
  character of any tool-call block that never emitted its closing tag. When a small
  quantized model finishes the JSON payload but forgets that tag, a valid
  `{"query": "..."}` loses its brace before reaching the tool. The repairer retries a few
  plausible suffixes, then a regex pull, then the raw string.
- **`_tokenize_query` uses jieba** — a naive CJK regex treats 最新科技新聞 as ONE token
  that never appears verbatim in a result, so every Chinese query scored ~0 and the
  cache/fallback logic silently misfired.
- **Relevance-gated cache** — a cache hit whose score is < 0.34 is *discarded* and the
  search re-runs. A query that yields no scorable tokens scores neutral (1.0), not 0:
  "cannot judge" is not evidence of a bad search, and scoring it 0 made such entries
  permanently unreachable.
- **Embedding rerank** (Granite-97M-multilingual, `EMBED_API_BASE`) rescues paraphrases
  the keyword score can't, e.g. "big news days" -> 重大新聞.
- **Fallbacks race concurrently** (DDGS | lite | Bing) instead of walking a serial chain
  of multi-second timeouts; DDGS is skipped for Chinese queries, where it never won.
- **`_entity_first_query`** regionalizes a bare topic-only Chinese query (最新科技新聞 ->
  台灣最新科技新聞) — nothing else in the query already names a place.

## Debugging a bad answer

```bash
curl "http://localhost:8888/search?q=AI+%E6%96%B0%E8%81%9E&format=json" | jq '.results[0].title'   # is SearXNG itself good?
curl "http://localhost:8000/api/search?q=AI+%E6%96%B0%E8%81%9E" | jq '{source, latency_ms, n:(.results|length)}'   # which backend answered?
python3 backend/tools/web_search.py 最新科技新聞      # the tool exactly as the agent calls it
```

Then read the log lines: `SearXNG relevance for '...' = 0.xx`, `web_search '...' source=<x>
N results in Xms`, and `[LLM Tool] ...`. `source` names the winner (`searxng`, `wttr.in`,
`searxng:bing`, `bing_scrape`, `+emb` when the reranker reordered).

## Endpoints

```
GET  /api/search?q=…&count=5                 # search alone
POST /api/tools/web_search {"query":"…"}    # the tool, directly
POST /api/chat {"text":"…","tools":true}    # one turn, returns tool_calls[] + audio_b64
WS   /ws/chat                                # streams tool_call / tool_result / llm_token / llm_reset / tts_chunk
```

When `VOICE_CHAT_TOKEN` is set these all need `X-Auth-Token: …` (or `?token=…`).
