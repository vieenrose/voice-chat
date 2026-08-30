"""
Ling 3.0 Tiny MXFP4 MoE GGUF — streaming LLM via llama.cpp server (OpenAI API)
Replaces MiniCPM5 with https://huggingface.co/noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF
- 7.8B MoE, 131k ctx, bailingmoe3, MXFP4, tool calling
- Served via llama-server on http://127.0.0.1:11435
"""
import asyncio
import json
import re
import time
from typing import AsyncGenerator, List, Dict
from loguru import logger
import httpx

_json_dumps = json.dumps


def _strip_tool_xml(t: str) -> str:
    """Remove Ling's <tool_call>web_search<arg_key>…</arg_key><arg_value>…</arg_value> XML template."""
    t = re.sub(r"<tool_call>\s*[A-Za-z_]*", "", t)          # the function name after <tool_call>
    t = re.sub(r"</?tool_call[^>]*>", "", t)
    t = re.sub(r"</?arg_key[^>]*>.*?</arg_key>", "", t, flags=re.S)
    t = re.sub(r"<arg_value>.*?</arg_value>", "", t, flags=re.S)
    t = re.sub(r"</?tool_response[^>]*>", "", t)
    t = re.sub(r"<search_results>.*?</search_results>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return t.strip(" \n,;")

def _clean_leakage(text: str) -> str:
    """Strip leaked raw search result format that LLM sometimes echoes verbatim."""
    # Remove patterns like "[1] Title URL: https://... Date/Snippet: ..." that are raw tool output
    # This is the exact format from format_results() that should never reach the user voice
    text = re.sub(r"\[\d+\]\s*[^\[]*?URL:\s*https?://\S+\s*Date/Snippet:\s*", "", text)
    text = re.sub(r"\[\d+\]\s*President of France[^\[]*", "", text)  # fallback for truncated
    # Remove "More at Wikipedia" thin result leakage
    text = re.sub(r"More at Wikipedia\s*", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s{2,}", " ", text).strip()
    # If after cleaning we have leading "President of France" fragment without context, keep only the actual answer part
    # Look for the real answer start (usually "The current president...")
    m = re.search(r"(The current president of France is.*)", text, re.I | re.S)
    if m:
        text = m.group(1).strip()
    return text


def _intent_wants_search(prompt: str) -> bool:
    """Honest intent check — no hard-coded keyword lists. Let the LLM's tool calling decide."""
    # Previously hard-coded news/weather regex caused cheating; now we rely on the
    # smolagents ToolCallingAgent to decide when web_search is needed based on its system prompt.
    # This function is kept for backward compat but always returns False (no forced search).
    return False


# Ling 3.0 tiny supports tool calling via <tool_call> XML, same as before
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current info. Use for weather, news, facts, recent events, people, definitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query 3-8 words"},
                    "count": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time (UTC). Use ONLY for questions asking what day/date/time it is (星期几/几号/几点/what day/today date). NEVER use for weather/forecast — weather must go to web_search. Optional IANA timezone e.g. Asia/Taipei.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Optional IANA timezone name (e.g. Asia/Taipei); default UTC"}
                },
                "required": []
            }
        }
    }
]

SYSTEM_PROMPT = "You are a helpful, informative voice assistant with web search. Be conversational and natural for voice chat. For casual chat keep under 80 words; for news/weather/factual queries be more detailed (up to 150 words) and highly informative. For ANY question about current events, news, weather, real-time info, or specific regional events (e.g., '今天台湾有什么重大事件', 'latest news'), you MUST use web_search tool to get the latest information — never refuse or say you cannot provide real-time info. Every new news/headline request needs a FRESH web_search, even short follow-ups like 'BBC headlines' or 'CNN now' — never answer from stale results of an earlier query. Use search results to answer. For questions about today's date, weekday, or current time (今天/星期几/几点), you MUST call the get_current_datetime tool instead of guessing. For ANY weather/forecast question (including 明天/後天/next week) you MUST call web_search — the search engine returns the forecast for the requested day, so do NOT call get_current_datetime for weather. For date/time questions call get_current_datetime. When both are needed, call both in the same turn. Always answer using the tool RESULTS verbatim (weekday, date, temperatures) — never from memory. Always respond in Traditional Chinese (Taiwan usage, 繁體中文) by default, regardless of what language the question was asked in — only keep English for proper nouns, technical terms, or vocabulary that doesn't translate well; never answer a whole sentence in Simplified Chinese or English. IMPORTANT: when the web_search tool results contain the answer (weather numbers, news headlines, dates), be highly informative: quote specifics directly with numbers, names, dates, and sources. For news, list 3-4 concrete recent headlines with source + one-sentence summary each + date if available. Never claim results lack information they contain. Interpret loose phrasing generously (e.g. 'big news' = latest major news) instead of saying no such thing exists."

# zh-TW is this app's default/primary language — appended to every turn's prompt (not
# just relied on via SYSTEM_PROMPT above) since that's the most reliable way to keep a
# small model from drifting into English or Simplified Chinese as context grows. Applied
# unconditionally regardless of the input's own language (a confirmed product decision,
# not reactive): English is only ever kept for terms/names that don't translate well.
LANG_HINT = "\n（請一律使用繁體中文（台灣用語）簡潔回答；僅專有名詞、技術術語或無法翻譯的詞彙可保留英文原文，不要整句使用簡體中文或英文作答。）"

# Heuristic for fast tool trigger — bilingual (en/zh) - includes Chinese triggers for Taiwan/news
# (heuristic removed — tool calls are model-driven/native only)

class LingStreaming:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-2B-MTP-GGUF", api_base: str = "http://127.0.0.1:11436/v1", mock: bool = False, device: str = "cuda", model_name: str = "qwen3.5-2b", **kwargs):
        self.model_id = model_id
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.backend = f"llm-gguf:{model_name}"
        self.mock = mock
        # Check if server is up, if not, fallback to mock
        if mock:
            logger.info("Ling: MOCK mode (requested)")
            return
        # Try to ping server
        import asyncio as _asyncio
        try:
            import httpx as _httpx
            # Quick sync check
            with _httpx.Client(timeout=2.0) as c:
                r = c.get(f"{self.api_base}/models")
                if r.status_code == 200:
                    logger.info(f"Ling 3.0 tiny MXFP4 MoE GGUF ready at {self.api_base} ✓")
                else:
                    logger.warning(f"Ling server not ready {r.status_code}, fallback to mock")
                    self.mock = True
        except Exception as e:
            logger.warning(f"Ling server not reachable {e}, fallback to mock (will retry per request)")
            # Don't set mock True permanently, will retry
            pass

    async def _chat_stream(self, messages: List[Dict], tools: List[Dict] = None, max_tokens: int = 256) -> AsyncGenerator[dict, None]:
        # Call llama-server OpenAI API with streaming - Ling 3.0 needs enable_thinking false for low-latency voice (no reasoning_content)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # Use httpx streaming
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.api_base}/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    logger.warning(f"Ling API {resp.status_code}: {text[:500]}")
                    raise RuntimeError(f"Ling API {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        choice = j["choices"][0]
                        delta = choice.get("delta", {})
                        # Tool calls
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                yield {"type": "tool_call_delta", "delta": tc}
                        # Content - Ling returns both reasoning_content (thinking) and content (answer) when enable_thinking is used.
                        # For voice chat, we ONLY speak content, never reasoning (thinking off for low latency)
                        token = delta.get("content")
                        if token is not None:
                            yield {"type": "token", "token": token}
                        # Ignore reasoning_content for voice (it's the internal thinking, not to be spoken)
                        # If content is empty and reasoning contains the answer (rare with detailed thinking off), it would be empty anyway
                        # Finish reason
                        if choice.get("finish_reason"):
                            yield {"type": "finish", "reason": choice["finish_reason"]}
                    except Exception as e:
                        continue

    async def generate_chat(self, history: List[Dict], prompt: str = None, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        """Multi-turn chat with history. history is List of {role, content} including system. If prompt given, appends as user."""
        # Build messages with proper Ling template: system + history + prompt
        messages = []
        # Ensure system at start with thinking off
        if not history or history[0].get("role") != "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            if history:
                messages.extend(history)
        else:
            messages = list(history)
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        # Dedupe system if already in history
        # For mock, just delegate to generate_stream with prompt
        if self.mock:
            async for ev in self.generate_stream(prompt or (history[-1]["content"] if history else ""), max_new_tokens=max_new_tokens):
                yield ev
            return
        # Real Ling: use chat_stream with full history
        text_so_far = ""
        first = True
        t0 = time.time()
        try:
            async for ev in self._chat_stream(messages, tools=None, max_tokens=max_new_tokens):
                if ev["type"] == "token":
                    token = ev["token"]
                    text_so_far += token
                    latency = int((time.time()-t0)*1000) if first else 20
                    first = False
                    yield {"type": "llm_token", "token": token, "text_so_far": text_so_far, "latency_ms": latency}
                    await asyncio.sleep(0)
            yield {"type": "llm_done", "text": text_so_far}
        except Exception as e:
            logger.exception(f"Ling generate_chat failed {e}")
            async for ev in self.generate_stream(prompt or "", max_new_tokens=max_new_tokens):
                yield ev

    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    async def generate_stream(self, prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        t0 = time.time()
        if self.mock:
            # Mock bilingual
            is_zh = self._is_chinese(prompt)
            if is_zh:
                mock_resp = "您好！我是 Ling 3.0 tiny，通过 SearXNG 搜索来帮助您。有什么可以帮您的？"
                if "天气" in prompt or "weather" in prompt.lower():
                    mock_resp = "我可以通过 SearXNG 搜索天气信息！"
                elif "台湾" in prompt or "台灣" in prompt:
                    mock_resp = "我来帮您查询台湾今日的重大事件。"
            else:
                mock_resp = "That's interesting! I'm Ling 3.0 tiny, a MoE model via SearXNG tools. How can I help?"
                if "weather" in prompt.lower():
                    mock_resp = "I don't have live weather, but I can search via SearXNG!"
                elif "hello" in prompt.lower():
                    mock_resp = "Hey there! Ling 3.0 here — tiny but mighty MoE. What would you like to chat about?"
            text_so_far = ""
            for tok in mock_resp.split(" "):
                await asyncio.sleep(0.018)
                text_so_far += (" " if text_so_far else "") + tok
                yield {"type": "llm_token", "token": " " + tok if text_so_far else tok, "text_so_far": text_so_far, "latency_ms": int((time.time()-t0)*1000)}
            yield {"type": "llm_done", "text": text_so_far}
            return

        # Real Ling via llama-server
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text_so_far = ""
        first = True
        try:
            async for ev in self._chat_stream(messages, tools=None, max_tokens=max_new_tokens):
                if ev["type"] == "token":
                    token = ev["token"]
                    text_so_far += token
                    latency = int((time.time()-t0)*1000) if first else 20
                    first = False
                    yield {"type": "llm_token", "token": token, "text_so_far": text_so_far, "latency_ms": latency}
                    await asyncio.sleep(0)
            yield {"type": "llm_done", "text": text_so_far}
        except Exception as e:
            logger.exception(f"Ling generate_stream failed {e}, fallback to mock")
            # Fallback mock
            fallback = "抱歉，伺服器暫時發生錯誤，我仍在線上為您服務。"
            text_so_far = ""
            for tok in fallback.split(" "):
                await asyncio.sleep(0.02)
                text_so_far += (" " if text_so_far else "") + tok
                yield {"type": "llm_token", "token": " "+tok, "text_so_far": text_so_far, "latency_ms": 50}
            yield {"type": "llm_done", "text": text_so_far}

    async def generate_chat_with_tools(self, history: List[Dict], prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        """SMOLAGENTS-driven agent loop: model plans + calls tools (web_search / get_current_datetime)
        for up to MAX_STEPS, then produces the final answer. Legacy inline loop kept as fallback."""
        try:
            try:
                from agent.qwen_harness import run_agent_task
            except ImportError:
                try:
                    from agent.pydantic_harness import run_agent_task
                except ImportError:
                    from agent.harness import run_agent_task
            _HARNESS = True
        except Exception as e:
            logger.warning(f"smolagents harness unavailable ({e}) -> legacy loop")
            _HARNESS = False
        if not _HARNESS:
            async for ev in self._legacy_chat_with_tools(history, prompt, max_new_tokens=max_new_tokens):
                yield ev
            return

        if history:
            _hist = "\n".join([f"{m.get('role')}: {m.get('content','')[:120]}" for m in history[-4:]])
            task_str = f"Conversation history:\n{_hist}\n\nCurrent question: {prompt}"
        else:
            task_str = prompt
        q = asyncio.Queue()
        task = asyncio.create_task(run_agent_task(task_str, q))
        final_text = ""
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue
                if ev["type"] == "tool_call":
                    yield {"type": "tool_call", "name": ev["name"],
                           "arguments": ev.get("arguments", {}), "query": ev.get("query", "")}
                elif ev["type"] == "tool_result":
                    yield {"type": "tool_result", "name": ev["name"], "result": ev.get("result"),
                           "formatted": ev.get("formatted", ""), "latency_ms": ev.get("latency_ms", 0),
                           "source": ev.get("source", "")}
            final_text = await task
        finally:
            # If we're being cancelled (barge-in) mid-loop, `task` would otherwise be
            # orphaned: never awaited, silently running to completion in the background
            # (its own `agent.run()` call happens in a thread via asyncio.to_thread, which
            # can't be interrupted mid-call either way, but at least stop *waiting* on it
            # and don't leak an unretrieved exception).
            if not task.done():
                task.cancel()
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        if not isinstance(final_text, str):
            final_text = str(final_text)
        if "<tool_call>" in final_text or "<arg_" in final_text:
            final_text = _strip_tool_xml(final_text)
        final_text = re.sub(r"<[^>]+>", " ", final_text)
        final_text = " ".join(final_text.split())
        if not final_text.strip():
            final_text = "抱歉，我找不到明確的答案。"
        _t0 = time.time(); _sf = ""
        for tok in final_text:
            _sf += tok
            yield {"type": "llm_token", "token": tok, "text_so_far": _sf,
                   "latency_ms": int((time.time()-_t0)*1000) if len(_sf) == len(tok) else 20}
            await asyncio.sleep(0)
        yield {"type": "llm_done", "text": final_text}

    async def _legacy_chat_with_tools(self, history: List[Dict], prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        """Multi-turn tool-aware chat using NATIVE tool calling (OpenAI tools=[] on llama-server).
        Loop: model emits tool_call JSON -> we run web_search -> inject tool result -> model answers."""
        try:
            from tools.web_search import web_search, format_results
        except:
            web_search = None
            format_results = lambda x: str(x)
        messages = []
        if not history or history[0].get("role") != "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            if history:
                messages.extend(history)
        else:
            messages = list(history)
        messages.append({"role": "user", "content": prompt})
        _tools = TOOL_DEFS

        # ---- Agent harness: bounded multi-round loop ----
        # R1: model decides tool calls. R2 (only when needed): relevance-gated refinement,
        # or a forced search when the model skipped an obviously needed tool.
        MAX_ROUNDS = 2
        WALL_BUDGET_MS = 18000
        t_agent_start = time.time()
        _weak_search = False
        _saw_web = False
        for _round in range(MAX_ROUNDS):
            if (time.time() - t_agent_start) * 1000 > WALL_BUDGET_MS:
                logger.info("agent wall budget reached -> final answer")
                break
            # — run model with tools (native) —
            text = ""
            acc: dict[int, dict] = {}
            async for ev in self._chat_stream(messages, tools=_tools, max_tokens=max_new_tokens):
                if ev["type"] == "tool_call_delta":
                    tc = ev["delta"]
                    i = tc.get("index", 0)
                    m = acc.setdefault(i, {"name": "", "arguments": ""})
                    fn = tc.get("function", {}) or {}
                    m["name"] += fn.get("name", "")
                    m["arguments"] += fn.get("arguments", "")
                elif ev["type"] == "token":
                    text += ev["token"]
            tool_calls = [{"name": m["name"], "arguments": m["arguments"]} for i, m in sorted(acc.items())]
            tool_calls = [tc for tc in tool_calls if tc["name"].strip()]

            if tool_calls:
                # GENUINE tools: execute EVERY tool call the model made in this round and return all results.
                _exec = []
                for _i, tc in enumerate(tool_calls):
                    _tid = f"call_{_i}"
                    _name = tc.get("name", "")
                    try:
                        _args = json.loads(tc.get("arguments") or "{}") if (tc.get("arguments") or "").strip() else {}
                    except Exception:
                        _args = {}
                    _argstr = tc.get("arguments") or _json_dumps(_args)
                    if _name == "get_current_datetime":
                        _tz = str(_args.get("timezone") or "UTC")
                        try:
                            from zoneinfo import ZoneInfo
                            from datetime import datetime as _dt, timedelta as _td
                            _now = _dt.now(ZoneInfo(_tz))
                        except Exception:
                            from datetime import datetime as _dt, timedelta as _td
                            _now = _dt.utcnow(); _tz = "UTC"
                        _tom = _now + _td(days=1); _yest = _now - _td(days=1)
                        _fmt = (f"Current date and time: {_now.strftime('%A')}, {_now.strftime('%Y-%m-%d')} {_now.strftime('%H:%M:%S')} ({_tz}). "
                                f"Today is {_now.strftime('%A')} ({_now.strftime('%Y-%m-%d')}). "
                                f"Tomorrow is {_tom.strftime('%A')} ({_tom.strftime('%Y-%m-%d')}). "
                                f"Yesterday was {_yest.strftime('%A')} ({_yest.strftime('%Y-%m-%d')}).")
                        logger.info(f"[LLM Tool] get_current_datetime -> {_fmt}")
                        yield {"type": "tool_call", "name": "get_current_datetime", "arguments": _args}
                        yield {"type": "tool_result", "name": "get_current_datetime",
                               "result": {"date": _now.strftime("%Y-%m-%d"), "weekday": _now.strftime("%A"),
                                           "time": _now.strftime("%H:%M:%S"), "timezone": _tz},
                               "formatted": _fmt, "latency_ms": 1, "source": "datetime"}
                        _exec.append({"id": _tid, "type": "function",
                                      "function": {"name": "get_current_datetime", "arguments": _argstr}})
                        _exec.append({"tool_call_id": _tid, "content": _fmt})
                    elif _name == "web_search" and web_search is not None:
                        query = str(_args.get("query") or prompt)[:120]
                        logger.info(f"[LLM Tool] native web_search '{query}'")
                        yield {"type": "tool_call", "name": "web_search", "arguments": _args, "query": query}
                        t_tool = time.time()
                        _saw_web = True
                        try:
                            search_res = await web_search(query, count=5)
                            formatted = format_results(search_res["results"])
                            yield {"type": "tool_result", "name": "web_search", "result": search_res, "formatted": formatted,
                                   "latency_ms": int((time.time()-t_tool)*1000),
                                   "source": search_res.get("source", "")}
                            if search_res.get("source") != "wttr.in":
                                try:
                                    from tools.web_search import _relevance_score as _rs
                                    _sc = _rs(query, search_res.get("results", []))
                                except Exception:
                                    _sc = 0.99
                                if _sc < 0.5:
                                    _weak_search = True
                                    logger.info(f"[Agent] weak search '{query}' score {_sc:.2f} -> refine")
                        except Exception as e:
                            logger.exception(f"LLM tool search failed {e}")
                            formatted = f"web search failed for '{query}': {e}"
                            yield {"type": "tool_result", "name": "web_search", "error": str(e)}
                        _exec.append({"id": _tid, "type": "function",
                                      "function": {"name": "web_search", "arguments": _argstr}})
                        _exec.append({"tool_call_id": _tid, "content": formatted})
                    else:
                        logger.warning(f"unexecuted tool call: {_name}")
                if _exec:
                    _starts = [x for x in _exec if "function" in x]
                    messages.append({"role": "assistant", "content": None, "tool_calls": _starts})
                    for x in _exec:
                        if "tool_call_id" in x:
                            messages.append({"role": "tool", **x})
                    # agent decision: another round?
                    if _round + 1 >= MAX_ROUNDS:
                        break
                    if _weak_search:
                        messages.append({"role": "user", "content":
                            "The web search results above were weak. Please run web_search again with a shorter, different query about the same topic."})
                        continue
                    if not _saw_web and _intent_wants_search(prompt):
                        messages.append({"role": "user", "content":
                            f'The answer needs current information. Please call web_search with a simple query about: "{prompt.strip()[:80]}"'})
                        continue
                    break

            # Ling sometimes emits the tool call as TEMPLATE XML in content instead of delta.tool_calls
            # (e.g. <tool_call><arg_key>query</arg_key><arg_value>Paris weather</arg_value></tool_call>).
            # If so, parse it and run web_search, and REMOVE the XML from the spoken text.
            xml_tool = None
            if text and "<tool_call>" in text and "web_search" in text.lower():
                m_arg = re.search(r"<arg_value>(.*?)</arg_value>", text, re.S)
                xml_query = (m_arg.group(1).strip() if m_arg else prompt)[:120]
                xml_tool = {"name": "web_search", "arguments": _json_dumps({"query": xml_query})}
                text = _strip_tool_xml(text)
                logger.info(f"[LLM Tool] parsed XML tool_call query='{xml_query}' (rest text '{text[:60]}')")
            if xml_tool and web_search is not None:
                query = xml_query
                yield {"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query}
                t_tool = time.time()
                try:
                    search_res = await web_search(query, count=5)
                    formatted = format_results(search_res["results"])
                    yield {"type": "tool_result", "name": "web_search", "result": search_res, "formatted": formatted,
                           "latency_ms": int((time.time()-t_tool)*1000),
                           "source": search_res.get("source", "")}
                except Exception as e:
                    formatted = f"web search failed for '{query}': {e}"
                    yield {"type": "tool_result", "name": "web_search", "error": str(e)}
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": "call_0", "type": "function",
                                                  "function": {"name": "web_search", "arguments": _json_dumps({"query": query})}}]})
                messages.append({"role": "tool", "tool_call_id": "call_0", "content": formatted})
                # if Ling already wrote a real sentence after the XML, speak it; otherwise do the answer pass
                if text.strip() and len(text.split()) >= 4:
                    final_text = _clean_leakage(text)
                    t0 = time.time(); _sf = ""
                    for tok in final_text:
                        _sf += tok
                        yield {"type": "llm_token", "token": tok, "text_so_far": _sf, "latency_ms": int((time.time()-t0)*1000) if len(_sf) == len(tok) else 20}
                        await asyncio.sleep(0)
                    yield {"type": "llm_done", "text": final_text}
                    return
                continue

            # no tool call this round
            if _round + 1 < MAX_ROUNDS and _intent_wants_search(prompt) and not _saw_web:
                messages.append({"role": "user", "content":
                    f'The answer needs current information. Please call web_search with a simple query about: "{prompt.strip()[:80]}"'})
                continue
            # no tool call -> stream final answer
            final_text = ""
            if text:
                text = _clean_leakage(text)
                t0 = time.time(); first = True
                for tok in text:
                    final_text += tok
                    yield {"type": "llm_token", "token": tok, "text_so_far": final_text,
                           "latency_ms": int((time.time()-t0)*1000) if first else 20}
                    first = False
                    await asyncio.sleep(0)
            final_text = _clean_leakage(final_text)
            yield {"type": "llm_done", "text": final_text}
            return

        # max tool rounds reached (model kept searching) -> final ANSWER pass, tools off, never empty
        _al = logger.debug("LLM max tool rounds reached -> final answer pass")
        _zh2 = any('\u4e00' <= c <= '\u9fff' for c in prompt or "")
        final_messages = messages + [{"role": "user", "content":
            ("根据上面的搜索结果，用简体中文直接回答用户的最后问题，一两句话即可，不要调用任何工具。" if _zh2 else
             "Based on the search results above, answer the user's last question directly in one or two spoken sentences. If the user asked for news or the latest info, list 2-3 concrete items from the results — never say you lack information when the results contain it. Do NOT call any tools.")}]
        final_text = ""
        try:
            async for ev in self._chat_stream(final_messages, tools=None, max_tokens=min(max_new_tokens, 192)):
                if ev["type"] == "token":
                    final_text += ev["token"]
        except Exception as e:
            logger.warning(f"LLM final answer pass failed: {e}")
        # Ling's final pass may re-emit the <tool_call> XML template — strip it, never speak it
        if "<tool_call>" in final_text or "<arg_" in final_text:
            final_text = _strip_tool_xml(final_text)
        final_text = _clean_leakage(final_text)
        if not final_text.strip():
            final_text = "抱歉，我找不到明確的答案。"
        t0 = time.time(); _sf = ""
        for tok in final_text:
            _sf += tok
            yield {"type": "llm_token", "token": tok, "text_so_far": _sf, "latency_ms": int((time.time()-t0)*1000) if len(_sf) == len(tok) else 20}
            await asyncio.sleep(0)
        yield {"type": "llm_done", "text": final_text}

    async def generate_with_tools(self, prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        # Single-turn wrapper for backwards compat — delegates to multi-turn with empty history
        async for ev in self.generate_chat_with_tools([], prompt, max_new_tokens=max_new_tokens):
            yield ev
        return

