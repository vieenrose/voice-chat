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

SYSTEM_PROMPT = "You are a helpful voice assistant with web search. Keep replies concise, conversational, under 80 words, speak naturally for voice chat. For ANY question about current events, news, weather, real-time info, or specific regional events (e.g., '今天台湾有什么重大事件', 'latest news'), you MUST use web_search tool to get the latest information — never refuse or say you cannot provide real-time info. Use search results to answer. For questions about today's date, weekday, or current time (今天/星期几/几点), you MUST call the get_current_datetime tool instead of guessing. For ANY weather/forecast question (including 明天/後天/next week) you MUST call web_search — the search engine returns the forecast for the requested day, so do NOT call get_current_datetime for weather. For date/time questions call get_current_datetime. When both are needed, call both in the same turn. Always answer using the tool RESULTS verbatim (weekday, date, temperatures) — never from memory. Always respond in the user's language (Chinese for Chinese queries, English for English queries)."

# Heuristic for fast tool trigger — bilingual (en/zh) - includes Chinese triggers for Taiwan/news
# (heuristic removed — tool calls are model-driven/native only)

class LingStreaming:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-2B-MTP-GGUF", api_base: str = "http://127.0.0.1:11436/v1", mock: bool = False, device: str = "cuda", model_name: str = "qwen3.5-2b", **kwargs):
        self.model_id = model_id
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.backend = f"llm-gguf:{model_name}"
        self.mock = mock
        self.client = httpx.AsyncClient(timeout=60.0)
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
            "temperature": 0.7,
            "top_p": 0.9,
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
            fallback = "Sorry, Ling server had an error, but I'm still here via mock."
            text_so_far = ""
            for tok in fallback.split(" "):
                await asyncio.sleep(0.02)
                text_so_far += (" " if text_so_far else "") + tok
                yield {"type": "llm_token", "token": " "+tok, "text_so_far": text_so_far, "latency_ms": 50}
            yield {"type": "llm_done", "text": text_so_far}

    async def generate_chat_with_tools(self, history: List[Dict], prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
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

        for _round in range(1):  # ONE native tool round (model decides query + whether to search) — then answer. No refine-re-search doubling.
            # — run model with tools (native) —
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
                        try:
                            search_res = await web_search(query, count=5)
                            formatted = format_results(search_res["results"])
                            yield {"type": "tool_result", "name": "web_search", "result": search_res, "formatted": formatted,
                                   "latency_ms": int((time.time()-t_tool)*1000),
                                   "source": search_res.get("source", "")}
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
                    continue

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
                    final_text = text
                    t0 = time.time(); _sf = ""
                    for tok in final_text:
                        _sf += tok
                        yield {"type": "llm_token", "token": tok, "text_so_far": _sf, "latency_ms": int((time.time()-t0)*1000) if len(_sf) == len(tok) else 20}
                        await asyncio.sleep(0)
                    yield {"type": "llm_done", "text": final_text}
                    return
                continue

            # no tool call -> stream final answer
            final_text = ""
            if text:
                t0 = time.time(); first = True
                for tok in text:
                    final_text += tok
                    yield {"type": "llm_token", "token": tok, "text_so_far": final_text,
                           "latency_ms": int((time.time()-t0)*1000) if first else 20}
                    first = False
                    await asyncio.sleep(0)
            yield {"type": "llm_done", "text": final_text}
            return

        # max tool rounds reached (model kept searching) -> final ANSWER pass, tools off, never empty
        _al = logger.debug("LLM max tool rounds reached -> final answer pass")
        _zh2 = any('\u4e00' <= c <= '\u9fff' for c in prompt or "")
        final_messages = messages + [{"role": "user", "content":
            ("根据上面的搜索结果，用简体中文直接回答用户的最后问题，一两句话即可，不要调用任何工具。" if _zh2 else
             "Based on the search results above, answer the user's last question directly in one or two spoken sentences. Do NOT call any tools.")}]
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
        if not final_text.strip():
            final_text = "Sorry, I could not find a clear answer to that."
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

