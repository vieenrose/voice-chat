"""
Agent harness (smolagents 1.26) for the voice chat.

ToolCallingAgent + OpenAI-compatible llama-server (granite-4.2-3b).
Tools: web_search (SearXNG + wttr.in), get_current_datetime.
Runs the agent in a worker thread with an OpenAI-compatible model; tool
executions emit {tool_call, tool_result} events back to the async caller.
"""
import asyncio
import importlib  # noqa: F401  (py3.14: makes importlib.resources visible)
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel

from agent._shared import set_emit_target, emit as _emit, agent_call_lock

import os
API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:11435/v1")
MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen3.5-2b")
MAX_STEPS = 2  # voice: 1 tool + final answer; keeps e2e <4s


class GraniteModel(OpenAIServerModel):
    def __init__(self, *args, **kwargs):
        # Move chat_template_kwargs into extra_body for openai>=2.28 compatibility
        ct = kwargs.pop("chat_template_kwargs", None)
        if ct is not None:
            eb = kwargs.get("extra_body") or {}
            eb = dict(eb)
            eb["chat_template_kwargs"] = ct
            kwargs["extra_body"] = eb
        super().__init__(*args, **kwargs)

    def _prepare_completion_kwargs(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, custom_role_conversions=None, convert_images_to_image_urls=False, tool_choice=None, **kwargs):
        if tool_choice is None:
            tool_choice = "auto"
        # Ensure chat_template_kwargs is sent via extra_body, not top-level (openai>=2.28 rejects top-level)
        extra = dict(kwargs.pop("extra_body", {}) or {})
        # Also check self.kwargs for extra_body from __init__
        if "chat_template_kwargs" not in extra:
            self_extra = {}
            if hasattr(self, "kwargs") and isinstance(self.kwargs.get("extra_body"), dict):
                self_extra = self.kwargs.get("extra_body", {})
            if "chat_template_kwargs" in self_extra:
                extra["chat_template_kwargs"] = self_extra["chat_template_kwargs"]
            else:
                extra["chat_template_kwargs"] = {"enable_thinking": False}
        # Also handle legacy top-level chat_template_kwargs if caller passed it directly
        if "chat_template_kwargs" in kwargs:
            extra["chat_template_kwargs"] = kwargs.pop("chat_template_kwargs")
        kwargs["extra_body"] = extra
        return super()._prepare_completion_kwargs(messages, stop_sequences, response_format, tools_to_call_from, custom_role_conversions, convert_images_to_image_urls, tool_choice, **kwargs)


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for current info (news, facts, people, definitions). For weather, use get_weather instead."
    inputs = {
        "query": {"type": "string", "description": "search query 3-8 words (or JSON list of queries for parallel search)"},
        "count": {"type": "integer", "description": "number of results", "nullable": True, "default": 5},
    }
    output_type = "string"

    def forward(self, query: str, count: int = 5) -> str:
        # Enforce get_weather for weather queries
        if any(w in query.lower() for w in ["weather", "forecast", "temperature", "天气", "天氣", "气温", "降雨"]):
            return "ERROR: Use get_weather(location, date) for weather queries, not web_search."
        from tools.web_search import web_search_sync, format_results
        import json as _json
        # Handle JSON string from some models: '{"query": "..."}' or '["q1","q2"]'
        _q = query.strip() if isinstance(query, str) else query
        if isinstance(_q, str) and _q.startswith('{'):
            try:
                d = _json.loads(_q)
                if isinstance(d, dict) and 'query' in d:
                    query = str(d['query']).strip()
                    _q = query
            except Exception:
                from tools.web_search import repair_truncated_json_query
                query = repair_truncated_json_query(_q)
                _q = query
        # Apodex-style multi-query: accept JSON list or single string
        queries = query
        if isinstance(_q, str) and _q.strip().startswith("["):
            try:
                parsed = _json.loads(_q)
                if isinstance(parsed, list):
                    queries = parsed
            except Exception:
                pass
        if isinstance(queries, list):
            # Parallel multi-query (Apodex harness style) — dedup by URL
            all_results = []
            seen = set()
            for q in queries:
                if not q or not str(q).strip():
                    continue
                _emit({"type": "tool_call", "name": "web_search", "arguments": {"query": str(q)}, "query": str(q)})
                t0 = time.time()
                res = web_search_sync(str(q).strip(), count=count)
                results = res.get("results") or []
                for r in results:
                    url = r.get("url","")
                    if url and url in seen:
                        continue
                    seen.add(url)
                    all_results.append(r)
                _emit({"type": "tool_result", "name": "web_search", "result": {"results": results, "source": res.get("source","")},
                            "formatted": format_results(results) if results else "No results", "latency_ms": int((time.time()-t0)*1000), "source": res.get("source","")})
                if len(all_results) >= count * 2:
                    break
            formatted = format_results(all_results[:count*2]) if all_results else "No search results found."
            return formatted
        # Single query
        t0 = time.time()
        _emit({"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query})
        res = web_search_sync(query, count=count)
        results = res.get("results") or []
        formatted = format_results(results) if results else "No results found."
        _emit({"type": "tool_result", "name": "web_search", "result": res,
                    "formatted": formatted, "latency_ms": int((time.time()-t0)*1000),
                    "source": res.get("source", "")})
        return formatted


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get weather forecast for ANY weather question. You MUST use this for weather, not web_search. Provide location and date."
    inputs = {
        "location": {"type": "string", "description": "city or location, e.g. 'Paris', '台中', 'Tokyo'"},
        "date": {"type": "string", "description": "today, tomorrow, or day_after_tomorrow", "nullable": True, "default": "today"}
    }
    output_type = "string"

    def forward(self, location: str, date: str = "today") -> str:
        from tools.web_search import web_search_sync, format_results
        # Normalize date -> query phrase for wttr day selection
        date_map = {"today": "", "tomorrow": "明天", "day_after_tomorrow": "後天"}
        dphrase = date_map.get(date, "")
        q = f"{location} {dphrase} 天气".strip() if any('\u4e00' <= c <= '\u9fff' for c in location) else f"weather in {location} {date}".strip()
        _emit({"type": "tool_call", "name": "get_weather", "arguments": {"location": location, "date": date}, "query": q})
        t0 = time.time()
        # Directly call web_search which already handles wttr.in for weather queries
        res = web_search_sync(q, count=5)
        formatted = format_results(res.get("results", [])) if res.get("results") else "No weather data"
        _emit({"type": "tool_result", "name": "get_weather", "result": res, "formatted": formatted, "latency_ms": int((time.time()-t0)*1000), "source": res.get("source","")})
        return formatted

class DateTimeTool(Tool):
    name = "get_current_datetime"
    description = "Get the current date and time (UTC). Use ONLY for questions about what day/date/time it is (星期几/几号/几点/what day/date). NEVER for weather."
    inputs = {"timezone": {"type": "string", "description": "optional IANA timezone (e.g. Asia/Taipei); default UTC", "nullable": True}}
    output_type = "string"

    def forward(self, timezone: str | None = None) -> str:
        _emit({"type": "tool_call", "name": "get_current_datetime", "arguments": {"timezone": timezone or "UTC"}})
        tz = timezone or "UTC"
        try:
            now = datetime.now(ZoneInfo(tz))
        except Exception:
            now = datetime.utcnow()
            tz = "UTC"
        tom = now + timedelta(days=1)
        yest = now - timedelta(days=1)
        fmt = (f"Current date and time: {now.strftime('%A')}, {now.strftime('%Y-%m-%d')} {now.strftime('%H:%M:%S')} ({tz}). "
               f"Today is {now.strftime('%A')} ({now.strftime('%Y-%m-%d')}). "
               f"Tomorrow is {tom.strftime('%A')} ({tom.strftime('%Y-%m-%d')}). "
               f"Yesterday was {yest.strftime('%A')} ({yest.strftime('%Y-%m-%d')}).")
        _emit({"type": "tool_result", "name": "get_current_datetime",
                    "result": {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A"),
                               "time": now.strftime("%H:%M:%S"), "timezone": tz},
                    "formatted": fmt, "latency_ms": 1, "source": "datetime"})
        return fmt


class _Agent:
    def __init__(self):
        # Prefer the live alias llm_manager actually has loaded over the MODEL_ID
        # constant frozen at import time, so a rebuilt agent (see reset_agent())
        # after a POST /api/model switch points at the model that's really running.
        model_id = MODEL_ID
        try:
            from llm_manager import llm_manager as _llm_mgr
            if _llm_mgr.current_alias:
                model_id = _llm_mgr.current_alias
        except Exception:
            pass
        self.model = GraniteModel(model_id=model_id, api_base=API_BASE, api_key="none",
                                  temperature=1.0, top_p=0.95,
                                  chat_template_kwargs={"enable_thinking": False})
        self.agent = ToolCallingAgent(tools=[WebSearchTool(), GetWeatherTool(), DateTimeTool()], model=self.model, max_steps=MAX_STEPS, verbosity_level=0, add_base_tools=False, instructions="CRITICAL: For weather use get_weather(location, date) — never web_search. For general search use web_search. For date/time use get_current_datetime. Always default to Traditional Chinese (Taiwan usage, 繁體中文) regardless of what language the question was asked in — only keep English for proper nouns, technical terms, or vocabulary that doesn't translate well; never answer a whole sentence in Simplified Chinese or English. When user says 'Search ...' call web_search. For greetings respond directly.")

    def run(self, task: str, history: list | None = None) -> str:
        # No greeting fast-path (see the identical note in qwen_harness.py): the
        # canned reply short-circuited the model and skewed every benchmark.
        hist = [{"role": m["role"], "content": m["content"]} for m in (history or [])
                if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
        with agent_call_lock:
            if hist:
                # Real multi-turn messages where the pinned smolagents supports them,
                # else a digest — either way better than dropping history entirely.
                try:
                    out = self.agent.run(messages=hist + [{"role": "user", "content": task}], reset=True)
                except (TypeError, ValueError) as e:
                    logger.debug(f"smolagents messages= unsupported ({e}); digest fallback")
                    digest = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in hist[-6:])
                    out = self.agent.run(f"Conversation history:\n{digest}\n\nCurrent question: {task}", reset=True)
            else:
                out = self.agent.run(task, reset=True)
        return out if isinstance(out, str) else (out.final_answer if hasattr(out, "final_answer") else str(out))


_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = _Agent()
    return _AGENT

def reset_agent():
    """Drop the cached _Agent so the next call rebuilds it against whichever
    model llm_manager.switch_to() just loaded — see the identical note in
    agent/qwen_harness.py. Called from app.py's POST /api/model handler."""
    global _AGENT
    _AGENT = None


def agent_alive() -> bool:
    try:
        import httpx
        with httpx.Client(timeout=3.0) as c:
            return c.get(f"{API_BASE}/models").status_code == 200
    except Exception:
        return False


async def run_agent_task(task: str, event_q: asyncio.Queue | None = None, history: list | None = None) -> str:
    """Run the smolagents loop for one user prompt; may emit tool events into event_q.

    `history` (real role/content turns) is forwarded to _Agent.run; see the note in
    agent/qwen_harness.py. Tool-call events are emitted by the Tool.forward()
    methods; final-answer streaming (llm_delta) is not available here because
    smolagents' thread-run has no incremental hook — the consumer replays the final
    text in that case, exactly as before.
    """
    loop = asyncio.get_running_loop()
    set_emit_target(loop, event_q)

    def _worker():
        agent = _get_agent()
        try:
            return agent.run(task, history=history)
        except Exception as e:
            # Spoken via TTS with no filtering for an "error"-looking string (see the
            # identical note in qwen_harness.py) — log the real detail, return a clean
            # generic fallback instead of the raw exception text.
            logger.exception(f"agent.run failed: {e}")
            return "抱歉，這個問題我暫時無法回答，請再試一次。"

    return await asyncio.to_thread(_worker)
