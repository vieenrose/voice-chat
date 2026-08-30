"""
Qwen-Agent harness for Qwen3.5-2B voice chat.
Wraps Qwen-Agent's Assistant with our SearXNG + wttr + granite tools.
"""
import os
os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = "3"  # voice: cap at 3 LLM calls (1 tool + final) to prevent 8× loop
import asyncio
import contextvars
from loguru import logger
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import json

# Import our search logic
try:
    from tools.web_search import web_search, format_results
    from tools.web_search import _wttr_weather
except ImportError:
    from web_search import web_search, format_results
    _wttr_weather = None

# Shared event emitter for WS streaming
import asyncio as _asyncio
# Per-call (loop, queue) target, NOT a mutable global: _agent below is a single shared
# Assistant reused across every session/turn, so two calls to run_agent_task() can be
# in flight at once (two sessions chatting concurrently, or an old barge-in'd turn's
# background thread still finishing when a new turn starts). A plain "last attach() wins"
# singleton would silently deliver one turn's tool_call/tool_result events into another
# turn's WS queue. asyncio.to_thread() copies the current contextvars context into the
# worker thread, so setting this per-call in run_agent_task (before to_thread) and
# reading it from the tool .call() methods (which execute on that thread) keeps each
# call's events routed to its own queue.
_emit_ctx: "contextvars.ContextVar[tuple | None]" = contextvars.ContextVar("_emit_ctx", default=None)

def _emit(ev: dict):
    ctx = _emit_ctx.get()
    if ctx:
        loop, q = ctx
        if loop and q:
            loop.call_soon_threadsafe(q.put_nowait, ev)

@register_tool('web_search', allow_overwrite=True)
class QwenWebSearch(BaseTool):
    name = 'web_search'
    description = 'Search the web for current information (weather, news, facts).'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'search query 3-8 words'}
        },
        'required': ['query']
    }
    def call(self, params, **kwargs):
        import asyncio, json as _json
        # Qwen-Agent sometimes passes JSON string '{"query": "..."}' or dict
        if isinstance(params, str):
            s = params.strip()
            if s.startswith('{'):
                try:
                    d = _json.loads(s)
                    if isinstance(d, dict) and 'query' in d:
                        query = str(d['query']).strip()
                    else:
                        query = s
                except:
                    query = s
            else:
                query = s
        else:
            query = params.get('query', '') if isinstance(params, dict) else ''
        _emit({"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query})
        # Use sync version for Qwen-Agent (which is sync)
        from tools.web_search import web_search_sync, format_results
        res = web_search_sync(query, count=5)
        formatted = format_results(res.get("results", [])) if res.get("results") else "No results"
        _emit({"type": "tool_result", "name": "web_search", "result": res, "formatted": formatted, "latency_ms": res.get("latency_ms", 0), "source": res.get("source","")})
        return formatted

@register_tool('get_weather', allow_overwrite=True)
class QwenGetWeather(BaseTool):
    name = 'get_weather'
    description = 'Get weather forecast for a location (today/tomorrow/day_after_tomorrow). Wraps wttr.in + SearXNG.'
    parameters = {
        'type': 'object',
        'properties': {
            'location': {'type': 'string', 'description': "city, e.g. 'Paris', '台中'"},
            'date': {'type': 'string', 'description': 'today, tomorrow, or day_after_tomorrow'}
        },
        'required': ['location']
    }
    def call(self, params, **kwargs):
        import json
        if isinstance(params, str):
            try: params = json.loads(params)
            except: params = {}
        loc = params.get('location','') if isinstance(params, dict) else ''
        date = params.get('date','today') if isinstance(params, dict) else 'today'
        dmap = {'today':'', 'tomorrow':'明天', 'day_after_tomorrow':'後天'}
        q = f"{loc} {dmap.get(date,'')} 天气".strip() if any('\u4e00' <= c <= '\u9fff' for c in loc) else f"weather in {loc} {date}".strip()
        _emit({"type": "tool_call", "name": "get_weather", "arguments": {"location": loc, "date": date}, "query": q})
        from tools.web_search import web_search_sync, format_results
        res = web_search_sync(q, count=5)
        formatted = format_results(res.get("results", [])) if res.get("results") else "No weather data"
        _emit({"type": "tool_result", "name": "get_weather", "result": res, "formatted": formatted, "latency_ms": res.get("latency_ms",0), "source": res.get("source","")})
        return formatted

@register_tool('get_current_datetime', allow_overwrite=True)
class QwenDateTime(BaseTool):
    name = 'get_current_datetime'
    description = 'Get current date and time (UTC). Use for today/weekday/time questions.'
    parameters = {
        'type': 'object',
        'properties': {
            'timezone': {'type': 'string', 'description': 'IANA timezone, default UTC'}
        },
        'required': []
    }
    def call(self, params, **kwargs):
        import json
        if isinstance(params, str):
            try: params = json.loads(params)
            except: params = {}
        tz = params.get('timezone', 'UTC') if isinstance(params, dict) else 'UTC'
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        try:
            now = datetime.now(ZoneInfo(tz))
        except:
            now = datetime.utcnow()
            tz = "UTC"
        tom = now + timedelta(days=1)
        fmt = f"Current: {now.strftime('%A %Y-%m-%d %H:%M:%S')} ({tz}). Today {now.strftime('%A')} {now.strftime('%Y-%m-%d')}, Tomorrow {tom.strftime('%A')} {tom.strftime('%Y-%m-%d')}."
        _emit({"type": "tool_call", "name": "get_current_datetime", "arguments": {"timezone": tz}})
        _emit({"type": "tool_result", "name": "get_current_datetime", "result": {"date": now.strftime("%Y-%m-%d")}, "formatted": fmt, "latency_ms": 1, "source": "datetime"})
        return fmt

def _make_agent():
    import os
    _base = os.getenv("LLM_API_BASE", "http://127.0.0.1:11435/v1")
    _model = os.getenv("LLM_MODEL_ID", "qwen3.5-2b")
    llm_cfg = {
        'model': _model,
        'model_server': _base,
        'api_key': 'none',
        'generate_cfg': {
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'chat_template_kwargs': {'enable_thinking': True},
        },
        # Thinking enabled: reasoning_content will be in delta but we filter it in run_agent_task (only final assistant content)
        'enable_thinking': True
    }
    return Assistant(
        llm=llm_cfg,
        function_list=['web_search', 'get_weather', 'get_current_datetime'],
        system_message="You are a helpful voice assistant. For weather use get_weather(location, date) — 1 call max. For general search use web_search — 1 call max with 3-8 words, then answer. For date/time use get_current_datetime — 1 call max. Never do more than 2 tool calls per turn. Always answer in the user's language."
    )

_agent = None
def _get_agent():
    global _agent
    if _agent is None:
        _agent = _make_agent()
    return _agent

async def run_agent_task(task: str, event_q=None) -> str:
    loop = _asyncio.get_running_loop()
    _emit_ctx.set((loop, event_q))
    agent = _get_agent()
    def _run():
        try:
            # Fast path for plain greetings (zh-TW 餵你好 etc., en hi/hello) — no tool call
            import re
            m = re.search(r"Current question:\s*(.*)", task, re.S)
            _cur = m.group(1).strip() if m else task.strip()
            _tl = _cur.lower()
            if len(_tl) < 30 and re.search(r'(hi|hello|hey|hola|bonjour|你好|您好|哈囉|嗨|餵|喂|欸|嘿)', _tl, re.I) and not re.search(r'(weather|news|search|find|what is|who is|how|can you|please|today|tomorrow|星期|几号|天气|新聞|頭條|天氣|分機|電話)', _tl, re.I):
                if len(_tl) < 15 or re.match(r'^\s*[\w\s]*(你好|您好|哈囉|嗨|餵|喂|hi|hello|hey)\b', _tl, re.I):
                    return "你好！有什麼可以幫你的？" if re.search(r'[\u4e00-\u9fff]', _cur) else "Hello! How can I help you today?"
            messages = [{'role': 'user', 'content': task}]
            all_resps = []
            last_resp = None
            for resp in agent.run(messages=messages):
                all_resps.append(resp)
                last_resp = resp
            # Collect all assistant messages with non-empty content from entire run (filter thinking leak)
            candidates_all = []
            for resp in all_resps:
                lst = resp if isinstance(resp, list) else [resp]
                for m in lst:
                    if isinstance(m, dict) and m.get('role') == 'assistant':
                        c = m.get('content')
                        # Filter: content must be non-empty and not just reasoning (reasoning_content is separate)
                        if isinstance(c, str) and c.strip() and len(c.strip()) > 2:
                            # Skip if it's just the reasoning dump (empty content with reasoning_content)
                            if not c.strip().startswith("["):
                                candidates_all.append(m)
                        elif isinstance(c, list):
                            txt = "".join(b.get('text','') if isinstance(b, dict) else str(b) for b in c)
                            if txt.strip():
                                candidates_all.append({"role": "assistant", "content": txt})
            if candidates_all:
                # Return the last non-empty assistant content (final answer, not intermediate reasoning)
                return candidates_all[-1].get('content','')
            # Fallback to memory (also filter empty)
            if hasattr(agent, 'memory') and agent.memory:
                hist = agent.memory.get_history() if hasattr(agent.memory, 'get_history') else []
                for m in reversed(hist):
                    if m.get('role') == 'assistant' and m.get('content'):
                        c = m['content']
                        if isinstance(c, str) and c.strip() and len(c.strip()) > 2 and not c.strip().startswith("["):
                            return c
            return "Sorry, I could not find an answer."
        except Exception as e:
            logger.exception(f"qwen agent failed: {e}")
            return f"(agent error: {e})"
    return await asyncio.to_thread(_run)
