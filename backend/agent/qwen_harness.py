"""
Qwen-Agent harness for Qwen3.5-2B voice chat.
Wraps Qwen-Agent's Assistant with our SearXNG + wttr + granite tools.
"""
import os
os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = "3"  # voice: cap at 3 LLM calls (1 tool + final) to prevent 8× loop
import asyncio
from loguru import logger
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import json

from agent._shared import set_emit_target, emit as _emit, agent_call_lock

# Import our search logic
try:
    from tools.web_search import web_search, format_results
    from tools.web_search import _wttr_weather
except ImportError:
    from web_search import web_search, format_results
    _wttr_weather = None

import asyncio as _asyncio

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
    # Prefer the live alias llm_manager actually has loaded (kept in sync across
    # POST /api/model switches) over a static env var, so a freshly-reset agent
    # (see reset_agent() below) picks up whichever model is currently running
    # instead of whatever was true at process startup.
    _model = os.getenv("LLM_MODEL_ID", "qwen3.5-2b")
    try:
        from llm_manager import llm_manager as _llm_mgr
        if _llm_mgr.current_alias:
            _model = _llm_mgr.current_alias
    except Exception:
        pass
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
        system_message="You are a helpful voice assistant. For weather use get_weather(location, date) — 1 call max. For general search use web_search — 1 call max with 3-8 words, then answer. For date/time use get_current_datetime — 1 call max. Never do more than 2 tool calls per turn. Always default to Traditional Chinese (Taiwan usage, 繁體中文) regardless of what language the question was asked in — only keep English for proper nouns, technical terms, or vocabulary that doesn't translate well; never answer a whole sentence in Simplified Chinese or English."
    )

_agent = None
def _get_agent():
    global _agent
    if _agent is None:
        _agent = _make_agent()
    return _agent

def reset_agent():
    """Drop the cached Assistant so the next call rebuilds it — needed after
    llm_manager.switch_to() changes which model llama-server serves, since
    _make_agent() bakes the model alias in at construction time and _agent is
    otherwise a permanent singleton for the process lifetime. Called from
    app.py's POST /api/model handler."""
    global _agent
    _agent = None

async def run_agent_task(task: str, event_q=None) -> str:
    loop = _asyncio.get_running_loop()
    set_emit_target(loop, event_q)
    agent = _get_agent()
    def _run():
        try:
            # Fast path for plain greetings (zh-TW 餵你好 etc., en hi/hello) — no tool call
            import re
            m = re.search(r"Current question:\s*(.*)", task, re.S)
            _cur = m.group(1).strip() if m else task.strip()
            _tl = _cur.lower()
            if len(_tl) < 30 and re.search(r'(hi|hello|hey|hola|bonjour|你好|您好|哈囉|嗨|餵|喂|欸|嘿)', _tl, re.I) and not re.search(r'(weather|news|search|find|what is|who is|how|can you|please|today|tomorrow|星期|几号|幾號|几点|幾點|天气|天氣|新聞|頭條|分機|電話)', _tl, re.I):
                if len(_tl) < 15 or re.match(r'^\s*[\w\s]*(你好|您好|哈囉|嗨|餵|喂|hi|hello|hey)\b', _tl, re.I):
                    # zh-TW default: greet in Traditional Chinese even for an English "hello"
                    # — matches this app's confirmed default-language behavior (LANG_HINT).
                    return "你好！有什麼可以幫你的？"
            messages = [{'role': 'user', 'content': task}]
            all_resps = []
            last_resp = None
            with agent_call_lock:
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
            return "抱歉，我找不到相關的答案。"
        except Exception as e:
            # This return value is spoken via TTS (generate_chat_with_tools tokenizes it
            # char-by-char with no filtering for an "error"-looking string) — a raw
            # exception message here (e.g. an httpx connection error from a model switch
            # killing the server mid-request) would get read aloud verbatim. Log the real
            # detail server-side; return a clean, generic fallback for the user to hear.
            logger.exception(f"qwen agent failed: {e}")
            return "抱歉，這個問題我暫時無法回答，請再試一次。"
    return await asyncio.to_thread(_run)
