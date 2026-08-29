"""
Qwen-Agent harness for Qwen3.5-2B voice chat.
Wraps Qwen-Agent's Assistant with our SearXNG + wttr + granite tools.
"""
import asyncio
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
class _EventEmitter:
    def __init__(self):
        self._loop = None
        self._q = None
    def attach(self, loop, q):
        self._loop = loop
        self._q = q
    def emit(self, ev: dict):
        if self._loop and self._q:
            self._loop.call_soon_threadsafe(self._q.put_nowait, ev)
_EMIT = _EventEmitter()

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
        _EMIT.emit({"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query})
        # Use sync version for Qwen-Agent (which is sync)
        from tools.web_search import web_search_sync, format_results
        res = web_search_sync(query, count=5)
        formatted = format_results(res.get("results", [])) if res.get("results") else "No results"
        _EMIT.emit({"type": "tool_result", "name": "web_search", "result": res, "formatted": formatted, "latency_ms": res.get("latency_ms", 0), "source": res.get("source","")})
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
        _EMIT.emit({"type": "tool_call", "name": "get_weather", "arguments": {"location": loc, "date": date}, "query": q})
        from tools.web_search import web_search_sync, format_results
        res = web_search_sync(q, count=5)
        formatted = format_results(res.get("results", [])) if res.get("results") else "No weather data"
        _EMIT.emit({"type": "tool_result", "name": "get_weather", "result": res, "formatted": formatted, "latency_ms": res.get("latency_ms",0), "source": res.get("source","")})
        return formatted

@register_tool('get_extension', allow_overwrite=True)
class QwenGetExtension(BaseTool):
    name = 'get_extension'
    description = '查詢公司通訊錄中某人的分機號碼（支援中文姓名，自動處理拼音/同音字/ASR辨識錯誤的模糊搜尋，100人資料庫）。請用此工具查詢任何人的分機，不要用 web_search。'
    parameters = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'description': '要查詢的人名（中文2-4字，可能有錯字或同音字，如 王小名/汪小明 會自動找到 王小明）'}
        },
        'required': ['name']
    }
    def call(self, params, **kwargs):
        import json as _json
        if isinstance(params, str):
            try:
                d = _json.loads(params)
                name = d.get('name','') if isinstance(d, dict) else params
            except:
                name = params
        else:
            name = params.get('name','') if isinstance(params, dict) else ''
        name = str(name).strip()
        _EMIT.emit({"type": "tool_call", "name": "get_extension", "arguments": {"name": name}, "query": name})
        from tools.contact_db import get_extension
        res = get_extension(name)
        msg = res.get("message", "")
        # Add candidates detail if any
        if res.get("candidates"):
            msg += " " + " ".join([f"{c['name']}({c['ext']})" for c in res["candidates"][:3]])
        _EMIT.emit({"type": "tool_result", "name": "get_extension", "result": res, "formatted": msg, "latency_ms": 5, "source": "contact_db"})
        return msg

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
        _EMIT.emit({"type": "tool_call", "name": "get_current_datetime", "arguments": {"timezone": tz}})
        _EMIT.emit({"type": "tool_result", "name": "get_current_datetime", "result": {"date": now.strftime("%Y-%m-%d")}, "formatted": fmt, "latency_ms": 1, "source": "datetime"})
        return fmt

def _make_agent():
    import os
    _base = os.getenv("LLM_API_BASE", "http://127.0.0.1:11435/v1")
    _model = os.getenv("LLM_MODEL_ID", "qwen3.5-0.8b")
    llm_cfg = {
        'model': _model,
        'model_server': _base,
        'api_key': 'none',
        'generate_cfg': {
            'max_tokens': 1024,
            'temperature': 0.7,
            'top_p': 0.9,
            'chat_template_kwargs': {'enable_thinking': True},
        },
        # Thinking enabled: reasoning_content will be in delta but we filter it in run_agent_task (only final assistant content)
        'enable_thinking': True
    }
    return Assistant(
        llm=llm_cfg,
        function_list=['web_search', 'get_weather', 'get_current_datetime', 'get_extension'],
        system_message="You are a helpful voice assistant (zh-TW). For weather use get_weather(location, date). For general search use web_search. For date/time use get_current_datetime. For 任何人的分機/聯絡人/電話查詢 一定要用 get_extension(name)（支援模糊拼音/同音字/ASR錯誤，如 王小名會找到王小明），不要用 web_search。Always answer in the user's language (zh-TW用繁體中文)."
    )

_agent = None
def _get_agent():
    global _agent
    if _agent is None:
        _agent = _make_agent()
    return _agent

async def run_agent_task(task: str, event_q=None) -> str:
    loop = _asyncio.get_running_loop()
    _EMIT.attach(loop, event_q)
    agent = _get_agent()
    def _run():
        try:
            messages = [{'role': 'user', 'content': task}]
            last_resp = None
            for resp in agent.run(messages=messages):
                last_resp = resp
            # Try to extract from last_resp first (most reliable)
            if last_resp:
                # last_resp can be List[Dict] or Dict
                candidates = last_resp if isinstance(last_resp, list) else [last_resp]
                for m in reversed(candidates):
                    if isinstance(m, dict) and m.get('role') == 'assistant' and m.get('content'):
                        c = m['content']
                        if isinstance(c, str) and c.strip():
                            return c
                        if isinstance(c, list):
                            # content as list of blocks
                            txt = "".join(b.get('text','') if isinstance(b, dict) else str(b) for b in c)
                            if txt.strip(): return txt
            # Fallback to memory
            if hasattr(agent, 'memory') and agent.memory:
                hist = agent.memory.get_history() if hasattr(agent.memory, 'get_history') else []
                for m in reversed(hist):
                    if m.get('role') == 'assistant' and m.get('content'):
                        c = m['content']
                        if isinstance(c, str) and c.strip() and 'could not find' not in c.lower():
                            return c
                        if isinstance(c, str) and c.strip():
                            return c
            # If last_resp was list but not captured, try stringifying
            if last_resp:
                return str(last_resp)[:2000]
            return "Sorry, I could not find an answer."
        except Exception as e:
            logger.exception(f"qwen agent failed: {e}")
            return f"(agent error: {e})"
    return await asyncio.to_thread(_run)
