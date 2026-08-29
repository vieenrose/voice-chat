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
        import asyncio
        query = params if isinstance(params, str) else params.get('query', '')
        _EMIT.emit({"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query})
        # Use sync version for Qwen-Agent (which is sync)
        from tools.web_search import web_search_sync, format_results
        res = web_search_sync(query, count=5)
        formatted = format_results(res.get("results", [])) if res.get("results") else "No results"
        _EMIT.emit({"type": "tool_result", "name": "web_search", "result": res, "formatted": formatted, "latency_ms": res.get("latency_ms", 0), "source": res.get("source","")})
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
        _EMIT.emit({"type": "tool_call", "name": "get_current_datetime", "arguments": {"timezone": tz}})
        _EMIT.emit({"type": "tool_result", "name": "get_current_datetime", "result": {"date": now.strftime("%Y-%m-%d")}, "formatted": fmt, "latency_ms": 1, "source": "datetime"})
        return fmt

def _make_agent():
    llm_cfg = {
        'model': 'qwen3.5-2b',
        'model_server': 'http://127.0.0.1:11435/v1',
        'api_key': 'none',
        'generate_cfg': {
            'max_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'chat_template_kwargs': {'enable_thinking': False}
        }
    }
    return Assistant(
        llm=llm_cfg,
        function_list=['web_search', 'get_current_datetime'],
        system_message="You are a helpful voice assistant. Be concise and natural. For weather/news/facts, call web_search. For date/time, call get_current_datetime. Always answer in the user's language."
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
            for resp in agent.run(messages=messages):
                pass
            # Get last response
            if hasattr(agent, 'memory') and agent.memory:
                # Try to get last assistant message
                for m in reversed(agent.memory.get_history() if hasattr(agent.memory, 'get_history') else []):
                    if m.get('role') == 'assistant' and m.get('content'):
                        return m['content']
            return "Sorry, I could not find an answer."
        except Exception as e:
            logger.exception(f"qwen agent failed: {e}")
            return f"(agent error: {e})"
    return await asyncio.to_thread(_run)
