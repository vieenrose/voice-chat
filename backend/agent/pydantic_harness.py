"""
PydanticAI harness for Apodex-2B voice chat.
Native parallel tool calls, async streaming, validation.
"""
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List
from loguru import logger

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent._shared import set_emit_target, emit as _emit, agent_call_lock

API_BASE = "http://127.0.0.1:11435/v1"
MODEL_ID = "apodex-1.0-2b"

import asyncio as _asyncio

def _make_model():
    # OpenAI-compatible llama-server, no thinking for voice
    return OpenAIChatModel(
        MODEL_ID,
        provider=OpenAIProvider(base_url=API_BASE, api_key="none"),
    )

SYSTEM = """You are a helpful voice assistant. Be concise and natural.
For weather/news/facts/current events, you MUST call web_search - never hallucinate.
For date/time questions (星期几/几号/几点), call get_current_datetime.
Always answer in the user's language. When you have search results, quote specifics with numbers and sources.
Never claim results lack information they contain."""

def _web_search(query: str) -> str:
    from tools.web_search import web_search_sync, format_results
    import time
    _emit({"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query})
    t0 = time.time()
    res = web_search_sync(query, count=5)
    formatted = format_results(res.get("results", [])) if res.get("results") else "No results"
    _emit({"type": "tool_result", "name": "web_search", "result": res, "formatted": formatted, "latency_ms": int((time.time()-t0)*1000), "source": res.get("source","")})
    return formatted

def _get_datetime(timezone: str = "UTC") -> str:
    _emit({"type": "tool_call", "name": "get_current_datetime", "arguments": {"timezone": timezone}})
    tz = timezone or "UTC"
    try:
        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.utcnow()
        tz = "UTC"
    tom = now + timedelta(days=1)
    fmt = f"Current: {now.strftime('%A %Y-%m-%d %H:%M:%S')} ({tz}). Today {now.strftime('%A')} {now.strftime('%Y-%m-%d')}, Tomorrow {tom.strftime('%A')} {tom.strftime('%Y-%m-%d')}."
    _emit({"type": "tool_result", "name": "get_current_datetime", "result": {"date": now.strftime("%Y-%m-%d")}, "formatted": fmt, "latency_ms": 1, "source": "datetime"})
    return fmt

_agent = None
def _get_agent():
    global _agent
    if _agent is None:
        model = _make_model()
        _agent = Agent(
            model,
            tools=[_web_search, _get_datetime],
            system_prompt=SYSTEM,
            retries=1,
        )
    return _agent

async def run_agent_task(task: str, event_q=None) -> str:
    loop = _asyncio.get_running_loop()
    set_emit_target(loop, event_q)
    agent = _get_agent()
    # Fast greeting path
    import re
    m = re.search(r"Current question:\s*(.*)", task, re.S)
    cur = m.group(1).strip() if m else task.strip()
    if re.match(r'^\s*(hi|hello|hey|你好|您好)\b', cur.lower()) and len(cur) < 50 and not re.search(r'(weather|news|search|find|what is|who is|星期|天气|新闻)', cur.lower()):
        return "Hello! How can I help you today?" if not re.search(r'[\u4e00-\u9fff]', cur) else "你好！有什么可以帮你的？"
    def _run():
        try:
            # PydanticAI run with history handling
            with agent_call_lock:
                result = agent.run_sync(task)
            return result.output if hasattr(result, 'output') else str(result)
        except Exception as e:
            logger.exception(f"pydantic agent failed: {e}")
            return f"(agent error: {e})"
    return await asyncio.to_thread(_run)
