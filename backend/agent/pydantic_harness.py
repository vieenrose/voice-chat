"""
PydanticAI harness for Apodex-2B voice chat.
Native parallel tool calls, async streaming, validation.
"""
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from loguru import logger

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio as _asyncio

from agent._shared import set_emit_target, emit as _emit, agent_call_lock

API_BASE = "http://127.0.0.1:11435/v1"
MODEL_ID = "apodex-1.0-2b"

def _make_model():
    # OpenAI-compatible llama-server, no thinking for voice
    return OpenAIChatModel(
        MODEL_ID,
        provider=OpenAIProvider(base_url=API_BASE, api_key="none"),
    )

SYSTEM = """You are a helpful voice assistant. Be concise and natural.
For weather/news/facts/current events, you MUST call web_search - never hallucinate.
For date/time questions (星期幾/幾號/幾點), call get_current_datetime.
Always default to Traditional Chinese (Taiwan usage, 繁體中文) regardless of what language the question was asked in — only keep English for proper nouns, technical terms, or vocabulary that doesn't translate well; never answer a whole sentence in Simplified Chinese or English. When you have search results, quote specifics with numbers and sources.
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

def reset_agent():
    """Interface parity with agent/qwen_harness.py and agent/harness.py's
    reset_agent() (called best-effort by app.py's POST /api/model handler on
    every switch) — a no-op beyond dropping the cache, since this harness
    targets a fixed, separate model (MODEL_ID above) unrelated to llm_manager's
    Qwen3.5 registry."""
    global _agent
    _agent = None

async def run_agent_task(task: str, event_q=None, history=None) -> str:
    """Run one turn; `history` (real role/content turns) becomes PydanticAI's
    message_history. See the identical note in agent/qwen_harness.py.
    """
    loop = _asyncio.get_running_loop()
    set_emit_target(loop, event_q)
    agent = _get_agent()
    hist = [{"role": m["role"], "content": m["content"]} for m in (history or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]

    def _run():
        try:
            # PydanticAI run with history handling
            with agent_call_lock:
                result = agent.run_sync(task, message_history=hist) if hist else agent.run_sync(task)
            return result.output if hasattr(result, 'output') else str(result)
        except Exception as e:
            # Spoken via TTS with no filtering for an "error"-looking string (see the
            # identical note in qwen_harness.py) — log the real detail, return a clean
            # generic fallback instead of the raw exception text.
            logger.exception(f"pydantic agent failed: {e}")
            return "抱歉，這個問題我暫時無法回答，請再試一次。"
    return await asyncio.to_thread(_run)
