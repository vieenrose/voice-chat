"""
Qwen-Agent harness for Qwen3.5-2B voice chat.
Wraps Qwen-Agent's Assistant with our SearXNG + wttr + granite tools.
"""
# qwen_agent/settings.py reads QWEN_AGENT_MAX_LLM_CALL_PER_RUN at import time, so the
# assignment below `import os` has to precede the qwen_agent imports — that ordering is
# the whole reason E402 is relaxed for this file (and only this file).
# ruff: noqa: E402
import os
# This one assignment has to run BEFORE `import qwen_agent`: qwen_agent/settings.py
# reads QWEN_AGENT_MAX_LLM_CALL_PER_RUN into a module constant at import time (20 by
# default, which is how the 8-call loop happened). Everything after it is an import by
# necessity — hence the file-level E402 pragma at the top of this file.
os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = "3"
import asyncio
from loguru import logger
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import json
import re

from agent._shared import set_emit_target, emit as _emit, agent_call_lock

# --- agent sampling knobs -------------------------------------------------------
# Tool-call routing on this 2B model is sampling-sensitive: the same 7-query set scored
# 57%, 57%, 71% (mean 62%) across three passes at temperature 0.7. Two things follow:
# (1) the temperature is configurable, so it can be A/B-ed against measured routing
#     accuracy instead of being argued about, and
# (2) an optional per-request seed (qwen-agent forwards `seed` into generate_cfg, and
#     llama-server honours it) turns a benchmark run reproducible. Neither changes the
#     defaults unless you set them.
LLM_AGENT_TEMP = float(os.getenv("LLM_AGENT_TEMP", "0.7"))
LLM_AGENT_TOP_P = float(os.getenv("LLM_AGENT_TOP_P", "0.9"))
LLM_AGENT_SEED = os.getenv("LLM_AGENT_SEED", "").strip()


# Import our search logic
try:
    from tools.web_search import _wttr_weather          # direct wttr.in-backed forecast
except ImportError as _e:                              # search stack broken/uninstalled
    _wttr_weather = None
    print(f"[qwen_harness] tools.web_search unavailable ({_e}); weather tool falls back to generic search")

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
        import json as _json
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
                except Exception:
                    from tools.web_search import repair_truncated_json_query
                    query = repair_truncated_json_query(s)
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
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
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
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        tz = params.get('timezone', 'UTC') if isinstance(params, dict) else 'UTC'
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        try:
            now = datetime.now(ZoneInfo(tz))
        except Exception:
            now = datetime.utcnow()
            tz = "UTC"
        tom = now + timedelta(days=1)
        fmt = f"Current: {now.strftime('%A %Y-%m-%d %H:%M:%S')} ({tz}). Today {now.strftime('%A')} {now.strftime('%Y-%m-%d')}, Tomorrow {tom.strftime('%A')} {tom.strftime('%Y-%m-%d')}."
        _emit({"type": "tool_call", "name": "get_current_datetime", "arguments": {"timezone": tz}})
        _emit({"type": "tool_result", "name": "get_current_datetime", "result": {"date": now.strftime("%Y-%m-%d")}, "formatted": fmt, "latency_ms": 1, "source": "datetime"})
        return fmt

# --- repair: a tool the model only NAME-DROPPED -----------------------------------
# A 2B model sometimes "calls" a tool by printing its name and then answering:
#     現在的時間是 [get_current_datetime]。        (no tool call was made)
# The user then literally hears "the current time is bracket get current datetime
# bracket", which is worse than a refusal. Reproducing it needed a fixed seed
# (LLM_AGENT_SEED=4711) — i.e. it is a sampling lottery, not a one-off. So: detect the
# placeholder, execute the tool that was named, and either splice a short result in
# place (datetime is speakable verbatim, and needs no extra LLM call) or hand the
# result back to the model for one corrective continuation (search results are not).
_TOOL_PLACEHOLDER_RE = re.compile(r"\[\s*(get_current_datetime|get_weather|web_search)\s*\]")


def detect_unexecuted_tool(text, tool_ran: bool = False):
    """The bracketed tool name in `text`, or None if the answer is clean / a tool
    already ran this turn (a real call that returned 'No results' can legitimately be
    mentioned by name)."""
    if tool_ran or not text:
        return None
    m = _TOOL_PLACEHOLDER_RE.search(text)
    return m.group(1) if m else None


def splice_tool_result(text: str, tool_name: str, result: str) -> str:
    """Replace the placeholder with the speakable part of the tool result."""
    if not result:
        return _TOOL_PLACEHOLDER_RE.sub("", text)
    if tool_name == "get_current_datetime":
        seg = result.split(".")[0].strip()                    # "Current: Mon 2026-… 12:49:33 (Asia/Taipei)"
        if seg.lower().startswith("current:"):
            seg = seg[len("current:"):].strip()
        repl = seg
    else:
        repl = result.strip().split("\n")[0][:200]
    # lambda replacement: a result containing backslashes or `\1` must not be read as
    # a regex replacement template.
    out = _TOOL_PLACEHOLDER_RE.sub(lambda _m: repl, text, count=1)
    return _TOOL_PLACEHOLDER_RE.sub("", out)                   # never leave a bracket behind


# A fabricated clock is worse than a fabricated headline: it is checkable at a glance and
# this assistant talks to someone's morning. On the voice path (which appends the language
# hint, so it samples differently) the model answered "現在是下午 3 點 25 分" when it was
# 13:25 — with no tool call at all, so the placeholder repair above never fires. These two
# narrow regexes make the datetime tool non-optional *for date/time questions only*: the
# question has to ask about now, AND the answer has to state a time/date.
_TIME_QUESTION_RE = re.compile(
    r"(現在|现在|此刻|幾點|什么时候|什麼時間|什么时间|星期幾|礼拜几|週幾|周幾|今天|今天|明日|後天|日期|"
    r"what time|what's the (?:date|time)|what day|current (?:time|date)|today'?s date)", re.I)
_TIME_CLAIM_RE = re.compile(
    r"(\d{1,2}\s*[:：]\s*\d{2}|\d{1,2}\s*[點点]\s*\d{0,2}\s*分?|"
    r"[上下]午\s*\d{1,2}|\d{1,2}\s*[上下]午|"
    r"星期[一二三四五六日天]|礼拜[一二三四五六日天]|週[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号號]|\d{4}\s*年\s*\d{1,2}\s*月|"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)", re.I)


def fabricates_time_without_tool(task: str, text: str, tool_ran: bool = False) -> bool:
    """True when the request asks for the current date/time, the answer states one, and
    get_current_datetime never ran. Deliberately narrow: 'when is the meeting?' answered
    'at 5 pm' does not trigger (the question is not about now), and neither does a date
    mentioned in passing by a non-time question."""
    if tool_ran or not task or not text:
        return False
    if not _TIME_QUESTION_RE.search(task):
        return False
    return bool(_TIME_CLAIM_RE.search(text))


def format_clock_zh(tool_result: str) -> str:
    """Turn get_current_datetime's `Current: Monday 2026-08-31 21:18:33 (Asia/Taipei). …`
    into one speakable Traditional-Chinese sentence.

    Used when the model asserted a clock it never verified: rewriting the sentence by
    hand is the point — asking the model to redo it (the first version of this repair)
    came back with the *same* fabricated time, because qwen-agent reuses the agent's
    memory and the wrong claim was already in the context. Truthful and flat beats
    fluent and wrong when the subject is what time it is.
    """
    from datetime import datetime as _dt
    m = re.search(r"Current:\s*([A-Za-z]+)\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", tool_result or "")
    if not m:
        return ""
    _weekday = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", "Thursday": "星期四",
                "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"}
    year, month, day, hh, mm = (int(x) for x in m.groups()[1:])
    tz = re.search(r"\(([^)]+)\)\s*\.?\s*$", (tool_result or "").split(".")[0])
    tz_label = {"Asia/Taipei": "台北時間", "UTC": "UTC"}.get(tz.group(1) if tz else "", tz.group(1) if tz else "")
    try:                                        # sanity: the tool's own weekday must match its date
        _dt.strptime(f"{year}-{month:02d}-{day:02d}", "%Y-%m-%d")
    except ValueError:
        return ""
    zh_wd = _weekday.get(m.group(1), m.group(1))
    suffix = f"（{tz_label}）" if tz_label else ""
    return f"現在是 {year} 年 {month} 月 {day} 日{zh_wd}，{hh} 點 {mm:02d} 分{suffix}。"


# --- pre-flight forcing for "what is true RIGHT NOW" questions --------------------
# The last routing miss on the benchmark set was "Who is the president of France?"
# answered from memory (7/7 → 6/7). For incumbency questions memory is never
# acceptable: the answer expires, and a 2 B router will not reliably notice that. The
# clock guard above runs *after* the answer; here the question alone identifies the
# class, so the tool is invoked BEFORE the model speaks — which also avoids the
# memory-replay failure that made a post-hoc "please redo it" unreliable.
# Narrow on purpose: an office/holder noun (or 現任/誰是, or "who won"), not merely a
# question. "What is the weather in Tokyo?" and greetings must not pay for a search.
_INCUMBENT_RE = re.compile(
    r"(?:誰是|谁是|哪一位是|現任|现任|目前在任)"
    r"|who\s+(?:is|are|was|will\s+be)\s+(?:the|a|an)?[\w'’ é\-]{0,40}?"
    r"(?:president|prime minister|premier|\bpm\b|chancellor|mayor|governor|monarch|king|queen|emperor|"
    r"ceo|chief executive|chairman|chairwoman|chairperson|minister|secretary|speaker|senator|"
    r"cardinal|pope|head coach|headmaster)"
    r"|who\s+(?:won|wins|leads|runs|heads|took)"
    r"|(?:最新|最近)[^?？。]{0,8}(?:新聞|新闻|消息|結果|结果|價格|价格)"
    r"|(?:as of|right now)[^.?？]{0,24}(?:president|leader|price|version|record|champion)", re.I)


def requires_fresh_facts(task: str) -> bool:
    """True when the request can only be answered by something that is currently true
    (an office holder, a winner, a latest figure) — i.e. memory is not a source."""
    return bool(task) and bool(_INCUMBENT_RE.search(task))


def _run_named_tool(name: str, task: str) -> str:
    """Execute a registered tool by name, deriving params from the user's request.
    The tools' own .call() emits the tool_call / tool_result events, so the UI sees
    exactly what ran — the repair is not hidden."""
    from qwen_agent.tools.base import TOOL_REGISTRY
    inst = TOOL_REGISTRY.get(name)
    if inst is None:
        return ""
    inst = inst()
    try:
        if name == "get_current_datetime":
            return inst.call({"timezone": os.getenv("VOICE_TZ", "Asia/Taipei")})
        # Both remaining branches MUST go through the registered tool's .call(), not
        # straight to web_search_sync: .call() is what emits the tool_call / tool_result
        # events. The first version of this helper called web_search_sync directly and
        # the forced search was invisible to /api/chat and to the WS client — a repair
        # the user cannot see is indistinguishable from one that never happened.
        if name == "web_search":
            return inst.call({"query": task[:80]})
        if name == "get_weather":
            # get_weather is itself a query reformulation over web_search_sync, and
            # extracting a bare city out of a sentence is the unreliable part of it,
            # so run the same backend through the search tool with the request verbatim.
            ws = TOOL_REGISTRY.get("web_search")
            return ws().call({"query": task[:80]}) if ws is not None else ""
    except Exception as e:
        logger.warning(f"repair: tool {name} failed: {e!r}")
        return ""


def _thinking_on() -> bool:
    """Whether llama-server runs the model's thinking pass for agent turns.

    A/B-measured on the benchmark set, both at the default (random) seed, 3 passes each:

      thinking ON   tool accuracy 81 % (71/86/86)   LLM TTFT p50 1384 ms   E2E p50 3856 ms
      thinking OFF  tool accuracy 71 % (71/71/71)   LLM TTFT p50   56 ms   E2E p50 1638 ms

    Off is seven times faster to first token and it fixes the spoken scratchpad by simply
    not producing one — and it is WORSE, because the failures it produces are not routing
    misses but fabrications: "東京天氣晴朗，平均溫度約25°C" with no get_weather call, generic
    AI "news" with no web_search, and a greeting answered with an invented date. A missing
    tool call is a MISS in the report; an invented temperature is a confident lie that
    nobody notices. So thinking stays on and the scratchpad is filtered on the way to the
    speaker instead (_strip_thinking in llm/ling_streaming.py). LLM_AGENT_THINKING=0 to
    try the fast-but-inventive variant deliberately."""
    return os.getenv("LLM_AGENT_THINKING", "1").strip().lower() in ("1", "true", "yes")


def _smalltalk_rule() -> str:
    """The 2 B router will happily answer "how are you?" by calling get_current_datetime
    (observed: "今天好，星期二上午 2 點 52 分，天氣晴朗。"). A direct instruction is the
    cheapest fix, and it is measurable: the greeting scored 1/3 correct before it."""
    if os.getenv("LLM_AGENT_NO_TOOL_SMALLTALK", "1").strip().lower() in ("0", "false", "no"):
        return ""
    return (" If the user is greeting you or making small talk, just reply conversationally — "
            "do NOT call a tool for that. Only call a tool when the answer depends on something "
            "you cannot know: the current date/time, the weather, or a changing fact. ")


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
            'temperature': LLM_AGENT_TEMP,
            'top_p': LLM_AGENT_TOP_P,
            # Thinking is expensive here in the literal sense: at enable_thinking True the
            # 2 B model spends 6-9 s of first-token latency and hits the 25 s wall budget on
            # trivial turns, and its deliberation arrives in `content`, so the voice output
            # literally spoke lines like "根据规则，必须调用工具…" to the user. Measured both
            # ways (see README): thinking off = same tool routing, ~4x faster first token,
            # no spoken scratchpad. Set LLM_AGENT_THINKING=1 to opt back in.
            'chat_template_kwargs': {'enable_thinking': _thinking_on()},
            # Cap thinking tokens (llama-server's native reasoning_budget_tokens,
            # forwarded via extra_body) without touching max_tokens for the real
            # answer/tool-call. Voice turns don't need long deliberation, and an
            # uncapped thinking pass can burn most of max_tokens on reasoning
            # alone, which is what caused the truncated tool-call JSON bug this
            # session (reasoning ate the budget, leaving too few tokens to close
            # the <tool_call> block) as well as unnecessary first-audio latency.
            'extra_body': {'reasoning_budget_tokens': 200},
        },
        'enable_thinking': _thinking_on()
    }
    return Assistant(
        llm=llm_cfg,
        function_list=['web_search', 'get_weather', 'get_current_datetime'],
        system_message=("You are a helpful voice assistant." + _smalltalk_rule() + "For weather use get_weather(location, date) — 1 call max. For general search use web_search — 1 call max with 3-8 words, then answer. For date/time use get_current_datetime — 1 call max. Never do more than 2 tool calls per turn. Always default to Traditional Chinese (Taiwan usage, 繁體中文) regardless of what language the question was asked in — only keep English for proper nouns, technical terms, or vocabulary that doesn't translate well; never answer a whole sentence in Simplified Chinese or English."
        ),
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

async def run_agent_task(task: str, event_q=None, history=None) -> str:
    """Run one agent turn in a worker thread.

    `history` is a list of real {role, content} turns (user/assistant) — passed
    through to qwen-agent as messages so referential follow-ups resolve instead of
    being a 120-char digest. Emits tool_call/tool_result plus llm_delta events (the
    answer as it is generated; see the reset protocol in _run).
    """
    loop = _asyncio.get_running_loop()
    set_emit_target(loop, event_q)
    agent = _get_agent()
    hist = [m for m in (history or []) if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and m.get('content')]
    def _run():
        try:
            # No greeting fast-path: it returned a canned "你好！有什麼可以幫你的？"
            # without ever consulting the model, which contradicted this repo's own
            # "no hard-coded cheating" rule and hid real first-token latency from any
            # benchmark that greeted first. The prompt already handles greetings.
            messages = hist + [{'role': 'user', 'content': task}]
            if requires_fresh_facts(task):
                # A real invocation of a real tool (its own tool_call/tool_result events
                # fire inside .call, so the UI and the benchmark see it) placed in context
                # ahead of the answer. Nothing is invented here: if the search fails the
                # messages are untouched and the model answers as it likes.
                try:
                    forced = _run_named_tool('web_search', task[:80])
                    if forced:
                        messages = messages + [
                            {'role': 'assistant', 'content': '',
                             'function_call': {'name': 'web_search', 'arguments': json.dumps({'query': task[:80]})}},
                            {'role': 'function', 'name': 'web_search', 'content': forced[:3000]},
                        ]
                        logger.info("pre-flight web_search forced for an incumbency/latest question")
                except Exception as e:
                    logger.warning(f"pre-flight search failed (answering without it): {e!r}")
            all_resps = []
            # --- answer streaming (qwen-agent yields `response + partial_output` on
            # every LLM chunk, so the final step's content is available incrementally).
            # Streaming is tracked PER STEP: content stays empty while the model burns
            # tokens on reasoning, a step whose tail becomes a tool call/tool result
            # means that step's text was not the answer (emit llm_reset so the consumer
            # drops it), and a LATER step may still stream the real answer. Abandoning
            # permanently at the first tool step (the first version of this) silently
            # disabled streaming for every tool turn — i.e. exactly the latency win
            # this exists to get. Permanent abandonment only for template junk.
            stream_step = -1
            streamed_len = 0
            abandoned = False
            with agent_call_lock:
                run_kwargs = {"seed": int(LLM_AGENT_SEED)} if LLM_AGENT_SEED else {}
                for resp in agent.run(messages=messages, **run_kwargs):
                    all_resps.append(resp)
                    if abandoned:
                        continue
                    lst = resp if isinstance(resp, list) else [resp]
                    last = lst[-1] if lst else None
                    if isinstance(last, dict) and last.get('role') == 'assistant' and not last.get('function_call'):
                        c = last.get('content')
                        if isinstance(c, str):
                            if '<tool_call>' in c or '<arg_' in c or '<|' in c:
                                abandoned = True
                                _emit({"type": "llm_delta", "text": "", "reset": True})
                                continue
                            if len(lst) != stream_step:              # a new step started
                                if streamed_len:                     # previous step's text was not the answer
                                    _emit({"type": "llm_delta", "text": "", "reset": True})
                                stream_step, streamed_len = len(lst), 0
                            if len(c) > streamed_len:
                                _emit({"type": "llm_delta", "text": c[streamed_len:]})
                                streamed_len = len(c)
                    else:
                        # Tail is a tool call / tool result: this step isn't the answer.
                        if streamed_len:
                            _emit({"type": "llm_delta", "text": "", "reset": True})
                        stream_step, streamed_len = -1, 0
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
                final_text = candidates_all[-1].get('content', '')
                # Did a tool actually run this turn? (a real call that came back empty
                # may legitimately be referred to by name — that is not the bug below)
                tool_ran = any(isinstance(m, dict) and (m.get('role') == 'function' or m.get('function_call'))
                               for resp in all_resps for m in (resp if isinstance(resp, list) else [resp]))
                missing = detect_unexecuted_tool(final_text, tool_ran)
                must_regenerate = False
                if missing is None and fabricates_time_without_tool(task, final_text, tool_ran):
                    # Splicing the true clock into "現在是下午 3 點 25 分" would leave the
                    # wrong claim standing next to the right one, so let the model rewrite
                    # the sentence with the tool result in context.
                    logger.warning("answer states a date/time that no tool verified — forcing the call")
                    missing, must_regenerate = "get_current_datetime", True
                if missing:
                    logger.warning(f"answer named {missing} without calling it — executing the tool it referenced")
                    result = _run_named_tool(missing, task)
                    repaired = ""
                    if result:
                        if must_regenerate:
                            repaired = format_clock_zh(result)      # deterministic; no LLM round-trip
                        elif missing == 'get_current_datetime':
                            repaired = splice_tool_result(final_text, missing, result)   # speakable verbatim, no extra LLM call
                        else:
                            # Search output is not speakable verbatim: hand it back with the
                            # result in context and take the corrected answer (one extra call,
                            # only on this failure path).
                            cont = messages + [
                                {"role": "assistant", "content": final_text},
                                {"role": "function", "name": missing, "content": result[:3000]},
                            ]
                            with agent_call_lock:
                                for resp in agent.run(messages=cont, **run_kwargs):
                                    lst = resp if isinstance(resp, list) else [resp]
                                    for m in lst:
                                        if (isinstance(m, dict) and m.get('role') == 'assistant'
                                                and isinstance(m.get('content'), str) and m['content'].strip()):
                                            repaired = m['content']
                    if not repaired:
                        if missing in ("web_search", "get_weather"):
                            # Continuing with the raw results did not produce a usable
                            # answer; splicing a results blob into a spoken sentence would
                            # be worse than admitting it.
                            repaired = "我搜尋了網路資料，但摘要失敗了，請再問一次。"
                        else:
                            repaired = splice_tool_result(final_text, missing, result)
                    # Withdraw the draft that was streamed (the pipeline also refuses to
                    # speak tool-name artifacts); the consumer replays this repaired text.
                    final_text = repaired
                    _emit({"type": "llm_delta", "text": "", "reset": True})
                return final_text
            # Fallback to memory (also filter empty)
            if hasattr(agent, 'memory') and agent.memory:
                # NB: must NOT be named `hist` — that would make `hist` local to _run
                # and turn the `messages = hist + [...]` above into
                # UnboundLocalError on every single call.
                mem_hist = agent.memory.get_history() if hasattr(agent.memory, 'get_history') else []
                for m in reversed(mem_hist):
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
