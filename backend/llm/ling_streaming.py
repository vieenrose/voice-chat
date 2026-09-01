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

_THINK_RE = re.compile(r"<\|im_start\|>think.*?(?:\*/?<\|im_end\|>|\*/|</think>|$)|<think\b[^>]*>.*?(?:</think>|$)", re.S)
# Fallback for models that emit plain-English thinking without tags (observed with
# Qwen3.5-0.8B via --reasoning-format deepseek where reasoning_budget truncated the
# thinking block and the remainder spilled into `content` as "Thinking Process: …").
# Shape-based, not per-question: matches the template header plus any bullet checklist.
_THINK_PLAIN_RE = re.compile(
    r"Thinking Process:.*?(?:\n\s*\n|$)|"
    r"(?:^|\n)\s*\*\*?Analyze the Request.*?(?:\n\s*\n|$)|"
    r"(?:^|\n)\s*Constraint Checklist.*?(?:\n\s*\n|$)",
    re.S | re.I,
)

# Generic reasoning-marker detector for streaming deltas that arrived as `content`
# after reasoning_content was truncated (reasoning_budget). No CJK + deliberation
# verbs + tool/meta nouns = thinking, regardless of exact wording.
_REASONING_MARKER_RE = re.compile(
    # Deliberation the model narrates about its own task. Kept to phrases that are
    # *about reasoning itself* — anything that is merely our own instruction text
    # coming back is caught generically by _is_own_prompt_echo() instead of being
    # listed here, so a reworded system prompt cannot silently start leaking.
    r"(Thinking Process|Analyze the Request|Constraint Checklist|Constraint Check|Confidence Score"
    r"|This is unusual|However,\s+I must|I must acknowledge|Let.s break down"
    r"|The user wants me to|the user.s prompt says|the instruction says|system instruction says"
    r"|looking closely at the prompt|prompt structure|System constraints"
    r"|\bQuery:\s*\"|\* Query:|\* Language:|\* Constraint:|Tool Call:|User asks:)",
    re.I,
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _norm_for_echo(t: str) -> str:
    """Normalize for substring comparison: casefold, drop punctuation/whitespace."""
    return re.sub(r"[^\w一-鿿]+", "", (t or "").lower())


class SpokenGuard:
    """Tracks what has already been spoken within one turn so nothing is said twice.

    The agent harness re-streams an assistant message from the beginning whenever
    qwen_agent opens a new step (it emits `reset` so the UI can redraw its bubble),
    but audio that has already played cannot be withdrawn. The re-stream also splits
    at different sentence boundaries than the first pass, so comparing whole
    sentences for equality misses it — "…輝達與 Lambda 簽署了…" the first time can come
    back as a bare "Lambda 簽署了…". Matching against the normalized concatenation of
    everything spoken so far catches the exact repeat and the re-chunked fragment
    alike, without needing to know anything about the content."""

    #

    MIN_DEDUP_LEN = 8  # below this, a legitimately repeated short reply is likelier than a bug

    def __init__(self) -> None:
        self._spoken = ""

    def should_speak(self, text: str) -> bool:
        norm = _norm_for_echo(text)
        if not norm:
            return False
        if len(norm) >= self.MIN_DEDUP_LEN and norm in self._spoken:
            return False
        self._spoken += norm
        return True


_ECHO_CORPUS: str | None = None


def _is_own_prompt_echo(s: str) -> bool:
    """True when the text is a verbatim span of our OWN instructions.

    The model sometimes replays part of its system prompt (or the appended
    language hint) as if it were the answer — observed spoken aloud as
    "proper nouns, technical terms, or vocabulary that doesn't translate
    well; never answer a whole sentence in Simplified Chinese or English."
    Nothing we told the model is ever a legitimate answer to the user, so
    rather than blocklisting the individual sentences that happened to leak
    (which would be per-case hard-coding that the next paraphrase escapes),
    check membership in the instruction text itself. Built lazily because
    SYSTEM_PROMPT/LANG_HINT are defined below this helper."""
    global _ECHO_CORPUS
    norm = _norm_for_echo(s)
    # Short fragments ("繁體中文", "web_search") legitimately appear in real answers.
    if len(norm) < 20:
        return False
    if _ECHO_CORPUS is None:
        parts = []
        for name in ("SYSTEM_PROMPT", "LANG_HINT"):
            v = globals().get(name)
            if isinstance(v, str):
                parts.append(v)
        try:  # the agent harness carries its own, differently-worded system message
            from agent.qwen_harness import AGENT_SYSTEM_MESSAGE
            if isinstance(AGENT_SYSTEM_MESSAGE, str):
                parts.append(AGENT_SYSTEM_MESSAGE)
        except Exception:
            pass
        _ECHO_CORPUS = _norm_for_echo(" ".join(parts))
    if not _ECHO_CORPUS:
        return False
    if norm in _ECHO_CORPUS:
        return True
    # The model also replays instructions *paraphrased or truncated* ("For weather use
    # get_weather...", "only keep English for proper nouns..."), which no exact-substring
    # test can catch. Compare character shingles instead: text that is mostly built out
    # of fragments of our own instructions is an echo, however it was chopped up.
    if len(norm) < 24:
        return False
    k = 8
    shingles = {norm[i:i + k] for i in range(0, len(norm) - k + 1)}
    if not shingles:
        return False
    hits = sum(1 for sh in shingles if sh in _ECHO_CORPUS)
    return hits / len(shingles) >= 0.6


# A deliberation checklist header ("2. Evaluate Tool Call Need:", "Determine the
# language:", "Tool Call Analysis:"). Shape-based: an English line that ends in a colon
# and is about the machinery of answering rather than about anything the user asked.
_META_NOUN = (r"tool|call|request|constraint|language|query|need|response|answer|output"
              r"|prompt|instruction|analysis|reasoning|evaluation|step|plan|check")
_CHECKLIST_HEADER_RE = re.compile(
    r"^\s*(?:[*\-#\d.]+\s*)*(?:\*\*)?\s*"
    r"(?:(?:evaluate|analyz[es]?|analyse|determine|identify|assess|consider|review|verify"
    r"|check|decide|formulate|construct)\b[^.?!\n]{0,60}?\b(?:" + _META_NOUN + r")s?\b"
    r"|(?:" + _META_NOUN + r")s?\b[^.?!\n]{0,40}?\b(?:" + _META_NOUN + r")s?\b)"
    r"[^.?!\n]{0,40}:\s*(?:\*\*)?\s*$",
    re.I,
)


def _is_reasoning_text(text: str) -> bool:
    """Heuristic: does this sentence look like internal deliberation, not an answer?"""
    if not text or not text.strip():
        return False
    s = text.strip()
    # Pure markup-thinking blocks
    if "Thinking Process" in text or "Analyze the Request" in text:
        return True
    if _REASONING_MARKER_RE.search(text):
        return True
    # Our own instructions replayed back at us are never an answer
    if _is_own_prompt_echo(s):
        return True
    if _CHECKLIST_HEADER_RE.match(s):
        return True
    # No CJK at all -> not an answer this app would give.
    #
    # This app answers in Traditional Chinese by product decision, keeping English only
    # for proper nouns and terms — and those appear *inside* a Chinese sentence, which
    # therefore still contains CJK. So a standalone, sentence-length, all-English span is
    # never the answer: in practice it is deliberation ("The question is about the current
    # president.") or replayed instructions. Judging by that property rather than by a list
    # of observed phrases is what keeps this from needing a new entry every time the model
    # rewords itself. Short English spans ("Emmanuel Macron.") stay speakable.
    if not _CJK_RE.search(text):
        low = s.lower()
        if re.search(r"^\s*\"?\s*wait\b", low):
            return True
        if re.search(r"\bi need to\b", low) and len(s) < 160:
            return True
        if len(_norm_for_echo(s)) >= 24:
            return True
        if len(s) > 20:
            has_first_person = any(p in low for p in ["i must", "i need to", "i should", "let me", "wait,", "however"])
            has_meta = any(m in low for m in ["tool", "instruction", "prompt", "search", "constraint", "checklist", "acknowledge", "system"])
            if has_first_person and has_meta:
                return True
            if text.count("*") >= 2 and "-" in text and "tool" in low:
                return True
            if re.match(r"^[A-Za-z\s,']{10,}\.?$", s) and has_first_person:
                return True
    return False


def _strip_thinking(text: str) -> str:
    """Remove the model's deliberation from text that is about to be SPOKEN.

    With enable_thinking on (which the tool-routing measurement above says we need), the
    agent path gets the thinking pass back inline in `content` rather than in
    reasoning_content — `llama-server` puts it in the message body for the qwen chat
    template. Observed in the benchmark answer previews, spoken to the user:

        "，但根据规则，必须调用工具。所以步骤应该是先调用 get_current_datetime…"
        "，然後再執行工具。但這裡用戶似乎直接給出了指令，可能是在測試我的指令遵循能力…"

    The template emits an unterminated `...*/` when the thinking pass runs out of budget,
    so the pattern must accept a thinking block with no closing marker, and must not
    require balanced delimiters: whatever follows an opened block on that turn IS the
    scratchpad. Shape-based only — no per-question rules (see
    test_live_paths_hold_no_benchmark_query_literals)."""
    if not text:
        return text
    # Strip tag-based thinking first
    if "<" in text:
        text = _THINK_RE.sub(" ", text)
    # Strip plain-English Thinking Process blocks (no tags, from truncated budget)
    if "Thinking Process" in text or "Analyze the Request" in text:
        text = _THINK_PLAIN_RE.sub(" ", text)
    # Always strip any remaining reasoning sentences — entry 9 leaked "The user wants me to
    # summarize... However, looking closely at the prompt structure..." as the final answer
    # without any Thinking Process header, so header-only filtering missed it.
    parts = re.split(r"(?<=[.!?。！？\n])\s+", text)
    kept = [p for p in parts if not _is_reasoning_text(p)]
    # Only apply if we actually removed at least one reasoning sentence and kept something
    # meaningful — avoids stripping a legitimate short answer that happens to contain a marker word.
    if kept and len(kept) != len(parts) and len(" ".join(kept).strip()) > 10:
        text = " ".join(kept)
    elif "Thinking Process" in text:
        text = _REASONING_MARKER_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean_leakage(text: str) -> str:
    """Strip leaked raw search-result format that the LLM sometimes echoes verbatim."""
    # format_results() renders "[1] Title — src\nURL: https://…\nDate/Snippet: …"; the
    # model occasionally pastes that straight into its answer, where TTS would read the
    # URLs aloud. Generic shape match only.
    text = re.sub(r"\[\d+\]\s*[^\[]*?URL:\s*https?://\S+\s*Date/Snippet:\s*", "", text)
    # A dangling numbered echo with no prose around it ("[2] …" lines to end of text).
    text = re.sub(r"(?:^|\n)\s*\[\d+\][^\n]*", "\n", text)
    # Remove "More at Wikipedia" thin-result leakage
    text = re.sub(r"More at Wikipedia\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _speakable(text: str) -> str:
    """Final shaping of text about to be synthesized.

    Markdown emphasis and links are the model answering in a format the caller cannot
    see (observed live: 法國總統是**愛德華·馬克龍**), so TTS used to read the asterisks'
    worth of silence or, worse, some backends spell them. `[text](url)` collapses to
    `text`; unpaired `*`/`_` emphasis markers are dropped. Tool XML and leaked result
    rows are handled upstream by _strip_tool_xml / _clean_leakage.
    """
    if not text:
        return text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # [text](url) -> text
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.S)  # **b** / __b__
    text = re.sub(r"(?<![\w\u4e00-\u9fff])[*_]{1,3}(?=\S)|(?<=\S)[*_]{1,3}(?![\w\u4e00-\u9fff])", "", text)
    text = text.replace("`", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()



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

# zh-TW is this app's default/primary language, stated once in SYSTEM_PROMPT above and
# in the agent harness's own system message. It is deliberately NOT appended to each
# user turn any more: glued onto the transcript it became part of what the model was
# asked, so it got copied verbatim into web_search queries and read back aloud as the
# answer. Kept here only so _is_own_prompt_echo() still recognizes the old wording if a
# model replays it; nothing sends it.
LANG_HINT = "\n（請一律使用繁體中文（台灣用語）簡潔回答；僅專有名詞、技術術語或無法翻譯的詞彙可保留英文原文，不要整句使用簡體中文或英文作答。）"

# Heuristic for fast tool trigger — bilingual (en/zh) - includes Chinese triggers for Taiwan/news
# (heuristic removed — tool calls are model-driven/native only)

def _history_digest(history: List[Dict]) -> str:
    """Text fallback for harnesses that only accept a single task string."""
    return "\n".join(f"{m.get('role')}: {str(m.get('content',''))[:400]}" for m in (history or [])[-6:])


def _clean_history(history: List[Dict]) -> List[Dict]:
    """Strip the system message (harnesses supply their own) and anything unusable,
    keeping real role/content turns so multi-turn referents survive."""
    out: List[Dict] = []
    for m in (history or []):
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out[-8:]


class LingStreaming:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-2B-MTP-GGUF", api_base: str = "http://127.0.0.1:11436/v1", mock: bool = False, device: str = "cuda", model_name: str = "qwen3.5-2b", degraded_mode: str = "error", **kwargs):
        self.model_id = model_id
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.backend = f"llm-gguf:{model_name}"
        self.mock = mock
        # "mock" = serve canned replies when the endpoint is gone (--mock opted into
        # simulation); "error" = tell the caller which endpoint is missing, because in a
        # real deployment a friendly non-answer about the weather is a worse failure than
        # an honest one.
        self.degraded_mode = degraded_mode
        self._degraded_warned = ""
        # (checked_at, ok, detail) — PER INSTANCE on purpose: llm_manager builds a new
        # adapter on a model switch, and a class-level cache would hand that new
        # api_base the previous endpoint's optimistic verdict for a whole second.
        self._last_probe = (0.0, True, "")
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

    _PROBE_TTL = 1.0          # seconds; a turn probes the LLM at most once per second

    async def _reachable(self) -> tuple[bool, str]:
        """Pre-flight: is the model server actually there?

        The constructor already probes once and sets self.mock when nothing answers, but
        it deliberately stays optimistic ("will retry per request") so a llama-server that
        starts later is picked up. The gap was in *where* the retry happened: the request
        itself raised httpx.ConnectError out of _chat_stream, which turned POST /api/chat
        into a 500 with a traceback (observed in the container, where LLM_API_BASE points
        at a compose service name that does not exist outside `docker compose up`). A
        mid-run crash is still possible; this closes the "server was never there" case,
        which is the one that used to look like a broken API rather than a missing model.
        """
        now = time.time()
        if now - self._last_probe[0] < self._PROBE_TTL:
            return self._last_probe[1], self._last_probe[2]
        ok, detail = True, ""
        try:
            async with httpx.AsyncClient(timeout=1.5) as c:
                r = await c.get(f"{self.api_base}/models")
                if r.status_code != 200:
                    ok, detail = False, f"HTTP {r.status_code} from {self.api_base}/models"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e} ({self.api_base})"
        self._last_probe = (now, ok, detail)
        if not ok:
            logger.warning(f"LLM server pre-flight failed: {detail}")
        return ok, detail

    async def _degraded(self, prompt: str, why: str) -> AsyncGenerator[dict, None]:
        """What the user hears instead of a stack trace."""
        if self.degraded_mode == "mock":
            if why != self._degraded_warned:
                logger.warning(f"--mock: model server unreachable ({why}) — serving canned mock replies. "
                               "/health reports the pipeline's mock state; this is not a real answer.")
                self._degraded_warned = why
            async for ev in self._mock_stream(prompt):
                yield ev
            return
        host = self.api_base.split("//")[-1].rstrip("/")
        if self._is_chinese(prompt):
            text = (f"模型伺服器 {host} 目前連不上，我無法生成回應。"
                    f"請確認 llama-server 已啟動後再問一次。（原因：{why.splitlines()[0][:80]}）")
        else:
            text = (f"The model server at {host} is not reachable, so I cannot answer. "
                    f"Start llama-server and ask again. (reason: {why.splitlines()[0][:80]})")
        text_so_far = ""
        for tok in re.findall(r"\S\S*|\s+", text):
            text_so_far += tok
            yield {"type": "llm_token", "token": tok, "text_so_far": text_so_far, "latency_ms": 30}
            await asyncio.sleep(0.01)
        yield {"type": "llm_done", "text": text_so_far}

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
                        # Reasoning vs answer: reasoning must never be spoken, but the UI should show it.
                        rc = delta.get("reasoning_content")
                        if rc is not None:
                            yield {"type": "reasoning", "token": rc}
                        token = delta.get("content")
                        if token is not None:
                            # Spillover guard: when reasoning_budget truncates, thinking leaks
                            # into content as plain English checklist. Route it to reasoning.
                            if _is_reasoning_text(token):
                                yield {"type": "reasoning", "token": token}
                            else:
                                yield {"type": "token", "token": token}
                        # Finish reason
                        if choice.get("finish_reason"):
                            yield {"type": "finish", "reason": choice["finish_reason"]}
                    except Exception:
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

    async def _mock_stream(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Canned bilingual replies. Split out of generate_stream so the degraded path can
        reuse it: --mock asks for simulated answers, and reaching a dead endpoint in that
        mode should still sound like the demo, not like an outage."""
        t0 = time.time()
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

    async def generate_stream(self, prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        t0 = time.time()
        if self.mock:
            async for ev in self._mock_stream(prompt):
                yield ev
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
        # Pre-flight FIRST — before the harness/legacy branch, not after it. The legacy
        # loop is where a light install (no qwen-agent) actually ends up, so a check
        # placed below that early return never ran there, and the container still 5xx'd.
        if not self.mock:
            _ok, _why = await self._reachable()
            if not _ok:
                async for ev in self._degraded(prompt, _why):
                    yield ev
                return
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

        # Hand the agent the REAL conversation (role/content pairs) rather than a
        # "role: content[:120]" digest — truncating history to 120 chars per turn is
        # what made referential follow-ups ("BBC headlines", "and tomorrow?")
        # unanswerable: the thing being referred to had usually been cut. Harnesses
        # that can't take messages still get a digest (see _history_digest fallback).
        _hist_msgs = _clean_history(history)
        task_str = prompt
        q = asyncio.Queue()
        task = asyncio.create_task(run_agent_task(task_str, q, history=_hist_msgs))
        final_text = ""
        streamed = ""          # text already emitted as it was generated
        any_streamed = False
        _t_stream0 = time.time()

        def _speak(text: str, so_far: str):
            """Build one llm_token event (kept as a helper so the streamed and the
            replayed paths stay identical in shape)."""
            return {"type": "llm_token", "token": text, "text_so_far": so_far,
                    "latency_ms": int((time.time()-_t_stream0)*1000)}

        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue
                if ev["type"] == "llm_reasoning":
                    yield {"type": "llm_reasoning", "text": ev.get("text", "")}
                elif ev["type"] == "tool_call":
                    yield {"type": "tool_call", "name": ev["name"],
                           "arguments": ev.get("arguments", {}), "query": ev.get("query", "")}
                elif ev["type"] == "tool_result":
                    yield {"type": "tool_result", "name": ev["name"], "result": ev.get("result"),
                           "formatted": ev.get("formatted", ""), "latency_ms": ev.get("latency_ms", 0),
                           "source": ev.get("source", "")}
                elif ev["type"] == "llm_delta" and ev.get("reset"):
                    # The harness streamed text it then had to take back (an XML tool
                    # call appeared, or a tool step followed). Tell the consumer to
                    # discard what it buffered, then speak the authoritative final
                    # answer in full.
                    streamed = ""
                    any_streamed = False
                    yield {"type": "llm_reset"}
                elif ev["type"] == "llm_delta":
                    # True incremental answer text from the harness's own generation
                    # stream. Before this existed the whole agent loop ran to completion
                    # and the finished answer was replayed character by character, so a
                    # tool turn produced no audio until everything was over (~18s
                    # measured) and every llm_ttft_ms for a tool turn described the
                    # replay, not the model.
                    d = ev.get("text") or ""
                    if not d:
                        continue
                    # Markdown emphasis markers are stream-safe to drop one character at
                    # a time (a `**` pair can straddle a chunk boundary, so only the pair
                    # regexes have to wait for the final text). Without this, an answer of
                    # 法國總統是**愛德華·馬克龍** got its asterisks synthesized.
                    d = d.replace("*", "").replace("`", "")
                    if not d:
                        continue
                    any_streamed = True
                    streamed += d
                    yield _speak(d, streamed)
            final_text = await task
        finally:
            # If we're being cancelled (barge-in) mid-loop, `task` would otherwise be
            # orphaned: never awaited, silently running to completion in the background
            # (its own `agent.run()` call happens in a thread via asyncio.to_thread, which
            # can't be interrupted mid-call either way, but at least stop *waiting* on it
            # and don't leak an unretrieved exception).
            if not task.done():
                task.cancel()
            def _drain(t):
                # Retrieving the exception prevents "Task exception was never retrieved",
                # but swallowing it silently is worse: an agent that died on
                # ConnectError would look exactly like an agent that found nothing, and
                # the user would hear "I couldn't find a clear answer" about a dead server.
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    logger.error(f"agent task died: {type(exc).__name__}: {exc} "
                                 f"(answer so far: {len(streamed) + len(final_text)} chars)")
            task.add_done_callback(_drain)
        if not isinstance(final_text, str):
            final_text = str(final_text)
        if "<tool_call>" in final_text or "<arg_" in final_text:
            final_text = _strip_tool_xml(final_text)
        final_text = re.sub(r"<[^>]+>", " ", final_text)
        # _strip_thinking is a BACKSTOP, not the fix: with --reasoning-format deepseek (what
        # llm_manager now requests) the deliberation never reaches `content` at all. It
        # matters on builds without that flag, or with LLM_REASONING_FORMAT=none, where a
        # completed thinking block still shows up in the answer text. Live deltas cannot be
        # filtered this way (you cannot know a block is thinking until it ends, and by then
        # it has been spoken), which is why the durable fix is server-side.
        final_text = _speakable(_clean_leakage(_strip_thinking(final_text)))
        final_text = " ".join(final_text.split())
        if not final_text.strip():
            final_text = "抱歉，我找不到明確的答案。"
        # Speak only what the live stream has not already said. The streamed text and
        # the authoritative final answer can differ (the stream is disabled the moment
        # an XML tool-call appears, or a tool step interleaved), so compare on the
        # common prefix rather than assuming they match.
        if any_streamed:
            common = 0
            for a, b in zip(streamed, final_text, strict=False):
                if a != b:
                    break
                common += 1
            _tail = final_text[common:] if len(final_text) > common else ""
            if _tail:
                _sf = final_text[:common]
                for tok in re.findall(r"\S\S*|\s+", _tail):
                    _sf += tok
                    yield _speak(tok, _sf)
                    await asyncio.sleep(0)
            yield {"type": "llm_done", "text": final_text}
            return
        _t0 = time.time()
        _sf = ""
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
        except Exception:
            web_search = None
            def format_results(x):
                # fallback shim: tools.web_search is absent, so there is nothing to
                # format — repr() is the honest degradation (E731: a def, not a lambda)
                return str(x)
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
                            _now = _dt.utcnow()
                            _tz = "UTC"
                        _tom = _now + _td(days=1)
                        _yest = _now - _td(days=1)
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
                    t0 = time.time()
                    _sf = ""
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
                t0 = time.time()
                first = True
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
        t0 = time.time()
        _sf = ""
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

