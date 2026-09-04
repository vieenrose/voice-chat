"""Live reasoning + tool-call + token-usage trace for the turn IN PROGRESS, for
the UI's collapsible "thinking" panel and its token-usage readout -- and, via
`set_answer()`, a fallback path for the reply text itself when the framework's
own transcript event goes missing (see that function's docstring).

Separate from tool_trace.py, which is the directory-lookup panel's own history
ACROSS turns (a deque, never cleared): this holds only the current turn, reset
at begin() and frozen at end(). The frontend polls it while a turn is running,
copies the last-seen snapshot into that turn's own bubble once end() flips
`done`, and then stops polling -- so only ONE turn's trace needs to exist
server-side at a time; per-turn history lives in the browser instead.

Thread-safe because it is written from the harness's worker thread
(agent/native_loop.py, via agent_handler._drive()) and read from the FastAPI
request thread handling GET /v1/turn-trace.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_current: dict[str, Any] = {
    "turn_id": None, "reasoning": "", "steps": [], "usage": None, "answer": "", "done": True,
}


def begin(turn_id: str | None) -> None:
    with _lock:
        _current.update(turn_id=turn_id, reasoning="", steps=[], usage=None, answer="", done=False)


def add_reasoning(delta: str) -> None:
    if not delta:
        return
    with _lock:
        _current["reasoning"] += delta


def add_tool_call(name: str, arguments: Any) -> None:
    with _lock:
        _current["steps"].append({"type": "tool_call", "name": name, "arguments": arguments})


def add_tool_result(name: str, result: Any) -> None:
    with _lock:
        _current["steps"].append({"type": "tool_result", "name": name, "result": result})


def set_usage(input_tokens: int | None, output_tokens: int | None) -> None:
    """Record token usage for this turn. Called once the provider reports it --
    typically only on the final streamed chunk/event, not per-token."""
    with _lock:
        _current["usage"] = {"input_tokens": input_tokens, "output_tokens": output_tokens}


def set_answer(text: str) -> None:
    """Record the final answer text this stage actually assembled.

    Exists because of a real, observed framework race, not on spec: speech_to_
    speech's own "speculative turn reopen" mechanism (VAD/vad_handler.py) can
    bump a turn's tracked revision while its answer is still being generated --
    a slow, multi-tool-call turn is exactly the case with room for it. Two
    different gates check that revision for the SAME chunk: LMOutputProcessor
    checks it once, synchronously, before forwarding to TTS; ResponseHandler
    .on_assistant_text checks it AGAIN, later, when the queued client-facing
    event is actually processed, via commit_if_latest_after_reopen_grace. If
    the revision moved in between, gate 1 already let the chunk reach TTS --
    it gets spoken -- while gate 2 silently drops the corresponding
    response.output_audio_transcript event, and the client never learns the
    text at all. Confirmed live: server logs show the TTS stage receiving and
    speaking the correct final sentence while the exported session's chat
    bubble for that same turn stayed empty.

    This module's own state never passes through that gate -- it is filled
    directly from agent_handler._drive(), the same source native_loop's
    events come from, not from the Realtime protocol's transcript path. The
    frontend uses this as a fallback ONLY when the live transcript truly never
    arrived (see App.svelte's pollTrace), not as the primary source: the live
    one still wins whenever it works, since it can render sentence-by-sentence
    as the turn streams, and this is only known once the whole turn is done.
    """
    with _lock:
        _current["answer"] = text


def end() -> None:
    with _lock:
        _current["done"] = True


def snapshot() -> dict[str, Any]:
    with _lock:
        # A shallow dict(...) would still alias `steps` (a list, mutated by
        # add_tool_call/add_tool_result) with the internal state -- copying it
        # too means the caller can read its result after releasing the lock
        # without racing a concurrent write.
        return {**_current, "steps": list(_current["steps"])}
