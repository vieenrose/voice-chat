"""Live reasoning + tool-call + token-usage trace for the turn IN PROGRESS, for
the UI's collapsible "thinking" panel and its token-usage readout.

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
_current: dict[str, Any] = {"turn_id": None, "reasoning": "", "steps": [], "usage": None, "done": True}


def begin(turn_id: str | None) -> None:
    with _lock:
        _current.update(turn_id=turn_id, reasoning="", steps=[], usage=None, done=False)


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
