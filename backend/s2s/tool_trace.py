"""Recent tool results, for the UI's lookup panel.

Its own module on purpose. The pipeline starts as ``python3 -m s2s.serve``, so
serve.py runs as ``__main__``; a ``from s2s.serve import ...`` elsewhere then
imports a SECOND copy of it, with its own module-level state. The route read one
deque while the LLM stage appended to the other, and the panel stayed empty.

The Realtime protocol has no server->client tool-result event -- ``ServerEvent``
is a closed union of OpenAI types -- so rather than fork the framework this is
published over the same HTTP surface as /v1/vram and the page reads it.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

TOOL_TRACE: deque = deque(maxlen=8)


def record(name: str, arguments: dict | None, result: Any) -> None:
    """Called by the LLM stage when a tool returns. Never raises into a turn."""
    try:
        TOOL_TRACE.append({"ts": time.time(), "name": name,
                           "arguments": arguments or {}, "result": result})
    except Exception:
        logger.exception("could not record a tool trace")


def snapshot() -> list[dict]:
    return list(TOOL_TRACE)
