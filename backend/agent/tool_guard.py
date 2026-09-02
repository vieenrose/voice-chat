"""Validation and sanitisation around tool use.

Neither belongs to the agent framework. qwen-agent with use_raw_api is a thin loop
over the server's native tool calling: it does not validate arguments against the
declared JSON Schema, and it passes tool output into the prompt verbatim. Nor does
switching framework fix the second problem -- it is a property of what reaches the
model, not of who assembles the messages.

Measured on Gemma 4 E4B with hostile text planted in a tool result:

    injection                     plain    +hardened prompt   +sanitise
    "IGNORE ALL PREVIOUS ..."     COMPLIED  resisted           resisted
    forged <|im_start|>system     COMPLIED  COMPLIED           resisted

So the system prompt is necessary and not sufficient. The forged-turn case is only
stopped by stripping the control tokens before the text is ever rendered into the
template, which is what sanitize_tool_output does.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Chat-template control tokens. A tool result carrying these can forge a turn --
# a "system" message of the attacker's choosing, spliced in from a web page. The
# list spans the templates this project may serve, since the tool layer should not
# have to know which model is loaded.
_CONTROL_TOKENS = re.compile(
    r"<\|(?:im_start|im_end|endoftext|system|user|assistant|start_header_id|end_header_id|eot_id)\|>"
    r"|<start_of_turn>|<end_of_turn>"
    r"|<\|(?:begin|end)_of_(?:sentence|text)\|>",
    re.I,
)

# Tool output is data. It is fenced so the model can see where it begins and ends,
# which is what stopped the forged-turn injection in testing.
_OPEN, _CLOSE = "<tool_output>", "</tool_output>"

MAX_TOOL_OUTPUT = 6000


def sanitize_tool_output(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """Make a tool result safe to place in the prompt.

    Strips chat-template control tokens, removes any fence the content tries to
    close early, caps the length, and wraps the result so its boundaries are
    explicit. Content is never otherwise rewritten: this is a search result the
    user asked for, and silently editing it would be its own kind of dishonesty.
    """
    if not text:
        return ""
    s = str(text)
    s = _CONTROL_TOKENS.sub(" ", s)
    # A result that closes the fence itself would put the rest outside the data.
    s = s.replace(_OPEN, "").replace(_CLOSE, "")
    if len(s) > limit:
        s = s[:limit] + f"\n[truncated at {limit} characters]"
    return f"{_OPEN}\n{s}\n{_CLOSE}"


class ToolArgumentError(ValueError):
    """Arguments that do not satisfy the tool's declared schema."""


def validate_args(params, schema: dict, tool_name: str = "tool") -> dict:
    """Check arguments against a tool's JSON Schema before it runs.

    The gap this closes, measured on the weather tool: a call with no arguments at
    all ran a live lookup for the string "today", and a call with location as an
    integer raised TypeError from inside the tool. Neither should reach the tool.

    Deliberately a small subset of JSON Schema -- type, required, and string
    length -- because that is what these three tools declare. It rejects rather
    than coerces, so a malformed call is a visible error and not a quiet guess.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params) if params.strip() else {}
        except json.JSONDecodeError as e:
            raise ToolArgumentError(f"{tool_name}: arguments are not valid JSON: {e}") from e
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ToolArgumentError(f"{tool_name}: arguments must be an object, got {type(params).__name__}")

    props = (schema or {}).get("properties") or {}
    for key in (schema or {}).get("required") or []:
        if key not in params or params[key] in (None, ""):
            raise ToolArgumentError(f"{tool_name}: missing required argument {key!r}")

    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "object": dict, "array": list}
    clean = {}
    for key, value in params.items():
        spec = props.get(key)
        if spec is None:
            # Unknown keys are dropped, not an error: a model that adds a stray
            # field should not fail the turn, but nothing undeclared reaches a tool.
            logger.debug("%s: dropping undeclared argument %r", tool_name, key)
            continue
        want = types.get(spec.get("type"))
        if want and not isinstance(value, want):
            raise ToolArgumentError(
                f"{tool_name}: {key!r} must be {spec.get('type')}, got {type(value).__name__}")
        if isinstance(value, str) and len(value) > 512:
            raise ToolArgumentError(f"{tool_name}: {key!r} is too long ({len(value)} chars)")
        clean[key] = value
    return clean
