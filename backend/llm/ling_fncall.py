"""Ling 3.0 tool-calling, spoken through Qwen-Agent.

Qwen-Agent drives tool use over plain text: it appends a `# Tools` block to the system
message and then parses the model's reply back into structured calls. Its default
("nous") dialect puts the call in JSON:

    <tool_call>
    {"name": "web_search", "arguments": {"query": "台北天氣"}}
    </tool_call>

Ling 3.0's own chat template (inclusionAI/Ling-3.0-tiny/chat_template.jinja) asks for the
name inline and the arguments as tag pairs:

    <tool_call>web_search
    <arg_key>query</arg_key>
    <arg_value>台北天氣</arg_value>
    </tool_call>

Everything around the call — the `<tools>` block holding the JSON signatures, and
`<tool_response>` for results — is already the same in both, so only the call body needs
translating. This module supplies that translation as a drop-in `fncall_prompt`, which is
the single object Qwen-Agent uses for both directions:

    llm.fncall_prompt = LingFnCallPrompt()

Values are emitted verbatim rather than JSON-encoded, matching Ling's template (a bare
`台北天氣`, not `"台北天氣"`). On the way back, a value that parses as JSON is decoded so
numbers and booleans keep their type, and anything else is kept as the literal string —
so an argument that merely looks like JSON cannot break the call.
"""
import json
import re
from typing import List, Literal, Union

from qwen_agent.llm.fncall_prompts.base_fncall_prompt import BaseFnCallPrompt
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, FunctionCall, Message

FN_CALL_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
If you need to use a function, for each function call, output the function name and arguments within the following XML format:
<tool_call>{{function-name}}
<arg_key>{{arg-key-1}}</arg_key>
<arg_value>{{arg-value-1}}</arg_value>
<arg_key>{{arg-key-2}}</arg_key>
<arg_value>{{arg-value-2}}</arg_value>
...
</tool_call>
"""

# The name sits on the opening line; the pairs follow. `</tool_call>` may be missing when
# the reply is still streaming or the model simply forgot it, so the closing tag is
# optional and a truncated block still yields whatever pairs did arrive.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([^\n<]*)\n?(.*?)(?:</tool_call>|\Z)", re.S)
_ARG_PAIR_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|\Z)", re.S)


def _decode_value(raw: str):
    """Keep JSON-shaped values typed, everything else literal."""
    v = raw.strip()
    if v[:1] in '{["' or v in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", v):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def parse_ling_tool_calls(text: str) -> List[dict]:
    """Extract `[{name, arguments}]` from Ling's tag syntax. Pure function, unit-tested."""
    out = []
    for name, body in _TOOL_CALL_RE.findall(text or ""):
        name = name.strip()
        if not name:
            continue
        args = {k.strip(): _decode_value(v) for k, v in _ARG_PAIR_RE.findall(body)}
        out.append({"name": name, "arguments": args})
    return out


def render_ling_tool_call(name: str, arguments) -> str:
    """The inverse: a call in Ling's syntax, for replaying history back to the model."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    lines = [f"<tool_call>{name}"]
    for k, v in arguments.items():
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>")
    lines.append("</tool_call>")
    return "\n".join(lines)


class LingFnCallPrompt(BaseFnCallPrompt):
    """Qwen-Agent's fncall dialect, rewritten in Ling 3.0's tag syntax."""

    @staticmethod
    def preprocess_fncall_messages(messages: List[Message],
                                   functions: List[dict],
                                   lang: Literal['en', 'zh'] = 'en',
                                   parallel_function_calls: bool = True,
                                   function_choice: Union[Literal['auto'], str] = 'auto',
                                   **kwargs) -> List[Message]:
        del lang, parallel_function_calls
        if function_choice != 'auto':
            raise NotImplementedError('Ling fncall supports function_choice="auto" only')

        out: List[Message] = []
        for msg in messages:
            role, content = msg.role, (msg.content or [])
            if role in (SYSTEM, USER):
                out.append(msg)
            elif role == ASSISTANT:
                content = list(content)
                if msg.function_call:
                    content.append(ContentItem(
                        text=render_ling_tool_call(msg.function_call.name, msg.function_call.arguments)))
                # Consecutive assistant turns are merged, matching the nous prompt: the
                # model sees one uninterrupted turn rather than several.
                if out and out[-1].role == ASSISTANT:
                    out[-1].content.extend(content)
                else:
                    out.append(Message(role=role, content=content,
                                       reasoning_content=msg.reasoning_content))
            elif role == FUNCTION:
                wrapped = ([ContentItem(text='<tool_response>\n')] + list(content)
                           + [ContentItem(text='\n</tool_response>')])
                # A tool result is fed back as a user turn, which is what Ling's template
                # expects (it special-cases a user message wrapped in <tool_response>).
                if out and out[-1].role == USER:
                    out[-1].content.append(ContentItem(text='\n'))
                    out[-1].content.extend(wrapped)
                else:
                    out.append(Message(role=USER, content=wrapped))
            else:
                raise TypeError(f'unexpected role {role!r}')

        tool_descs = '\n'.join(json.dumps({'type': 'function', 'function': f}, ensure_ascii=False)
                               for f in functions)
        tool_system = FN_CALL_TEMPLATE.format(tool_descs=tool_descs)
        if out and out[0].role == SYSTEM:
            out[0].content.append(ContentItem(text='\n\n' + tool_system))
        else:
            out = [Message(role=SYSTEM, content=[ContentItem(text=tool_system)])] + out
        return out

    @staticmethod
    def postprocess_fncall_messages(messages: List[Message],
                                    parallel_function_calls: bool = True,
                                    function_choice: Union[Literal['auto'], str] = 'auto',
                                    **kwargs) -> List[Message]:
        del parallel_function_calls
        if function_choice != 'auto':
            raise NotImplementedError('Ling fncall supports function_choice="auto" only')

        new_messages: List[Message] = []
        tool_id = 1
        for msg in messages:
            role, content, extra = msg.role, msg.content, (msg.extra or {})
            if role in (SYSTEM, USER) or not isinstance(content, list):
                new_messages.append(msg)
                continue
            if msg.reasoning_content:
                new_messages.append(Message(role=role, content='',
                                            reasoning_content=msg.reasoning_content, extra=extra))

            pending: List[ContentItem] = []
            for item in content:
                item_type, item_text = item.get_type_and_value()
                if item_type != 'text':
                    pending.append(item)
                    continue
                head, sep, _ = item_text.partition('<tool_call>')
                if not sep:
                    if item_text:
                        pending.append(ContentItem(text=item_text))
                    continue
                if head.strip():
                    pending.append(ContentItem(text=head))
                for call in parse_ling_tool_calls(item_text):
                    if pending:
                        new_messages.append(Message(role=role, content=pending, extra=extra))
                        pending = []
                    _extra = dict(extra)
                    _extra['function_id'] = str(tool_id)
                    tool_id += 1
                    new_messages.append(Message(
                        role=ASSISTANT,
                        content=[],
                        function_call=FunctionCall(
                            name=call['name'],
                            arguments=json.dumps(call['arguments'], ensure_ascii=False),
                        ),
                        extra=_extra,
                    ))
            if pending:
                new_messages.append(Message(role=role, content=pending, extra=extra))
        return new_messages


def looks_like_ling(model_or_alias: str) -> bool:
    """Does this llama-server alias / model id name a Ling (Bailing) model?"""
    return bool(re.search(r"\bling\b|bailing", (model_or_alias or ""), re.I))
