"""Ling 3.0's tool-call dialect, as consumed by Qwen-Agent.

The format comes from inclusionAI/Ling-3.0-tiny/chat_template.jinja:

    <tool_call>{function-name}
    <arg_key>{k}</arg_key>
    <arg_value>{v}</arg_value>
    </tool_call>
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.ling_fncall import (  # noqa: E402
    LingFnCallPrompt, looks_like_ling, parse_ling_tool_calls, render_ling_tool_call,
)
from qwen_agent.llm.schema import ASSISTANT, ContentItem, Message, SYSTEM, USER  # noqa: E402


class TestParse(unittest.TestCase):

    def test_the_shape_the_template_documents(self):
        calls = parse_ling_tool_calls(
            '<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>台北天氣</arg_value>\n</tool_call>')
        self.assertEqual(calls, [{"name": "web_search", "arguments": {"query": "台北天氣"}}])

    def test_multiple_arguments_keep_their_order_and_values(self):
        calls = parse_ling_tool_calls(
            '<tool_call>get_weather\n'
            '<arg_key>location</arg_key>\n<arg_value>台北</arg_value>\n'
            '<arg_key>date</arg_key>\n<arg_value>today</arg_value>\n</tool_call>')
        self.assertEqual(calls[0]["arguments"], {"location": "台北", "date": "today"})

    def test_prose_before_the_call_is_not_swallowed_and_the_call_is_still_found(self):
        calls = parse_ling_tool_calls(
            '好的，我來查詢。\n<tool_call>web_search\n<arg_key>query</arg_key>\n'
            '<arg_value>最新科技新聞</arg_value>\n</tool_call>')
        self.assertEqual(calls, [{"name": "web_search", "arguments": {"query": "最新科技新聞"}}])

    def test_two_calls_in_one_reply(self):
        calls = parse_ling_tool_calls(
            '<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>台北</arg_value>\n</tool_call>\n'
            '<tool_call>get_current_datetime\n<arg_key>timezone</arg_key>\n'
            '<arg_value>Asia/Taipei</arg_value>\n</tool_call>')
        self.assertEqual([c["name"] for c in calls], ["get_weather", "get_current_datetime"])

    def test_a_truncated_call_still_yields_the_arguments_that_arrived(self):
        """Streaming, or a model that forgot the closing tag. The JSON dialect lost a
        whole call to this (it produced `{"query": "…"` and searched for that string
        verbatim); the tag syntax degrades to the pairs it actually got."""
        calls = parse_ling_tool_calls(
            '<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>最新科技新聞')
        self.assertEqual(calls, [{"name": "web_search", "arguments": {"query": "最新科技新聞"}}])

    def test_typed_values_survive_and_free_text_stays_literal(self):
        calls = parse_ling_tool_calls(
            '<tool_call>t\n<arg_key>n</arg_key>\n<arg_value>5</arg_value>\n'
            '<arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n'
            '<arg_key>q</arg_key>\n<arg_value>1 + 1 是多少</arg_value>\n</tool_call>')
        self.assertEqual(calls[0]["arguments"], {"n": 5, "flag": True, "q": "1 + 1 是多少"})

    def test_a_value_that_merely_looks_like_json_does_not_break_the_call(self):
        calls = parse_ling_tool_calls(
            '<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>{unclosed</arg_value>\n</tool_call>')
        self.assertEqual(calls[0]["arguments"], {"query": "{unclosed"})

    def test_text_with_no_call_yields_nothing(self):
        self.assertEqual(parse_ling_tool_calls('台北今天多雲，最高溫 33 度。'), [])
        self.assertEqual(parse_ling_tool_calls(''), [])


class TestRoundTrip(unittest.TestCase):

    def test_render_then_parse_is_identity(self):
        for name, args in [
            ("web_search", {"query": "最新科技新聞"}),
            ("get_weather", {"location": "台北", "date": "today"}),
            ("get_current_datetime", {"timezone": "Asia/Taipei"}),
        ]:
            parsed = parse_ling_tool_calls(render_ling_tool_call(name, args))
            self.assertEqual(parsed, [{"name": name, "arguments": args}], name)

    def test_render_accepts_the_json_string_qwen_agent_stores(self):
        out = render_ling_tool_call("web_search", json.dumps({"query": "台北天氣"}, ensure_ascii=False))
        self.assertEqual(parse_ling_tool_calls(out)[0]["arguments"], {"query": "台北天氣"})


class TestPromptPlumbing(unittest.TestCase):

    FUNCTIONS = [{"name": "web_search", "description": "search",
                  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]

    def test_preprocess_documents_lings_syntax_not_the_json_one(self):
        out = LingFnCallPrompt.preprocess_fncall_messages(
            [Message(role=USER, content=[ContentItem(text="台北天氣？")])],
            functions=self.FUNCTIONS, lang="zh")
        sys_text = "".join(c.text or "" for c in out[0].content)
        self.assertIn("<arg_key>", sys_text)
        self.assertIn("<tools>", sys_text)
        self.assertNotIn('{"name": <function-name>', sys_text)

    def test_postprocess_turns_a_reply_into_a_function_call_message(self):
        out = LingFnCallPrompt.postprocess_fncall_messages([Message(
            role=ASSISTANT,
            content=[ContentItem(text='<tool_call>web_search\n<arg_key>query</arg_key>\n'
                                      '<arg_value>台北天氣</arg_value>\n</tool_call>')])])
        calls = [m for m in out if m.function_call]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function_call.name, "web_search")
        self.assertEqual(json.loads(calls[0].function_call.arguments), {"query": "台北天氣"})

    def test_prose_and_call_become_separate_messages(self):
        out = LingFnCallPrompt.postprocess_fncall_messages([Message(
            role=ASSISTANT,
            content=[ContentItem(text='好的。\n<tool_call>web_search\n<arg_key>query</arg_key>\n'
                                      '<arg_value>x</arg_value>\n</tool_call>')])])
        self.assertTrue(any(m.function_call for m in out))
        self.assertTrue(any((m.content and "好的" in "".join(c.text or "" for c in m.content))
                            for m in out if not m.function_call))

    def test_a_plain_answer_passes_through_untouched(self):
        out = LingFnCallPrompt.postprocess_fncall_messages(
            [Message(role=ASSISTANT, content=[ContentItem(text="台北今天多雲。")])])
        self.assertFalse(any(m.function_call for m in out))
        self.assertIn("台北今天多雲。", "".join(c.text or "" for c in out[0].content))

    def test_system_message_is_extended_not_replaced(self):
        out = LingFnCallPrompt.preprocess_fncall_messages(
            [Message(role=SYSTEM, content=[ContentItem(text="You are helpful.")]),
             Message(role=USER, content=[ContentItem(text="hi")])],
            functions=self.FUNCTIONS, lang="en")
        sys_text = "".join(c.text or "" for c in out[0].content)
        self.assertIn("You are helpful.", sys_text)
        self.assertIn("# Tools", sys_text)


class TestModelDetection(unittest.TestCase):

    def test_only_ling_models_get_the_ling_dialect(self):
        for yes in ("ling-3.0-tiny", "Ling-3.0-tiny-Q8_0", "bailingmoe3", "LING"):
            self.assertTrue(looks_like_ling(yes), yes)
        for no in ("qwen3.5-2b", "qwen3.5-4b", "apodex-0.8b-q8", "", None):
            self.assertFalse(looks_like_ling(no), no)

    def test_a_word_merely_containing_ling_is_not_a_ling_model(self):
        # "sterling", "darling" — the alias is matched as a word, not a substring.
        self.assertFalse(looks_like_ling("sterling-7b"))


if __name__ == "__main__":
    unittest.main()
