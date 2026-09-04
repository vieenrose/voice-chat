"""Coverage for OpenCode Go's two supported wire formats:

- agent.qwen_harness._wire_format / _endpoint: which format a (base, model)
  pair should use.
- agent.native_loop._run_turn_responses_api: the OpenAI Responses API loop
  added for Muse Spark / Grok / GPT-5.6 Luna, alongside the existing Chat
  Completions loop those tests already cover implicitly via the agent handler
  tests. httpx is mocked throughout -- no network, no real model server.
"""
import os
import unittest
from unittest import mock

from agent import native_loop, qwen_harness


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeClient:
    """One per call to httpx.Client(...); .stream() replays one scripted step."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, json=None, headers=None):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return _FakeStreamResponse(self._steps.pop(0))


def _sse(events):
    return [f"data: {__import__('json').dumps(e)}" for e in events] + ["data: [DONE]"]


class _EchoTool:
    name = "get_weather"
    description = "d"
    parameters = {"type": "object", "properties": {}}

    def call(self, args):
        return "sunny"


class TestWireFormat(unittest.TestCase):
    def test_local_base_is_always_chat_completions(self):
        self.assertEqual(qwen_harness._wire_format("http://127.0.0.1:11435/v1", "muse-spark-1.3-contributor"),
                          "chat_completions")

    def test_a_responses_api_model_on_a_hosted_base_is_detected(self):
        self.assertEqual(
            qwen_harness._wire_format("https://opencode.ai/zen/go/v1", "muse-spark-1.3-contributor"),
            "responses")
        self.assertEqual(qwen_harness._wire_format("https://opencode.ai/zen/go/v1", "grok-4.6"), "responses")

    def test_a_chat_completions_model_on_a_hosted_base_is_unaffected(self):
        self.assertEqual(qwen_harness._wire_format("https://opencode.ai/zen/go/v1", "mimo-v2.5"),
                          "chat_completions")

    def test_endpoint_reports_the_wire_format_as_its_fourth_element(self):
        with mock.patch.dict(os.environ, {"LLM_API_BASE": "https://opencode.ai/zen/go/v1",
                                          "LLM_MODEL_ID": "muse-spark-1.3-contributor",
                                          "LLM_API_KEY": "sk-test"}):
            base, model, key, wire = qwen_harness._endpoint()
        self.assertEqual((base, model, key, wire),
                          ("https://opencode.ai/zen/go/v1", "muse-spark-1.3-contributor",
                           "sk-test", "responses"))


class TestResponsesApiLoop(unittest.TestCase):
    def _run(self, steps, tools=None):
        client = _FakeClient(steps)
        with mock.patch("agent.native_loop.httpx.Client", return_value=client):
            answer = native_loop._run_turn_responses_api(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                tools or {}, api_base="https://opencode.ai/zen/go/v1", model="muse-spark-1.3-contributor",
                api_key="sk-test", generate_cfg={"temperature": 0.2, "max_tokens": 64})
        return answer, client

    def test_a_plain_text_reply_needs_one_step(self):
        steps = [_sse([
            {"type": "response.output_text.delta", "delta": "台北"},
            {"type": "response.output_text.delta", "delta": "是首都。"},
        ])]
        answer, client = self._run(steps)
        self.assertEqual(answer, "台北是首都。")
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["url"].endswith("/responses"))
        self.assertEqual(client.calls[0]["headers"]["Authorization"], "Bearer sk-test")

    def test_the_request_carries_flat_tool_schemas_and_translated_sampling_params(self):
        steps = [_sse([{"type": "response.output_text.delta", "delta": "ok"}])]
        _, client = self._run(steps, tools={"get_weather": _EchoTool()})
        body = client.calls[0]["json"]
        self.assertEqual(body["tools"], [{"type": "function", "name": "get_weather",
                                          "description": "d", "parameters": _EchoTool.parameters}])
        self.assertEqual(body["max_output_tokens"], 64)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["instructions"], "sys")

    def test_a_tool_call_is_executed_and_the_result_fed_back(self):
        steps = [
            _sse([
                {"type": "response.output_item.added",
                 "item": {"type": "function_call", "call_id": "c1", "name": "get_weather"}},
                {"type": "response.function_call_arguments.delta", "call_id": "c1", "delta": "{}"},
                {"type": "response.output_item.done",
                 "item": {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}},
            ]),
            _sse([{"type": "response.output_text.delta", "delta": "今天晴天。"}]),
        ]
        answer, client = self._run(steps, tools={"get_weather": _EchoTool()})
        self.assertEqual(answer, "今天晴天。")
        self.assertEqual(len(client.calls), 2)
        second_input = client.calls[1]["json"]["input"]
        kinds = [item.get("type") for item in second_input]
        self.assertIn("function_call", kinds)
        self.assertIn("function_call_output", kinds)
        output_item = next(i for i in second_input if i.get("type") == "function_call_output")
        self.assertEqual(output_item["call_id"], "c1")
        self.assertIn("sunny", output_item["output"])

    def test_an_unknown_tool_call_gets_a_sanitised_error_result_not_a_crash(self):
        steps = [
            _sse([
                {"type": "response.output_item.added",
                 "item": {"type": "function_call", "call_id": "c1", "name": "no_such_tool"}},
                {"type": "response.output_item.done",
                 "item": {"type": "function_call", "call_id": "c1", "name": "no_such_tool", "arguments": "{}"}},
            ]),
            _sse([{"type": "response.output_text.delta", "delta": "done"}]),
        ]
        answer, client = self._run(steps)
        self.assertEqual(answer, "done")
        second_input = client.calls[1]["json"]["input"]
        output_item = next(i for i in second_input if i.get("type") == "function_call_output")
        self.assertIn("no such tool", output_item["output"])

    def test_a_response_failed_event_raises_rather_than_returning_silently(self):
        steps = [_sse([{"type": "response.failed", "response": {"error": {"message": "boom"}}}])]
        with self.assertRaises(RuntimeError):
            self._run(steps)


if __name__ == "__main__":
    unittest.main()
