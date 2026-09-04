"""End-to-end check that AgentLanguageModelHandler.process() actually publishes
into s2s.turn_trace as it runs -- not just that native_loop emits the right
event shapes (test_wire_format.py) or that turn_trace's own API works in
isolation (test_turn_trace.py), but that the handler's real _drive() wiring
between the two is connected. Drives the REAL _drive()/process() (unlike
tests/test_s2s_agent_handler.py's chunking tests, which override _drive()
entirely) through a fake harness object yielding raw event dicts, the same
shape agent/native_loop.py's run_turn produces.
"""
import asyncio
import sys
import threading
import unittest
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import GenerateResponseRequest

from s2s import turn_trace
from s2s.agent_handler import AgentLanguageModelHandler


class _FakeAgen:
    def __init__(self, events):
        self._it = iter(events)

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        pass


class _FakeLLM:
    def __init__(self, events):
        self._events = events

    def generate_chat_with_tools(self, *a, **k):
        return _FakeAgen(self._events)


class _Handler(AgentLanguageModelHandler):
    """setup() only, so the real _drive()/process() run unmodified."""

    def setup(self, harness_events=None, **kw):
        self.cancel_scope = None
        self.speculative_turns = None
        self.max_new_tokens = 64
        self.llm = _FakeLLM(harness_events or [])
        self.text_output_queue = Queue()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()


def _build(harness_events, transcript="今天台北天氣如何"):
    h = _Handler(Event(), Queue(), Queue(), setup_kwargs={"harness_events": harness_events})
    h._transcribe = lambda audio, sr: transcript
    req = GenerateResponseRequest(
        runtime_config=RuntimeConfig(),
        audio=np.zeros(2, dtype=np.float32), audio_sample_rate=16000,
        turn_id="t1", turn_revision=0,
    )
    return h, req


class TestTurnTraceIntegration(unittest.TestCase):
    def test_reasoning_tool_call_and_usage_reach_turn_trace(self):
        events = [
            {"type": "llm_reasoning", "text": "Let me check the weather. "},
            {"type": "tool_call", "name": "get_weather", "arguments": {"location": "台北"}},
            {"type": "tool_result", "name": "get_weather", "result": {"forecast": "sunny"}},
            {"type": "llm_usage", "input_tokens": 88, "output_tokens": 21},
            {"type": "llm_done", "text": "今天台北天氣晴朗。"},
        ]
        h, req = _build(events)
        list(h.process(req))

        snap = turn_trace.snapshot()
        self.assertEqual(snap["turn_id"], "t1")
        self.assertEqual(snap["reasoning"], "Let me check the weather. ")
        self.assertEqual(snap["steps"], [
            {"type": "tool_call", "name": "get_weather", "arguments": {"location": "台北"}},
            {"type": "tool_result", "name": "get_weather", "result": {"forecast": "sunny"}},
        ])
        self.assertEqual(snap["usage"], {"input_tokens": 88, "output_tokens": 21})
        self.assertTrue(snap["done"])

    def test_a_turn_with_no_reasoning_or_tools_leaves_an_empty_but_closed_trace(self):
        events = [{"type": "llm_done", "text": "你好。"}]
        h, req = _build(events)
        list(h.process(req))

        snap = turn_trace.snapshot()
        self.assertEqual(snap["reasoning"], "")
        self.assertEqual(snap["steps"], [])
        self.assertTrue(snap["done"])

    def test_a_new_turn_resets_the_previous_ones_trace(self):
        h1, req1 = _build([
            {"type": "llm_reasoning", "text": "stale reasoning"},
            {"type": "llm_done", "text": "ok"},
        ])
        list(h1.process(req1))
        self.assertEqual(turn_trace.snapshot()["reasoning"], "stale reasoning")

        h2, req2 = _build([{"type": "llm_done", "text": "ok2"}], transcript="another question")
        list(h2.process(req2))
        snap = turn_trace.snapshot()
        self.assertEqual(snap["turn_id"], "t1")   # both requests reuse turn_id "t1" in this fixture
        self.assertEqual(snap["reasoning"], "")


if __name__ == "__main__":
    unittest.main()
