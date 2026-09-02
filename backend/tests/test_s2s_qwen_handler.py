"""Contract tests for the Qwen-Agent LLM stage of the HF speech-to-speech pipeline.

These exercise the handler against fakes, so they run without llama-server, the
STT/TTS weights, or a live s2s pipeline. What they pin down is the part that is
easy to get wrong and expensive to debug live: sentence-sized chunking (the
LMOutputProcessor forwards to TTS 1:1), cancellation on barge-in, and the
retract/reset protocol the harness uses when it withdraws streamed text.
"""

import sys
import unittest
from pathlib import Path
from queue import Queue
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import make_assistant_message, make_system_message, make_user_message
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk

from s2s.qwen_agent_handler import QwenAgentLanguageModelHandler


class FakeCancelScope:
    def __init__(self):
        self._gen = 0

    @property
    def generation(self):
        return self._gen

    def cancel(self):
        self._gen += 1

    def is_stale(self, gen):
        return gen != self._gen


def make_runtime_config(turns):
    """turns: [(role, text)] -> a real RuntimeConfig carrying a real Chat."""
    rc = RuntimeConfig()
    for role, text in turns:
        if role == "system":
            rc.chat.init_chat(make_system_message(text))
        elif role == "user":
            rc.chat.add_item(make_user_message(text))
        else:
            rc.chat.add_item(make_assistant_message(text))
    return rc


class _Handler(QwenAgentLanguageModelHandler):
    """Skips setup()'s network/LLM construction; the harness is injected."""

    def setup(self, events=None, cancel_scope=None, **kw):
        self.cancel_scope = cancel_scope
        self.speculative_turns = None
        self.max_new_tokens = 64
        self._events = events or []
        self.llm = None
        self._loop = None

    def _drive(self, history, prompt):
        self.seen = (history, prompt)
        yield from self._events


def build(events, chat_turns, cancel_scope=None):
    h = _Handler(
        Event(), Queue(), Queue(),
        setup_kwargs={"events": events, "cancel_scope": cancel_scope},
    )
    req = GenerateResponseRequest(
        runtime_config=make_runtime_config(chat_turns),
        turn_id="t1", turn_revision=0,
    )
    return h, req


class TestChunking(unittest.TestCase):
    def test_flushes_on_cjk_sentence_end(self):
        h, req = build([("delta", "今天天氣很好。"), ("delta", "明天會下雨。")],
                       [("user", "天氣如何")])
        spoken = [o.text for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertEqual(spoken, ["今天天氣很好。", "明天會下雨。"])

    def test_partial_sentence_is_buffered_then_flushed_at_end(self):
        h, req = build([("delta", "台灣"), ("delta", "半導體"), ("delta", "很強")],
                       [("user", "x")])
        spoken = [o.text for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertEqual(spoken, ["台灣半導體很強"], "no ender -> one flush at completion")

    def test_always_terminates_with_end_of_response(self):
        h, req = build([("delta", "好。")], [("user", "x")])
        out = list(h.process(req))
        self.assertIsInstance(out[-1], EndOfResponse)
        self.assertIsNone(out[-1].error)

    def test_short_cjk_sentence_flushes_immediately(self):
        # "好。" is a complete reply; a character-count floor would hold it back
        # and delay first audio for the whole turn.
        h, req = build([("delta", "好。"), ("delta", "還有別的問題嗎？")], [("user", "x")])
        spoken = [o.text for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertEqual(spoken, ["好。", "還有別的問題嗎？"])

    def test_numbered_list_marker_is_not_spoken_as_its_own_chunk(self):
        h, req = build([("delta", "1. 台積電。"), ("delta", "2. 聯發科。")], [("user", "x")])
        spoken = [o.text for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertTrue(all(t.strip() not in ("1.", "2.") for t in spoken),
                        f"list markers must not reach TTS alone: {spoken}")

    def test_long_clause_without_punctuation_still_reaches_tts(self):
        h, req = build([("delta", "a" * 300)], [("user", "x")])
        spoken = [o.text for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertGreater(len(spoken), 1, "a run-on must not block TTS until the end")


class TestCancellation(unittest.TestCase):
    def test_barge_in_stops_generation_midstream(self):
        scope = FakeCancelScope()
        events = [("delta", "第一句。"), ("cancel", ""), ("delta", "第二句。"), ("delta", "第三句。")]

        class H(_Handler):
            def _drive(self, history, prompt):
                for kind, payload in events:
                    if kind == "cancel":
                        scope.cancel()      # user starts talking
                        continue
                    yield kind, payload

        h = H(Event(), Queue(), Queue(), setup_kwargs={"cancel_scope": scope})
        req = GenerateResponseRequest(
            runtime_config=make_runtime_config([("user", "x")]),
            turn_id="t1", turn_revision=0,
        )
        out = list(h.process(req))
        spoken = [o.text for o in out if isinstance(o, LLMResponseChunk)]
        self.assertEqual(spoken, ["第一句。"], "nothing after the barge-in may reach TTS")
        self.assertIsInstance(out[-1], EndOfResponse)

    def test_chunks_carry_the_generation_they_were_produced_in(self):
        scope = FakeCancelScope()
        h, req = build([("delta", "好。")], [("user", "x")], cancel_scope=scope)
        chunks = [o for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertTrue(all(c.cancel_generation == 0 for c in chunks),
                        "the send loop drops output whose generation is stale")


class TestRetraction(unittest.TestCase):
    def test_reset_drops_only_unspoken_buffer(self):
        # Harness streams a partial clause, then retracts it as reasoning.
        h, req = build([("delta", "Let me think"), ("reset", ""), ("delta", "答案是七。")],
                       [("user", "x")])
        spoken = [o.text for o in h.process(req) if isinstance(o, LLMResponseChunk)]
        self.assertEqual(spoken, ["答案是七。"], "retracted text must never be spoken")


class TestPromptExtraction(unittest.TestCase):
    def test_newest_user_turn_is_the_prompt_and_rest_is_history(self):
        h, req = build([("delta", "ok。")],
                       [("system", "sys"), ("user", "第一問"), ("assistant", "第一答"), ("user", "第二問")])
        list(h.process(req))
        history, prompt = h.seen
        self.assertEqual(prompt, "第二問")
        self.assertEqual(history, [{"role": "user", "content": "第一問"},
                                   {"role": "assistant", "content": "第一答"}])

    def test_empty_prompt_ends_the_turn_without_calling_the_model(self):
        h, req = build([("delta", "should not run")], [("system", "sys")])
        out = list(h.process(req))
        self.assertEqual([o for o in out if isinstance(o, LLMResponseChunk)], [])
        self.assertIsInstance(out[-1], EndOfResponse)


if __name__ == "__main__":
    unittest.main()
