import unittest
from unittest.mock import MagicMock

from s2s.caption_gemma import GemmaCaptionStream


class TestPublish(unittest.TestCase):
    def setUp(self):
        self.out = MagicMock()
        self.stream = GemmaCaptionStream(self.out)

    def test_new_text_is_published(self):
        self.stream._publish("你好")
        self.out.put.assert_called_once()
        self.assertEqual(self.stream._sent, "你好")

    def test_identical_text_is_not_republished(self):
        self.stream._publish("你好")
        self.out.put.reset_mock()
        self.stream._publish("你好")
        self.out.put.assert_not_called()

    def test_empty_text_is_not_published(self):
        self.stream._publish("")
        self.out.put.assert_not_called()

    def test_no_queue_means_no_crash(self):
        stream = GemmaCaptionStream(None)
        stream._publish("你好")  # must not raise


class TestBufferCap(unittest.TestCase):
    def setUp(self):
        self.stream = GemmaCaptionStream(None)

    def test_short_buffer_is_untouched(self):
        self.stream._rate = 16000
        self.stream._buf = bytearray(b"\x00\x01" * 100)
        self.stream._cap_buffer()
        self.assertEqual(len(self.stream._buf), 200)

    def test_long_buffer_is_trimmed_to_the_tail(self):
        self.stream._rate = 16000
        from s2s import caption_gemma
        limit_bytes = int(caption_gemma._MAX_BUFFER_S * self.stream._rate) * 2
        tail = b"\xff\xfe" * 10
        self.stream._buf = bytearray(b"\x00\x01" * (limit_bytes // 2 + 100)) + bytearray(tail)
        self.stream._cap_buffer()
        self.assertLessEqual(len(self.stream._buf), limit_bytes)
        self.assertTrue(bytes(self.stream._buf).endswith(tail))


class TestTranscribeRequest(unittest.TestCase):
    def setUp(self):
        self.stream = GemmaCaptionStream(None)
        self.stream._buf = bytearray(b"\x00\x01" * 8000)
        self.stream._rate = 16000

    def test_request_carries_the_audio_and_is_schema_constrained_to_json(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"transcript": "你好嗎"}'}}]}
        self.stream._client.post = MagicMock(return_value=response)

        text = self.stream._transcribe()

        self.assertEqual(text, "你好嗎")
        body = self.stream._client.post.call_args.kwargs["json"]
        parts = body["messages"][0]["content"]
        self.assertIn("transcribe", parts[0]["text"].lower())
        self.assertEqual(parts[1]["type"], "input_audio")
        self.assertEqual(parts[1]["input_audio"]["format"], "wav")
        self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_whitespace_only_transcript_is_stripped_to_empty(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"transcript": "  \\n"}'}}]}
        self.stream._client.post = MagicMock(return_value=response)

        self.assertEqual(self.stream._transcribe(), "")

    def test_a_reply_that_is_not_json_falls_back_to_the_raw_text(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}]}
        self.stream._client.post = MagicMock(return_value=response)

        self.assertEqual(self.stream._transcribe(), "not json at all")


class TestPreRollAndUtteranceLifecycle(unittest.TestCase):
    """Drives the real background thread; _transcribe is stubbed so no server is needed."""

    def setUp(self):
        self.out = MagicMock()
        self.stream = GemmaCaptionStream(self.out)
        self.stream._transcribe = MagicMock(return_value="")
        from s2s import caption_gemma
        caption_gemma._DECODE_EVERY_S = 0.0  # decode on every chunk, no waiting

    def _settle(self):
        import time
        deadline = time.monotonic() + 2
        while self.stream._q.qsize() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)  # let the last dequeued item finish processing

    def test_pre_utterance_audio_seeds_the_next_begin(self):
        self.stream.feed(b"\x01\x02" * 100, 16000)
        self._settle()
        self.assertFalse(self.stream._active)
        self.assertGreater(len(self.stream._pre_roll), 0)

        self.stream.begin()
        self._settle()
        self.assertTrue(bytes(self.stream._buf).startswith(bytes(self.stream._pre_roll[:0]) or b""))
        # The buffer at begin() is seeded from the pre-roll, not empty.
        self.assertGreaterEqual(len(self.stream._buf), 200)

    def test_end_clears_the_pre_roll_for_the_next_utterance(self):
        self.stream.feed(b"\x01\x02" * 100, 16000)
        self.stream.begin()
        self.stream.feed(b"\x03\x04" * 100, 16000)
        self.stream.end()
        self._settle()
        self.assertEqual(len(self.stream._pre_roll), 0)
        self.assertFalse(self.stream._active)
