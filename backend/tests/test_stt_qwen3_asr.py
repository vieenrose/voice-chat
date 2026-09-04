import unittest
from unittest.mock import MagicMock, patch

from s2s import stt_qwen3_asr as stt


class TestTranscribe(unittest.TestCase):
    def setUp(self):
        stt._processor = MagicMock()
        stt._model = MagicMock()

    def tearDown(self):
        stt._processor = None
        stt._model = None

    def _wire(self, decoded_text):
        inputs = MagicMock()
        inputs.to.return_value = inputs
        inputs.__getitem__.return_value = MagicMock(shape=(1, 5))
        stt._processor.apply_transcription_request.return_value = inputs
        stt._processor.decode.return_value = [decoded_text]
        return inputs

    def test_transcribes_and_strips_the_result(self):
        self._wire("  你好嗎  ")
        with patch.object(stt, "_load"):
            text = stt.transcribe(b"\x00\x01" * 8000, 16000)
        self.assertEqual(text, "你好嗎")
        stt._processor.apply_transcription_request.assert_called_once()
        self.assertEqual(stt._processor.decode.call_args.kwargs["return_format"],
                          "transcription_only")

    def test_passes_the_configured_language(self):
        self._wire("hi")
        with patch.object(stt, "_load"), patch.object(stt, "_LANGUAGE", "zh"):
            stt.transcribe(b"\x00\x01" * 8000, 16000)
        self.assertEqual(
            stt._processor.apply_transcription_request.call_args.kwargs["language"], "zh")

    def test_a_load_failure_returns_empty_rather_than_raising(self):
        with patch.object(stt, "_load", side_effect=RuntimeError("no GPU")):
            self.assertEqual(stt.transcribe(b"\x00\x01" * 8000, 16000), "")

    def test_a_generation_failure_returns_empty_rather_than_raising(self):
        stt._processor.apply_transcription_request.side_effect = RuntimeError("bad audio")
        with patch.object(stt, "_load"):
            self.assertEqual(stt.transcribe(b"\x00\x01" * 8000, 16000), "")


class TestLoad(unittest.TestCase):
    def tearDown(self):
        stt._processor = None
        stt._model = None

    def test_load_is_a_no_op_once_the_model_is_set(self):
        stt._model = MagicMock()
        stt._processor = MagicMock()
        with patch("transformers.AutoProcessor") as ap:
            stt._load()
        ap.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
