"""
Last-resort mock STT adapter — the terminal rung of pipeline/speech_to_speech.py's
STT import ladder.

Why this exists: every rung of that ladder had a hard third-party import
(sherpa-onnx, funasr, ARK, whisper), so on a machine with none of them installed the
ladder's LAST rung raised ImportError out of the module import and `python app.py
--mock` could not start at all — i.e. the flag documented as "run without downloading
models" only worked if the model libraries were still installed. With this rung the
service always boots, and /health reports stt backend = "mock".

Emits stt_partial/stt_final from energy-based utterance segmentation so the whole
pipeline (barge-in, turn_id, WS protocol, frontend) is exercisable without weights.
"""
import asyncio
import time
from typing import AsyncGenerator
import numpy as np
from loguru import logger

SAMPLE_RATE = 16000
_MOCK_WORDS = ["hello", "how", "are", "you", "today", "voice", "assistant", "streaming"]


class StreamingXASR:
    def __init__(self, model_id: str = "mock", device: str = "cpu", mock: bool = True, chunk_ms: int = 160, **kwargs):
        self.model_id = model_id
        self.device = device
        self.mock = True
        self.chunk_ms = chunk_ms
        self.sample_rate = SAMPLE_RATE
        self.backend = "mock"
        logger.warning("STT: MOCK adapter in use — no recognizer available. "
                       "Transcripts are fake; install sherpa-onnx for the real stack.")

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event, session_id: str | None = None) -> AsyncGenerator[dict, None]:
        utterance = np.zeros(0, dtype=np.float32)
        partial = ""
        words = 0
        last_partial_t = time.time()
        last_speech_t = time.time()
        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(pcm_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                item = None
            now = time.time()
            if item is not None:
                if isinstance(item, dict) and item.get("type") == "flush":
                    if len(utterance) > 0:
                        yield {"type": "stt_final", "text": " ".join(_MOCK_WORDS[:max(words, 1)]), "latency_ms": 20}
                        utterance, partial, words = np.zeros(0, dtype=np.float32), "", 0
                    continue
                pcm = item.astype(np.float32) / 32768.0
                utterance = np.concatenate([utterance, pcm])
                if float(np.sqrt(np.mean(pcm ** 2))) > 0.01:
                    last_speech_t = now
                    if words < len(_MOCK_WORDS):
                        words += 1
            if len(utterance) > 0 and (now - last_partial_t) * 1000 >= 300:
                last_partial_t = now
                partial = " ".join(_MOCK_WORDS[:words])
                yield {"type": "stt_partial", "text": partial, "latency_ms": 20}
            if len(utterance) > 0.8 * SAMPLE_RATE and (now - last_speech_t) * 1000 > 600:
                text = " ".join(_MOCK_WORDS[:max(words, 1)])
                if text.strip():
                    yield {"type": "stt_final", "text": text.strip(), "latency_ms": 20}
                utterance, partial, words = np.zeros(0, dtype=np.float32), "", 0

    async def transcribe_once(self, pcm_f32: np.ndarray) -> str:
        return " ".join(_MOCK_WORDS[:4])
