"""
Last-resort mock TTS adapter — terminal rung of pipeline/speech_to_speech.py's TTS
import ladder (mirrors stt/mock_streaming.py for the same reason: without it, the
documented `--mock` mode could not start on a machine lacking the real TTS wheels).

Yields a quiet 220 Hz tone shaped into ~1.2s chunks so the transport, the frontend's
pre-roll/jitter buffer, barge-in and the turn_id protocol are all exercisable with no
weights and no GPU. It is NOT intelligible speech and /health reports it as "mock".
"""
import asyncio
import re
import time
from typing import AsyncGenerator
import numpy as np
from loguru import logger

SAMPLE_RATE = 24000
SENTENCE_END = re.compile(r'[.!?。！？\n]')


class StreamingPrimeTTS:
    def __init__(self, model_id: str = "mock", device: str = "cpu", mock: bool = True, **kwargs):
        self.model_id = model_id
        self.device = device
        self.mock = True
        self.backend = "mock"
        self.sample_rate = SAMPLE_RATE
        self.VOICE_PRESETS = {"mock": {"type": "speaker", "name": "mock"}}
        self._vv = "mock"
        self.speaker = "mock"
        logger.warning("TTS: MOCK adapter in use — no TTS engine available. "
                       "Audio is a tone, not speech; install faster-qwen3-tts for the real stack.")

    @property
    def voices(self):
        return list(self.VOICE_PRESETS.keys())

    def set_voice(self, name: str):
        if name not in self.VOICE_PRESETS:
            raise KeyError(f"unknown voice {name}; available {self.voices}")
        self._vv = name

    def _preset(self, voice: str | None = None):
        if voice and voice not in self.VOICE_PRESETS:
            raise KeyError(f"unknown voice {voice}; available {self.voices}")
        return self.VOICE_PRESETS[self._vv], self.speaker

    async def synthesize_streaming(self, text: str, chunk_frames: int = 24, voice: str = None) -> AsyncGenerator[np.ndarray, None]:
        self._preset(voice)
        # ~0.25s of tone per ~6 characters of text, in 1.2s-capable chunks: enough
        # audio duration to exercise client-side scheduling without being instant.
        total_s = min(6.0, max(0.4, 0.05 + 0.04 * len(text)))
        n = int(total_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        tone = (0.15 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        step = int(1.2 * SAMPLE_RATE)
        for i in range(0, n, step):
            await asyncio.sleep(0.02)      # let the consumer cancel (barge-in)
            yield tone[i:i + step]

    async def synthesize(self, text: str, voice: str = None) -> np.ndarray:
        parts = [c async for c in self.synthesize_streaming(text, voice=voice)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    async def stream_tts(self, token_stream) -> AsyncGenerator[dict, None]:
        buf = ""
        async for chunk in token_stream:
            if chunk["type"] == "llm_token":
                buf += chunk["token"]
                if SENTENCE_END.search(chunk["token"]) and buf.strip():
                    txt, buf = buf.strip(), ""
                    async for pcm in self.synthesize_streaming(txt):
                        yield {"type": "tts_chunk", "pcm": pcm, "text": txt,
                               "sampleRate": self.sample_rate, "latency_ms": 20}
            elif chunk["type"] == "llm_done":
                break
        if buf.strip():
            async for pcm in self.synthesize_streaming(buf.strip()):
                yield {"type": "tts_chunk", "pcm": pcm, "text": buf.strip(),
                       "sampleRate": self.sample_rate, "latency_ms": 20}
        yield {"type": "tts_end"}

    async def tts_from_text(self, text: str) -> AsyncGenerator[dict, None]:
        async for pcm in self.synthesize_streaming(text):
            yield {"type": "tts_chunk", "pcm": pcm, "text": text,
                   "sampleRate": self.sample_rate, "latency_ms": 20}
        yield {"type": "tts_end"}
