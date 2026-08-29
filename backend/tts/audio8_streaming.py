"""
Audio8-TTS-0.1B-ONNX-INT8 — CPU streaming runtime via local HTTP service (port 8024).
NDJSON /api/tts/stream: start(44100, s16le) -> audio_chunk(seq, frame_count, pcm_b64) -> complete.
0.1B INT8 Falcon-H1 hybrid: CPU-only, RTF ~0.4, 44.1kHz, bilingual zh/en, voice-clone presets.
"""
import asyncio
import base64
import json
import re
import time
from typing import AsyncGenerator
import numpy as np
from loguru import logger

SAMPLE_RATE = 44100
FLUSH_TOKENS = 26
SENTENCE_END = re.compile(r'[.!?。！？\n]')
SERVER = "http://127.0.0.1:8024"


class StreamingPrimeTTS:
    def __init__(self, model_id: str = "Audio8/audio8-TTS-0.1B-ONNX-INT8", device: str = "cpu", mock: bool = False):
        self.model_id = model_id
        self.device = "cpu"
        self.mock = False
        self.backend = "audio8-tts-0.1b-int8"
        self.sample_rate = SAMPLE_RATE
        import httpx
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(f"{SERVER}/api/health")
                r.raise_for_status()
                h = r.json()
            logger.info(f"Audio8-TTS-0.1B server ready: {h.get('model')} 44.1k CPU")
        except Exception as e:
            raise RuntimeError(f"Audio8-TTS server unreachable on {SERVER}: {e} — start it first "
                               f"(onnx_runtime_0_1b_int8: ARKTTS_MODEL_DIR=model python3 -m uvicorn arktts_runtime.service:app --port 8024)")
        self.VOICE_PRESETS = {
            "默认": "default",
            "中文女声": "default",
            "中文男声": "default",
            "台湾腔": "default",
            "Aiden": "default",
        }
        self._vv = "默认"

    @property
    def voice(self) -> str:
        return self._vv

    def voices(self) -> list[str]:
        return list(self.VOICE_PRESETS.keys())

    def set_voice(self, name: str):
        if name not in self.VOICE_PRESETS:
            raise KeyError(f"unknown voice {name}; available {self.voices()}")
        self._vv = name

    async def synthesize_streaming(self, text: str, chunk_frames: int = 24) -> AsyncGenerator[np.ndarray, None]:
        """True streaming: read NDJSON chunks as the server decodes them (pooled to ~0.5s buffers)."""
        import httpx
        payload = {"text": text, "voice_name": self.VOICE_PRESETS[self._vv], "max_new_tokens": 256}
        _buf = np.zeros(0, dtype=np.int16)
        _first = time.time()
        async with httpx.AsyncClient(timeout=180.0) as c:
            async with c.stream("POST", f"{SERVER}/api/tts/stream", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()[:200]
                    logger.error(f"Audio8 stream {resp.status_code}: {body}")
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    et = ev.get("event")
                    if et == "start":
                        logger.info(f"Audio8-TTS stream start @{self.sample_rate}Hz")
                    elif et == "audio_chunk":
                        pcm = np.frombuffer(base64.b64decode(ev["pcm_b64"]), dtype=np.int16)
                        if len(pcm) == 0:
                            continue
                        _buf = np.concatenate([_buf, pcm])
                        if len(_buf) / self.sample_rate >= 0.5:
                            yield _buf
                            _buf = np.zeros(0, dtype=np.int16)
                    elif et == "complete":
                        break
                    elif et == "cancelled":
                        break
        if len(_buf):
            yield _buf
        _buf = np.zeros(0, dtype=np.int16)

    async def stream_tts(self, token_stream: AsyncGenerator[dict, None]) -> AsyncGenerator[dict, None]:
        buf = ""
        token_count = 0
        async for chunk in token_stream:
            if chunk["type"] == "llm_token":
                token = chunk["token"]
                buf += token
                token_count += 1
                if SENTENCE_END.search(token) and buf.strip():
                    txt = buf.strip(); buf = ""; token_count = 0
                    async for pcm_chunk in self.synthesize_streaming(txt):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": txt,
                               "sampleRate": self.sample_rate, "latency_ms": 40}
            elif chunk["type"] == "llm_done":
                if buf.strip():
                    async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(),
                               "sampleRate": self.sample_rate, "latency_ms": 40}
                yield {"type": "tts_end"}
                return
        if buf.strip():
            async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(),
                       "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def tts_from_text(self, text: str) -> AsyncGenerator[dict, None]:
        async for pcm_chunk in self.synthesize_streaming(text):
            yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": text,
                   "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def synthesize(self, text: str, reference_audio: str = None) -> np.ndarray:
        parts = []
        async for c in self.synthesize_streaming(text):
            parts.append(c)
        return np.concatenate(parts) if parts else np.zeros(int(SAMPLE_RATE*0.4), dtype=np.int16)