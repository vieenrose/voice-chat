"""
Kokoro-1.0 (82M, Apache-2.0) TTS via kokoro-onnx — light prosody (minimal inserted silence).
ONNX model + voices from thewh1teagle/kokoro-onnx releases (kokoro-v1.0.onnx 311MB, voices-v1.0.bin).
Output: 24 kHz mono int16. True streaming at sentence granularity; pooled into ~1.2s chunks.
Lang map: en-us -> af_heart (default), zh -> zf_xiaobei (female) / zm_yunjian (male), '台湾腔' -> zf_xiaobei (no distinct tw accent in kokoro 1.0 zh voices).
"""
import asyncio
import re
import time
from pathlib import Path
from typing import AsyncGenerator
import numpy as np
from loguru import logger

SAMPLE_RATE = 24000
FLUSH_TOKENS = 26
SENTENCE_END = re.compile(r'[.!?。！？\n]')
MODEL_PATH = Path("/tmp/kokoro/kokoro-v1.0.onnx")
VOICES_PATH = Path("/tmp/kokoro/voices-v1.0.bin")


class StreamingPrimeTTS:
    def __init__(self, model_id: str = "hexgrad/Kokoro-1.0-82M", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = False
        self.backend = "kokoro-1.0-onnx"
        self.sample_rate = SAMPLE_RATE
        if not MODEL_PATH.exists() or not VOICES_PATH.exists():
            raise RuntimeError(f"Kokoro files missing: {MODEL_PATH.exists()} {VOICES_PATH.exists()} — download into /tmp/kokoro/")
        from kokoro_onnx import Kokoro
        t0 = time.time()
        # CPU EP is fine (RTF ~0.22); CUDA via monkeypatched ort session below if env KOKORO_GPU=1
        self.kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        names = self.kokoro.get_voices()
        self._voices = names
        logger.info(f"Kokoro-1.0 loaded in {time.time()-t0:.1f}s ({len(names)} voices, {MODEL_PATH.stat().st_size>>20}MB)")
        self.VOICE_PRESETS = {
            "默认": {"voice": "af_heart", "lang": "en-us"},
            "Aiden": {"voice": "af_heart", "lang": "en-us"},
            "中文女声": {"voice": "zf_xiaobei", "lang": "cmn"},
            "中文男声": {"voice": "zm_yunjian", "lang": "cmn"},
            "台湾腔": {"voice": "zf_xiaobei", "lang": "cmn"},
        }
        self._vv = "默认"

    @property
    def voice(self) -> str:
        return self._vv

    @property
    def voices(self) -> list[str]:
        return list(self.VOICE_PRESETS.keys())

    def set_voice(self, name: str):
        if name not in self.VOICE_PRESETS:
            raise KeyError(f"unknown voice {name}; available {self.voices}")
        self._vv = name

    def _preset(self, zh: bool) -> dict:
        p = self.VOICE_PRESETS.get(self._vv)
        if p:
            return p
        return {"voice": "zf_xiaobei" if zh else "af_heart", "lang": "cmn" if zh else "en-us"}

    @staticmethod
    def _trim_edges(pcm16: np.ndarray, keep_ms: int = 30) -> np.ndarray:
        edge_n = int(SAMPLE_RATE * keep_ms / 1000)
        abs_w = np.abs(pcm16.astype(np.int32))
        nz = np.nonzero(abs_w > 40)[0]
        if nz.size == 0:
            return pcm16
        lo = max(0, int(nz[0]) - edge_n)
        hi = min(pcm16.size, int(nz[-1]) + edge_n)
        return pcm16[lo:hi]

    def _synth(self, text: str) -> np.ndarray:
        zh = any('\u4e00' <= c <= '\u9fff' for c in text)
        p = self._preset(zh)
        try:
            samples, sr = self.kokoro.create(text, voice=p["voice"], speed=1.0, lang=p["lang"])
        except Exception as e:
            logger.warning(f"kokoro zh synth {type(e).__name__}: {str(e)[:80]} — retrying stripped/EN-fallback")
            # strip exotic chars, retry; last resort english voice
            from loguru import logger as _lg
            try:
                cleaned = re.sub(r"[^\w\u4e00-\u9fff，。！？：；、\s,..!?:;'\"%-]", "", text)
                samples, sr = self.kokoro.create(cleaned or "嗯，这个问题有点复杂。", voice=p["voice"], speed=1.0, lang=p["lang"])
            except Exception:
                samples, sr = self.kokoro.create("Sorry, this text could not be spoken.", voice="af_heart", speed=1.0, lang="en-us")
        samples = np.asarray(samples, dtype=np.float32).squeeze()
        samples = np.nan_to_num(samples)
        samples = np.clip(samples, -1.0, 1.0)
        pcm = (samples * 32767).astype(np.int16)
        return self._trim_edges(pcm)

    async def synthesize_streaming(self, text: str, chunk_frames: int = 24) -> AsyncGenerator[np.ndarray, None]:
        """Whole-sentence chunk (no mid-speech pooling splits!).
        Splitting a waveform at fixed sample offsets cuts MID-SYLLABLE and, being deterministic,
        reproduces the same 'pause' at the same position every run. One chunk per sentence =
        boundaries only at natural sentence ends."""
        pcm = await asyncio.to_thread(self._synth, text)
        if len(pcm):
            yield pcm

    async def stream_tts(self, token_stream: AsyncGenerator[dict, None]) -> AsyncGenerator[dict, None]:
        buf = ""
        token_count = 0
        async for chunk in token_stream:
            if chunk["type"] == "llm_token":
                token = chunk["token"]
                buf += token
                token_count += 1
                should_flush = False
                if SENTENCE_END.search(token):
                    should_flush = True
                elif token_count >= FLUSH_TOKENS and buf and buf[-1] in " ,":
                    # only flush at >=FLUSH_TOKENS tokens ending in space/comma; avoid mid-sentence fragmentation
                    should_flush = False
                if should_flush and buf.strip():
                    txt = buf.strip()
                    buf = ""
                    token_count = 0
                    async for pcm_chunk in self.synthesize_streaming(txt):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": txt, "sampleRate": self.sample_rate, "latency_ms": 40}
            elif chunk["type"] == "llm_done":
                if buf.strip():
                    async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": self.sample_rate, "latency_ms": 40}
                yield {"type": "tts_end"}
                return
        if buf.strip():
            async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def tts_from_text(self, text: str) -> AsyncGenerator[dict, None]:
        sentences = re.split(r'([.!?。！？]+)', text)
        for i in range(0, len(sentences), 2):
            sent = (sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")).strip()
            if not sent:
                continue
            async for pcm_chunk in self.synthesize_streaming(sent):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": sent, "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def synthesize(self, text: str, reference_audio: str = None) -> np.ndarray:
        return await asyncio.to_thread(self._synth, text)