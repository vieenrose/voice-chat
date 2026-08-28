"""
MOSS-TTS-Nano-100M — TRUE streaming via onnxruntime (CUDA EP), Apache-2.0, 100M, 20 langs, 48k.
Uses OpenMOSS browser_onnx export (moss_tts graphs + audio tokenizer) through OnnxTtsRuntime.
WHY ONNX: torch path produces Non-finite text logits under transformers 5.15 (NaN, all dtypes/devices);
onnxruntime has no such issue and CUDAExecutionProvider gives GPU speed.
Voices via built-in presets (voice clone): Yuewen(台湾腔/zh_4), Ava(en_2), Junhao(zh_1), ...
"""
import asyncio
import time
import re
import os, sys
from pathlib import Path
from typing import AsyncGenerator, Iterator
import numpy as np
from loguru import logger

SAMPLE_RATE = 48000
FLUSH_TOKENS = 8
SENTENCE_END = re.compile(r'[.!?。！？\n]')
_MOSS_REPO = "/tmp/MOSS-TTS-Nano"
if _MOSS_REPO not in sys.path:
    sys.path.insert(0, _MOSS_REPO)

class StreamingPrimeTTS:
    def __init__(self, model_id: str = "OpenMOSS-Team/MOSS-TTS-Nano-100M", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = False
        self.backend = "moss-nano-100m-onnx"
        self.runtime = None
        self.sample_rate = SAMPLE_RATE
        self.voices = ["Junhao", "Yuewen", "Ava", "Xiaoyu", "Lingyu"]
        self._vv = "Junhao"
        try:
            from onnx_tts_runtime import OnnxTtsRuntime
            ep = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            logger.info(f"MOSS-ONNX: loading (EP={ep})...")
            self.runtime = OnnxTtsRuntime(
                model_dir=None,
                thread_count=8,
                sample_mode="topk",
                execution_provider=ep,
            )
            self.sample_rate = SAMPLE_RATE
            _v = self.runtime.list_builtin_voices()
            self.voices = [v.get("voice") for v in _v] or self.voices
            logger.info(f"MOSS-ONNX TRUE-STREAMING loaded ✓ (EP={ep}) sr=48000 voices={len(self.voices)}")
        except Exception as e:
            logger.error(f"MOSS-ONNX load failed {e}")
            import traceback; traceback.print_exc()
            raise RuntimeError(f"MOSS-Nano-ONNX required: {e}")

    @property
    def voice(self): return self._vv
    def set_voice(self, name: str):
        key = {"台湾腔": "Yuewen", "中文女声": "Yuewen", "中文男声": "Junhao"}.get(name, name)
        if key in self.voices:
            self._vv = key
            return
        raise KeyError(f"unknown voice {name}; available {self.voices}")

    def _synth_one(self, text: str, max_new_frames: int | None = None) -> np.ndarray:
        """Synthesize one span -> mono float32 (progressive per call)."""
        zh = any('\u4e00' <= c <= '\u9fff' for c in text)
        voice = self._vv if self._vv in self.voices else ("Yuewen" if zh else "Ava")
        mf = max_new_frames or (max(40, min(200, int(len(text)*1.8))) if zh else max(40, min(200, len(text.split())*16)))
        res = self.runtime.synthesize(text=text, voice=voice, streaming=True,
                                      max_new_frames=mf, enable_wetext=False,
                                      enable_normalize_tts_text=False,
                                      voice_clone_max_text_tokens=512)  # one chunk per span -> no inter-chunk pauses
        rows = res.get("chunk_results", [])
        waves = [np.asarray(r["waveform"], dtype=np.float32) for r in rows if "waveform" in r]
        if not waves:
            return np.zeros(0, dtype=np.float32)
        w = np.concatenate(waves)
        if w.ndim == 2:
            w = w.mean(axis=1)
        # trim leading/trailing silence (MOSS appends tail silences per span -> "lots of pauses")
        w = self._trim_silence(w, keep_ms=40)
        pk = float(np.max(np.abs(w))) or 1.0
        if pk > 0:
            w = w / pk * 0.95
        return w

    @staticmethod
    def _trim_silence(w: np.ndarray, keep_ms: int = 40, thresh: float = 1e-3) -> np.ndarray:
        """Trim quiet leading/trailing samples, keeping keep_ms of padding."""
        if w.size == 0:
            return w
        keep = int(SAMPLE_RATE * keep_ms / 1000.0)
        abs_w = np.abs(w)
        nz = np.nonzero(abs_w > thresh)[0]
        if nz.size == 0:
            return w
        lo = max(0, int(nz[0]) - keep)
        hi = min(w.size, int(nz[-1]) + keep)
        return w[lo:hi]

    async def synthesize_streaming(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """TRUE streaming at sentence granularity (progressive; first span plays while rest generate)."""
        sents = [s for s in re.split(r'([.!?。！？]+)', text) if s and s.strip()]
        if not sents:
            sents = [text]
        # merge to build spans: keep single sentences; longer text -> each sentence one span
        spans = []
        cur = ""
        for s in sents:
            if s.strip() in ".!?。！？":
                cur += s
            else:
                if cur: spans.append(cur.strip())
                cur = s
        if cur.strip(): spans.append(cur.strip())
        if not spans: spans = [text.strip()]
        chunks = []
        loop = asyncio.get_running_loop()
        for i, span in enumerate(spans[:6]):
            t0 = time.time()
            w = await asyncio.to_thread(self._synth_one, span)
            if w.size == 0:
                continue
            pcm = np.clip(w * 32767, -32768, 32767).astype(np.int16)
            if i == 0:
                logger.info(f"MOSS-ONNX streaming TTFA {int((time.time()-t0)*1000)}ms (span 1/{(len(spans[:6]))})")
            chunks.append(pcm)
            yield pcm
            await asyncio.sleep(0.01)  # let consumer play first span

    async def synthesize(self, text: str, reference_audio: str = None) -> np.ndarray:
        parts = []
        async for c in self.synthesize_streaming(text):
            parts.append(c)
        if not parts:
            return np.zeros(int(SAMPLE_RATE*0.4), dtype=np.int16)
        return np.concatenate(parts)

    async def stream_tts(self, token_stream):
        buf=""; cnt=0
        async for ev in token_stream:
            if ev["type"]=="llm_token":
                buf+=ev["token"]; cnt+=1
                fl=False
                if SENTENCE_END.search(ev["token"]): fl=True
                elif cnt>=FLUSH_TOKENS and buf and buf[-1] in " ,": fl=True
                if fl and buf.strip():
                    txt=buf.strip(); buf=""; cnt=0
                    async for c in self.synthesize_streaming(txt):
                        yield {"type":"tts_chunk","pcm":c,"text":txt,"sampleRate":self.sample_rate,"latency_ms":20}
            elif ev["type"]=="llm_done":
                if buf.strip():
                    async for c in self.synthesize_streaming(buf.strip()):
                        yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":20}
                yield {"type":"tts_end"}; return
        if buf.strip():
            async for c in self.synthesize_streaming(buf.strip()):
                yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":20}
        yield {"type":"tts_end"}

    async def tts_from_text(self, text):
        async for c in self.synthesize_streaming(text):
            yield {"type":"tts_chunk","pcm":c,"text":text,"sampleRate":self.sample_rate,"latency_ms":20}
        yield {"type":"tts_end"}