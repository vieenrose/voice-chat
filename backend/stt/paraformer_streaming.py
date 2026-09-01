"""
Paraformer STT wrapper (FunASR) — official STT for HF speech-to-speech stack.
Model: iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch (zh+en, 16k, CUDA)
Replaces ARK custom wrapper (which had float vs bf16 conv1 bug).
API compatible with pipeline: transcribe_once(pcm_f32) -> str, transcribe_stream(queue, stop).
"""
import asyncio
import io
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator
import numpy as np
from loguru import logger
import soundfile as sf

MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

class StreamingXASR:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", mock: bool = False, chunk_ms: int = 160, **kwargs):
        self.model_id = model_id
        self.device = device
        self.mock = False
        self.backend = "paraformer-seaco"
        self.chunk_ms = chunk_ms
        self.model = None
        try:
            from funasr import AutoModel
            logger.info(f"Loading Paraformer {model_id} on {device}...")
            self.model = AutoModel(
                model=model_id,
                device="cuda:0" if device == "cuda" else "cpu",
                disable_update=True,
            )
            logger.info(f"Paraformer loaded ✓ {model_id}")
        except Exception as e:
            logger.error(f"Paraformer load failed {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Paraformer required: {e}") from e

    def _generate(self, pcm_f32_16k: np.ndarray) -> str:
        # pcm_f32_16k: float32 -1..1 @16k mono
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, pcm_f32_16k, 16000)
            tf.flush()
            path = tf.name
        try:
            res = self.model.generate(input=path, batch_size_s=300, is_final=True)
            if res and len(res) > 0:
                txt = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
                return str(txt).strip()
            return ""
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    async def transcribe_once(self, pcm_f32_16k: np.ndarray) -> str:
        return await asyncio.to_thread(self._generate, pcm_f32_16k)

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event, session_id: str | None = None) -> AsyncGenerator[dict, None]:
        """Accumulate PCM until flush, transcribe; yield stt_partial every ~1s, stt_final on flush."""
        buf: list[np.ndarray] = []
        buf_len = 0
        last_partial = asyncio.get_event_loop().time()
        transcript_cache = {"text": ""}
        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(pcm_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                item = None
            if isinstance(item, dict) and item.get("type") == "flush":
                if buf:
                    pcm = np.concatenate(buf) if len(buf) > 1 else buf[0]
                    pcm_f32 = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else np.array(pcm, dtype=np.float32)
                    text = await self.transcribe_once(pcm_f32)
                    transcript_cache["text"] = text
                    yield {"type": "stt_final", "text": text, "latency_ms": 0}
                buf = []
                buf_len = 0
                continue
            if item is None:
                continue
            # item is int16 PCM chunk
            buf.append(item)
            buf_len += len(item)
            now = asyncio.get_event_loop().time()
            # Yield partial every ~1s of accumulated audio
            if buf_len >= 16000 * 1.0 and (now - last_partial) > 1.0:
                last_partial = now
                pcm = np.concatenate(buf)
                pcm_f32 = pcm.astype(np.float32) / 32768.0
                # capped partial (first 2s) to keep it fast
                partial = await self.transcribe_once(pcm_f32[: 16000 * 2])
                if partial:
                    yield {"type": "stt_partial", "text": partial, "latency_ms": 0}
        # final flush on stop
        if buf:
            pcm = np.concatenate(buf)
            pcm_f32 = pcm.astype(np.float32) / 32768.0
            text = await self.transcribe_once(pcm_f32)
            yield {"type": "stt_final", "text": text, "latency_ms": 0}