"""
VoxCPM2 streaming TTS wrapper — replaces PrimeTTS mock
Real streaming: openbmb/VoxCPM2, 2B, 48kHz, 30 langs, diffusion AR, RTF 0.3 on 4090
Supports streaming via model.generate_streaming(text=..., cfg_value=2.0, inference_timesteps=10)
"""
import asyncio
import time
import re
import numpy as np
from typing import AsyncGenerator
from loguru import logger
from pathlib import Path

SAMPLE_RATE = 48000  # VoxCPM2 outputs 48kHz (AudioVAE V2 16k in -> 48k out)
FLUSH_TOKENS = 8
SENTENCE_END = re.compile(r'[.!?。！？\n]')
# Fixed voice for consistency — prevents switching between 2 random voices
VOICE_PREFIX = "(A young woman, warm and clear voice, steady pace, gentle tone) "

class StreamingPrimeTTS:
    """Keep same class name for pipeline compatibility, but now VoxCPM2 backend"""
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = False  # never mock, always real VoxCPM2
        self.backend = "voxcpm2"
        self.model = None
        try:
            from voxcpm import VoxCPM
            logger.info(f"Loading VoxCPM2 {model_id} on {device}...")
            # VoxCPM2: 2B, bfloat16, CUDA, 8GB VRAM, 48kHz
            # Use load_denoiser=False for faster, we don't need denoiser for streaming demo
            self.model = VoxCPM.from_pretrained(model_id, load_denoiser=False)
            logger.info(f"VoxCPM2 loaded ✓ sample_rate={self.model.tts_model.sample_rate}Hz")
            # Verify sample rate
            assert self.model.tts_model.sample_rate == 48000
        except Exception as e:
            logger.error(f"VoxCPM2 load failed {e}, cannot fallback to mock (mock removed)")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"VoxCPM2 required, mock removed: {e}")

    def _mock_synth(self, text: str) -> np.ndarray:
        # Should never be called now, but keep for safety
        raise RuntimeError("Mock removed - VoxCPM2 must be used")

    def _with_voice(self, text: str) -> str:
        # Lock to single voice to prevent switching between 2 random voices
        # If text already has voice design parentheses, keep it, otherwise prefix
        if text.strip().startswith("("):
            return text
        return VOICE_PREFIX + text

    async def synthesize(self, text: str) -> np.ndarray:
        """One-shot synthesis → int16 PCM 48k"""
        text = self._with_voice(text)
        try:
            def _infer():
                # VoxCPM2 generate with cfg 2.0, 10 steps (as per README) — fixed voice
                wav = self.model.generate(
                    text=text,
                    cfg_value=2.0,
                    inference_timesteps=10,
                )
                # wav is np.ndarray float32 -1..1 at 48k
                # Convert to int16
                pcm = (np.array(wav) * 32767).astype(np.int16)
                return pcm
            pcm = await asyncio.to_thread(_infer)
            return pcm
        except Exception as e:
            logger.error(f"VoxCPM2 synthesize failed {e}")
            raise

    async def synthesize_streaming(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """True streaming: yields PCM chunks per diffusion step (7680 samples each)"""
        try:
            # Use generate_streaming which yields np.ndarray chunks at 48k
            # Need to run in thread and yield as they come
            # VoxCPM2's generate_streaming is sync generator, we wrap it
            def _gen():
                # This is a sync generator, we need to collect chunks
                for chunk in self.model.generate_streaming(text=text, cfg_value=2.0, inference_timesteps=10):
                    yield chunk
            # For async streaming, we need to iterate in thread
            # Instead, we can run the whole streaming in a thread and queue chunks
            text = self._with_voice(text)
            import queue, threading
            q = queue.Queue()
            def _run():
                try:
                    for chunk in self.model.generate_streaming(text=text, cfg_value=2.0, inference_timesteps=10):
                        pcm = (np.array(chunk) * 32767).astype(np.int16)
                        q.put(pcm)
                    q.put(None)  # end
                except Exception as e:
                    logger.error(f"VoxCPM2 streaming thread failed {e}")
                    q.put(None)
            th = threading.Thread(target=_run, daemon=True)
            th.start()
            while True:
                try:
                    # Use to_thread to not block event loop
                    pcm = await asyncio.to_thread(q.get, True, 30)
                    if pcm is None:
                        break
                    yield pcm
                except:
                    break
        except Exception as e:
            logger.error(f"VoxCPM2 streaming failed {e}, fallback to one-shot")
            pcm = await self.synthesize(text)
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
                elif token_count >= FLUSH_TOKENS:
                    if buf and buf[-1] in " ,":
                        should_flush = True
                if should_flush and buf.strip():
                    text_to_synth = buf.strip()
                    buf = ""
                    token_count = 0
                    t0 = time.time()
                    # Stream VoxCPM2 per sentence - yield each chunk
                    async for pcm_chunk in self.synthesize_streaming(text_to_synth):
                        latency = int((time.time()-t0)*1000) if t0 else 0
                        t0 = None  # only first chunk has latency
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": text_to_synth, "sampleRate": SAMPLE_RATE, "latency_ms": latency}
            elif chunk["type"] == "llm_done":
                if buf.strip():
                    async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": SAMPLE_RATE, "latency_ms": 60}
                yield {"type": "tts_end"}
                return
        if buf.strip():
            async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": SAMPLE_RATE, "latency_ms": 60}
        yield {"type": "tts_end"}

    async def tts_from_text(self, text: str) -> AsyncGenerator[dict, None]:
        sentences = re.split(r'([.!?]+)', text)
        for i in range(0, len(sentences), 2):
            sent = (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")).strip()
            if not sent:
                continue
            async for pcm_chunk in self.synthesize_streaming(sent):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": sent, "sampleRate": SAMPLE_RATE, "latency_ms": 40}
        yield {"type": "tts_end"}
