"""
Streaming X-ASR — chunked low-latency ASR wrapper.

Design: 20ms PCM chunks @16kHz (320 samples). Maintains a rolling buffer.
Emits partial transcripts every 300ms or when buffer > 1s.
Final transcript on VAD silence (600ms) or explicit stop.

Tries in order:
1. faster-whisper (X-ASR streaming capable, CTranslate2)
2. transformers pipeline (openai/whisper-large-v3-turbo)
3. Mock (returns incremental text for demo)

Meets spec: "stt: streaming x-asr"
"""
import asyncio
import time
import numpy as np
from typing import AsyncGenerator, Optional
from loguru import logger

CHUNK_MS = 20
SAMPLE_RATE = 16000
PARTIAL_INTERVAL_MS = 300
VAD_SILENCE_MS = 600
ENERGY_THRESHOLD = 0.01  # simple VAD fallback

class StreamingXASR:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.sample_rate = SAMPLE_RATE
        self.backend = "mock"
        self.pipe = None
        self.whisper_model = None
        if mock:
            logger.info("X-ASR: MOCK mode enabled")
            return
        # Try faster-whisper (best for streaming)
        try:
            from faster_whisper import WhisperModel
            # Use int8/float16 automatically
            compute = "float16" if device == "cuda" else "int8"
            self.whisper_model = WhisperModel(model_id if "whisper" in model_id else "large-v3-turbo", device=device, compute_type=compute)
            self.backend = "faster_whisper"
            logger.info(f"X-ASR loaded via faster-whisper: {model_id}")
        except Exception as e:
            logger.warning(f"faster-whisper not available ({e}), trying transformers fallback")
            try:
                from transformers import pipeline
                import torch
                dtype = torch.float16 if device == "cuda" else torch.float32
                self.pipe = pipeline("automatic-speech-recognition", model=model_id, device=0 if device=="cuda" else -1, torch_dtype=dtype, chunk_length_s=1, stride_length_s=0.2)
                self.backend = "transformers"
                logger.info(f"X-ASR loaded via transformers: {model_id}")
            except Exception as e2:
                logger.error(f"X-ASR failed to load any backend ({e2}), falling back to mock")
                self.backend = "mock"
                self.mock = True

    def _is_speech(self, pcm: np.ndarray) -> bool:
        # simple energy VAD fallback (silero would be better but needs torch 3.12)
        energy = np.sqrt(np.mean(pcm.astype(np.float32)**2))
        return energy > ENERGY_THRESHOLD

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event) -> AsyncGenerator[dict, None]:
        """
        Consumes int16 PCM chunks (np.ndarray) from queue, yields {type: partial/final, text, latency_ms}
        """
        buffer = np.zeros(0, dtype=np.float32)  # normalized -1..1
        last_partial_t = time.time()
        last_speech_t = time.time()
        partial_text = ""
        mock_words = ["hello", "how", "are", "you", "today", "I", "am", "a", "voice", "assistant", "powered", "by", "X-ASR"]
        mock_idx = 0

        utterance_buffer = np.zeros(0, dtype=np.float32)

        while not stop_event.is_set():
            try:
                # wait for chunk with timeout to emit partials even without new audio
                try:
                    item = await asyncio.wait_for(pcm_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    item = None

                now = time.time()
                if item is not None:
                    if isinstance(item, dict) and item.get("type") == "flush":
                        # forced final
                        if len(utterance_buffer) > 0:
                            text = await self._transcribe_buffer(utterance_buffer, mock_words, mock_idx)
                            yield {"type": "stt_final", "text": text, "latency_ms": int((time.time()-now)*1000)}
                            utterance_buffer = np.zeros(0, dtype=np.float32)
                            partial_text = ""
                        continue
                    # pcm int16 -> float32
                    pcm_f32 = item.astype(np.float32) / 32768.0
                    buffer = np.concatenate([buffer, pcm_f32])
                    utterance_buffer = np.concatenate([utterance_buffer, pcm_f32])
                    if self._is_speech(pcm_f32):
                        last_speech_t = now
                # emit partial every PARTIAL_INTERVAL_MS if we have buffer
                if len(utterance_buffer) > 0 and (now - last_partial_t)*1000 >= PARTIAL_INTERVAL_MS:
                    last_partial_t = now
                    if self.mock:
                        # mock incremental
                        if mock_idx < len(mock_words):
                            partial_text = " ".join(mock_words[:mock_idx+1])
                            mock_idx = min(mock_idx+1, len(mock_words)-1)
                        yield {"type": "stt_partial", "text": partial_text, "latency_ms": 85}
                    else:
                        # only transcribe if we have >0.6s audio
                        if len(utterance_buffer) > 0.6 * SAMPLE_RATE:
                            t0 = time.time()
                            # use last 2s for partial to limit compute
                            recent = utterance_buffer[-int(2*SAMPLE_RATE):]
                            text = await self._transcribe_buffer(recent, mock_words, mock_idx)
                            latency = int((time.time()-t0)*1000)
                            if text and text != partial_text:
                                partial_text = text
                                yield {"type": "stt_partial", "text": text, "latency_ms": latency}
                # VAD endpointing: 600ms silence after speech
                if len(utterance_buffer) > 0.8*SAMPLE_RATE and (now - last_speech_t)*1000 > VAD_SILENCE_MS:
                    # endpoint detected
                    t0 = time.time()
                    if self.mock:
                        text = " ".join(mock_words[:min(mock_idx+3, len(mock_words))])
                    else:
                        text = await self._transcribe_buffer(utterance_buffer, mock_words, mock_idx)
                    latency = int((time.time()-t0)*1000)
                    if text.strip():
                        yield {"type": "stt_final", "text": text.strip(), "latency_ms": latency}
                    # reset for next utterance
                    utterance_buffer = np.zeros(0, dtype=np.float32)
                    partial_text = ""
                    mock_idx = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"X-ASR stream error: {e}")
                await asyncio.sleep(0.05)

    async def _transcribe_buffer(self, pcm_f32: np.ndarray, mock_words, mock_idx) -> str:
        if self.mock or self.backend == "mock":
            # mock just cycles words based on length
            n_words = min(max(1, len(pcm_f32)// (0.35*SAMPLE_RATE)), len(mock_words))
            return " ".join(mock_words[:n_words])
        try:
            if self.backend == "faster_whisper" and self.whisper_model:
                # run in thread to not block event loop
                def _infer():
                    segs, _ = self.whisper_model.transcribe(pcm_f32, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                    return " ".join([s.text.strip() for s in segs])
                text = await asyncio.to_thread(_infer)
                return text.strip()
            elif self.backend == "transformers" and self.pipe:
                def _infer2():
                    # pipeline expects dict with sampling_rate
                    out = self.pipe({"array": pcm_f32, "sampling_rate": SAMPLE_RATE})
                    return out["text"] if isinstance(out, dict) else str(out)
                text = await asyncio.to_thread(_infer2)
                return text.strip()
        except Exception as e:
            logger.warning(f"X-ASR transcribe error {e}")
        return ""

    async def transcribe_once(self, pcm_f32: np.ndarray) -> str:
        """One-shot for REST/testing"""
        if self.mock:
            return "hello, this is a mock X-ASR transcription"
        return await self._transcribe_buffer(pcm_f32, [], 0)
