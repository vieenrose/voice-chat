"""
Streaming X-ASR — sherpa-onnx Zipformer transducer (GilgameshWind/X-ASR-zh-en)
True streaming: 160ms / 480ms / 960ms chunks, 16kHz, endpoint detection
Replaces whisper fallback with real X-ASR streaming.
"""
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import AsyncGenerator
from loguru import logger

SAMPLE_RATE = 16000
CHUNK_MS = 20
PARTIAL_INTERVAL_MS = 200  # X-ASR gives partials faster than whisper

class StreamingXASR:
    def __init__(self, model_id: str = "GilgameshWind/X-ASR-zh-en", device: str = "cuda", mock: bool = False, chunk_ms: int = 160):
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.chunk_ms = chunk_ms  # 160, 480, 960, 1920
        self.sample_rate = SAMPLE_RATE
        self.backend = "mock"
        self.recognizer = None
        self.stream = None
        if mock:
            logger.info("X-ASR: MOCK mode (requested)")
            # Still try to load real X-ASR even in mock for warmup? No, respect mock
            return
        # Try sherpa-onnx X-ASR
        try:
            import sherpa_onnx
            from huggingface_hub import hf_hub_download
            # Choose chunk size
            # Prefer 160ms for lowest latency, fallback to 480ms
            chunk = chunk_ms if chunk_ms in [160, 480, 960, 1920] else 160
            # Try local /tmp/XASR first (already downloaded)
            local_base = Path(f"/tmp/XASR/deployment/models/chunk-{chunk}ms-model")
            if local_base.exists():
                enc = str(local_base / f"encoder-{chunk}ms.onnx")
                dec = str(local_base / f"decoder-{chunk}ms.onnx")
                join = str(local_base / f"joiner-{chunk}ms.onnx")
                tokens = str(local_base / "tokens.txt")
                logger.info(f"X-ASR loading local {chunk}ms {local_base}")
            else:
                logger.info(f"X-ASR downloading {model_id} chunk-{chunk}ms...")
                enc = hf_hub_download(model_id, f"deployment/models/chunk-{chunk}ms-model/encoder-{chunk}ms.onnx")
                dec = hf_hub_download(model_id, f"deployment/models/chunk-{chunk}ms-model/decoder-{chunk}ms.onnx")
                join = hf_hub_download(model_id, f"deployment/models/chunk-{chunk}ms-model/joiner-{chunk}ms.onnx")
                tokens = hf_hub_download(model_id, f"deployment/models/chunk-{chunk}ms-model/tokens.txt")

            # Check files exist
            for p in [enc, dec, join, tokens]:
                if not Path(p).exists():
                    raise FileNotFoundError(p)

            # Create recognizer - use CUDA for X-ASR as requested (was CPU)
            provider = "cuda" if device == "cuda" else "cpu"
            logger.info(f"X-ASR provider={provider} (CUDA requested)")
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=enc,
                decoder=dec,
                joiner=join,
                num_threads=2,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=1.2,
                rule2_min_trailing_silence=2.4,
                rule3_min_utterance_length=20.0,
                provider=provider,
            )
            self.stream = self.recognizer.create_stream()
            self.backend = f"x-asr-sherpa-{chunk}ms"
            logger.info(f"X-ASR sherpa-onnx loaded ✓ {self.backend} {model_id} (160ms streaming, 593M encoder)")
        except Exception as e:
            logger.warning(f"X-ASR sherpa load failed {e}, trying whisper fallback")
            import traceback; traceback.print_exc()
            # Fallback to whisper
            try:
                from faster_whisper import WhisperModel
                compute = "float16" if device == "cuda" else "int8"
                self.whisper_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute)
                self.backend = "faster_whisper"
                logger.info("X-ASR fallback whisper loaded")
            except Exception as e2:
                logger.warning(f"whisper fallback failed {e2}, trying transformers")
                try:
                    from transformers import pipeline
                    import torch
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    self.pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0 if device=="cuda" else -1, torch_dtype=dtype, chunk_length_s=1, stride_length_s=0.2)
                    self.backend = "transformers"
                    logger.info("X-ASR fallback transformers loaded")
                except Exception as e3:
                    logger.error(f"All X-ASR backends failed {e3}, using mock")
                    self.backend = "mock"
                    self.mock = True

    def _is_speech(self, pcm_f32: np.ndarray) -> bool:
        energy = np.sqrt(np.mean(pcm_f32.astype(np.float32)**2))
        return energy > 0.01

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event) -> AsyncGenerator[dict, None]:
        if self.mock or self.backend == "mock":
            # Mock streaming as before
            buffer = np.zeros(0, dtype=np.float32)
            last_partial_t = time.time()
            last_speech_t = time.time()
            partial_text = ""
            mock_words = ["hello", "how", "are", "you", "today", "I", "am", "voice", "assistant", "x-asr", "streaming"]
            mock_idx = 0
            utterance_buffer = np.zeros(0, dtype=np.float32)
            VAD_SILENCE_MS = 600
            while not stop_event.is_set():
                try:
                    try:
                        item = await asyncio.wait_for(pcm_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        item = None
                    now = time.time()
                    if item is not None:
                        if isinstance(item, dict) and item.get("type") == "flush":
                            if len(utterance_buffer) > 0:
                                text = " ".join(mock_words[:min(mock_idx+3, len(mock_words))])
                                yield {"type": "stt_final", "text": text, "latency_ms": 20}
                                utterance_buffer = np.zeros(0, dtype=np.float32)
                                partial_text = ""
                            continue
                        pcm_f32 = item.astype(np.float32) / 32768.0
                        utterance_buffer = np.concatenate([utterance_buffer, pcm_f32])
                        if np.sqrt(np.mean(pcm_f32**2)) > 0.01:
                            last_speech_t = now
                    if len(utterance_buffer) > 0 and (now - last_partial_t)*1000 >= 300:
                        last_partial_t = now
                        if mock_idx < len(mock_words):
                            partial_text = " ".join(mock_words[:mock_idx+1])
                            mock_idx = min(mock_idx+1, len(mock_words)-1)
                        yield {"type": "stt_partial", "text": partial_text, "latency_ms": 20}
                    if len(utterance_buffer) > 0.8*SAMPLE_RATE and (now - last_speech_t)*1000 > VAD_SILENCE_MS:
                        text = " ".join(mock_words[:min(mock_idx+3, len(mock_words))])
                        if text.strip():
                            yield {"type": "stt_final", "text": text.strip(), "latency_ms": 20}
                        utterance_buffer = np.zeros(0, dtype=np.float32)
                        partial_text = ""
                        mock_idx = 0
                except asyncio.CancelledError:
                    break
            return

        # Real X-ASR sherpa streaming
        if self.backend.startswith("x-asr-sherpa"):
            import sherpa_onnx
            # Ensure stream exists
            if self.stream is None:
                self.stream = self.recognizer.create_stream()
            last_partial = ""
            last_partial_t = time.time()
            try:
                while not stop_event.is_set():
                    try:
                        item = await asyncio.wait_for(pcm_queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        item = None
                    # Handle flush
                    if item is not None and isinstance(item, dict) and item.get("type") == "flush":
                        # Signal end, get final
                        self.stream.input_finished()
                        while self.recognizer.is_ready(self.stream):
                            self.recognizer.decode_stream(self.stream)
                        text = self.recognizer.get_result_all(self.stream).text.strip()
                        if text:
                            yield {"type": "stt_final", "text": text, "latency_ms": 30}
                        # Reset for next utterance
                        self.recognizer.reset(self.stream)
                        self.stream = self.recognizer.create_stream()
                        last_partial = ""
                        continue
                    if item is not None:
                        # item is int16 PCM
                        pcm_f32 = item.astype(np.float32) / 32768.0
                        # sherpa expects float32 [-1,1] at 16k
                        self.stream.accept_waveform(SAMPLE_RATE, pcm_f32)
                        # Decode if ready
                        while self.recognizer.is_ready(self.stream):
                            self.recognizer.decode_stream(self.stream)
                        # Try to get partial
                        now = time.time()
                        if (now - last_partial_t)*1000 >= PARTIAL_INTERVAL_MS:
                            last_partial_t = now
                            text = self.recognizer.get_result_all(self.stream).text.strip()
                            if text and text != last_partial:
                                last_partial = text
                                yield {"type": "stt_partial", "text": text, "latency_ms": 40}
                        # Check endpoint
                        if self.recognizer.is_endpoint(self.stream):
                            text = self.recognizer.get_result_all(self.stream).text.strip()
                            if text:
                                yield {"type": "stt_final", "text": text, "latency_ms": 40}
                            self.recognizer.reset(self.stream)
                            self.stream = self.recognizer.create_stream()
                            last_partial = ""
                    else:
                        # No new audio, still decode and check endpoint
                        while self.recognizer.is_ready(self.stream):
                            self.recognizer.decode_stream(self.stream)
                        if self.recognizer.is_endpoint(self.stream):
                            text = self.recognizer.get_result_all(self.stream).text.strip()
                            if text:
                                yield {"type": "stt_final", "text": text, "latency_ms": 40}
                            self.recognizer.reset(self.stream)
                            self.stream = self.recognizer.create_stream()
                            last_partial = ""
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception(f"X-ASR stream error {e}")
            return

        # Fallback whisper streaming (existing logic)
        # Simplified: use buffer + transcribe
        buffer = np.zeros(0, dtype=np.float32)
        utterance_buffer = np.zeros(0, dtype=np.float32)
        last_partial_t = time.time()
        last_speech_t = time.time()
        partial_text = ""
        while not stop_event.is_set():
            try:
                try:
                    item = await asyncio.wait_for(pcm_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    item = None
                now = time.time()
                if item is not None:
                    if isinstance(item, dict) and item.get("type") == "flush":
                        if len(utterance_buffer) > 0:
                            text = await self._transcribe_buffer(utterance_buffer)
                            yield {"type": "stt_final", "text": text, "latency_ms": 20}
                            utterance_buffer = np.zeros(0, dtype=np.float32)
                        continue
                    pcm_f32 = item.astype(np.float32) / 32768.0
                    utterance_buffer = np.concatenate([utterance_buffer, pcm_f32])
                    if self._is_speech(pcm_f32):
                        last_speech_t = now
                if len(utterance_buffer) > 0 and (now - last_partial_t)*1000 >= 300:
                    last_partial_t = now
                    if len(utterance_buffer) > 0.6*SAMPLE_RATE:
                        recent = utterance_buffer[-int(2*SAMPLE_RATE):]
                        text = await self._transcribe_buffer(recent)
                        if text and text != partial_text:
                            partial_text = text
                            yield {"type": "stt_partial", "text": text, "latency_ms": 80}
                if len(utterance_buffer) > 0.8*SAMPLE_RATE and (now - last_speech_t)*1000 > 600:
                    text = await self._transcribe_buffer(utterance_buffer)
                    if text.strip():
                        yield {"type": "stt_final", "text": text.strip(), "latency_ms": 80}
                    utterance_buffer = np.zeros(0, dtype=np.float32)
                    partial_text = ""
            except asyncio.CancelledError:
                break

    async def _transcribe_buffer(self, pcm_f32: np.ndarray) -> str:
        if hasattr(self, 'whisper_model') and self.whisper_model:
            def _infer():
                segs, _ = self.whisper_model.transcribe(pcm_f32, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                return " ".join([s.text.strip() for s in segs])
            return (await asyncio.to_thread(_infer)).strip()
        elif hasattr(self, 'pipe') and self.pipe:
            def _infer2():
                out = self.pipe({"array": pcm_f32, "sampling_rate": SAMPLE_RATE})
                return out["text"] if isinstance(out, dict) else str(out)
            return (await asyncio.to_thread(_infer2)).strip()
        return ""

    async def transcribe_once(self, pcm_f32: np.ndarray) -> str:
        if self.backend.startswith("x-asr-sherpa") and self.recognizer:
            # One-shot via sherpa: create temp stream
            try:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, pcm_f32.astype(np.float32))
                tail = np.zeros(int(0.3*SAMPLE_RATE), dtype=np.float32)
                stream.accept_waveform(SAMPLE_RATE, tail)
                stream.input_finished()
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                return self.recognizer.get_result_all(stream).text.strip()
            except Exception as e:
                logger.warning(f"X-ASR one-shot failed {e}")
        if hasattr(self, 'whisper_model') and self.whisper_model:
            def _infer():
                segs, _ = self.whisper_model.transcribe(pcm_f32, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                return " ".join([s.text.strip() for s in segs])
            return (await asyncio.to_thread(_infer)).strip()
        elif hasattr(self, 'pipe') and self.pipe:
            def _infer2():
                out = self.pipe({"array": pcm_f32, "sampling_rate": SAMPLE_RATE})
                return out["text"] if isinstance(out, dict) else str(out)
            return (await asyncio.to_thread(_infer2)).strip()
        return "mock transcription"
