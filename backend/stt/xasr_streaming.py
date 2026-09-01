"""
Streaming X-ASR — sherpa-onnx Zipformer transducer (GilgameshWind/X-ASR-zh-en)
True streaming: 160ms / 480ms / 960ms chunks, 16kHz, endpoint detection
Replaces whisper fallback with real X-ASR streaming.
"""
import asyncio
import os
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
            # Try local dir first (already downloaded). STT_MODEL_DIR (set by
            # Dockerfile/docker-compose to /models/stt) overrides the bare-metal dev
            # default of /tmp/XASR/deployment/models — without this, the Docker image
            # never found its mounted model volume and always fell through to whisper/mock.
            stt_base_dir = os.getenv("STT_MODEL_DIR", "/tmp/XASR/deployment/models")
            local_base = Path(stt_base_dir) / f"chunk-{chunk}ms-model"
            if local_base.exists():
                # Prefer int8 quantized models if available (153M vs 587M total, ~3.9× smaller)
                enc_int8 = local_base / f"encoder-{chunk}ms.int8.onnx"
                dec_int8 = local_base / f"decoder-{chunk}ms.int8.onnx"
                join_int8 = local_base / f"joiner-{chunk}ms.int8.onnx"
                enc = str(enc_int8 if enc_int8.exists() else local_base / f"encoder-{chunk}ms.onnx")
                dec = str(dec_int8 if dec_int8.exists() else local_base / f"decoder-{chunk}ms.onnx")
                join = str(join_int8 if join_int8.exists() else local_base / f"joiner-{chunk}ms.onnx")
                tokens = str(local_base / "tokens.txt")
                is_int8 = enc_int8.exists()
                logger.info(f"X-ASR loading local {chunk}ms {local_base} {'(int8 152M)' if is_int8 else ''}")
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
            self.backend = f"x-asr-sherpa-{chunk}ms" + ("-int8" if "int8" in enc else "")
            enc_mb = Path(enc).stat().st_size/1024/1024
            logger.info(f"X-ASR sherpa-onnx loaded ✓ {self.backend} {model_id} (160ms streaming, {enc_mb:.0f}M encoder{' int8' if 'int8' in enc else ''})")
        except Exception as e:
            logger.warning(f"X-ASR sherpa load failed {e}, trying whisper fallback")
            import traceback
            traceback.print_exc()
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

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event, session_id: str | None = None) -> AsyncGenerator[dict, None]:
        """Yield stt_partial/stt_final events for one connection.

        `session_id` is accepted for interface parity with the other STT adapters
        (the paraformer/ark wrappers share this signature); this backend doesn't
        need it because each call creates its own private decoder stream.
        """
        if self.mock or self.backend == "mock":
            # Mock streaming as before
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
            # ONE STREAM PER CONNECTION. This used to reuse the single self.stream
            # created in __init__, so two browser tabs (or a tab plus a test client)
            # both pushed audio into the same decoder state and their utterances
            # interleaved into each other's transcripts. `recognizer` is safe to
            # share; `stream` is per-utterance-by-design, so it is created here.
            # All sherpa calls are serialized because each burst is awaited before the
            # next is issued (no two threads touch the same stream concurrently).
            stream = self.recognizer.create_stream()
            last_partial = ""
            last_partial_t = time.time()

            def _burst(st: "object", finish: bool = False):
                """Blocking sherpa ONNX inference, run OFF the event loop.

                These calls are synchronous native code on the hot path: run inline
                they stalled the whole asyncio loop (LLM token streaming, WS sends,
                every other session) once per 20ms audio frame."""
                if finish:
                    st.input_finished()
                while self.recognizer.is_ready(st):
                    self.recognizer.decode_stream(st)
                # OnlineStream has no .text attribute — the result comes from the
                # recognizer (get_result_all), which is also why this whole burst is
                # one function run off-loop.
                return self.recognizer.get_result_all(st).text.strip()

            try:
                while not stop_event.is_set():
                    try:
                        item = await asyncio.wait_for(pcm_queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        item = None
                    # Handle flush
                    if item is not None and isinstance(item, dict) and item.get("type") == "flush":
                        # Signal end, get final
                        text = await asyncio.to_thread(_burst, stream, True)
                        if text:
                            yield {"type": "stt_final", "text": text, "latency_ms": 30}
                        # Reset for next utterance
                        # Discard the utterance's stream outright (a fresh one is
                        # created anyway); reset() only mutates state we throw away.
                        stream = self.recognizer.create_stream()
                        last_partial = ""
                        continue
                    if item is not None:
                        # item is int16 PCM
                        pcm_f32 = item.astype(np.float32) / 32768.0
                        # sherpa expects float32 [-1,1] at 16k
                        stream.accept_waveform(SAMPLE_RATE, pcm_f32)
                        # Decode if ready (off-loop)
                        text = await asyncio.to_thread(_burst, stream)
                        # Try to get partial
                        now = time.time()
                        if (now - last_partial_t)*1000 >= PARTIAL_INTERVAL_MS:
                            last_partial_t = now
                            if text and text != last_partial:
                                last_partial = text
                                yield {"type": "stt_partial", "text": text, "latency_ms": 40}
                        # Check endpoint
                        if self.recognizer.is_endpoint(stream):
                            if text:
                                yield {"type": "stt_final", "text": text, "latency_ms": 40}
                            stream = self.recognizer.create_stream()
                            last_partial = ""
                    else:
                        # No new audio, still decode and check endpoint
                        text = await asyncio.to_thread(_burst, stream)
                        if self.recognizer.is_endpoint(stream):
                            if text:
                                yield {"type": "stt_final", "text": text, "latency_ms": 40}
                            stream = self.recognizer.create_stream()
                            last_partial = ""
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception(f"X-ASR stream error {e}")
            return

        # Fallback whisper streaming (existing logic)
        # Simplified: use utterance_buffer + transcribe
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
                segs, _ = self.whisper_model.transcribe(pcm_f32, language=None, beam_size=1, vad_filter=False, without_timestamps=True)  # auto-detect per utterance — zh-TW is default, but never hardcode away English entirely
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
                segs, _ = self.whisper_model.transcribe(pcm_f32, language=None, beam_size=1, vad_filter=False, without_timestamps=True)  # auto-detect per utterance — zh-TW is default, but never hardcode away English entirely
                return " ".join([s.text.strip() for s in segs])
            return (await asyncio.to_thread(_infer)).strip()
        elif hasattr(self, 'pipe') and self.pipe:
            def _infer2():
                out = self.pipe({"array": pcm_f32, "sampling_rate": SAMPLE_RATE})
                return out["text"] if isinstance(out, dict) else str(out)
            return (await asyncio.to_thread(_infer2)).strip()
        return "mock transcription"
