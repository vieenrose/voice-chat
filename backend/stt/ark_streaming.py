"""
ARK-ASR 0.6B streaming wrapper — Audio8/ARK-ASR-0.6B
Multilingual ASR (19 langs: zh,en,de,ja,fr,ko,es,pl,it,ro,hu,cs,nl,fi,hr,sk,sl,et,lt)
16kHz, Whisper-style encoder + Qwen2 decoder, trust_remote_code
For streaming, we buffer 20ms chunks and do partials every 300ms via model.generate
"""
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import AsyncGenerator
from loguru import logger

SAMPLE_RATE = 16000

class StreamingXASR:
    def __init__(self, model_id: str = "Audio8/ARK-ASR-0.6B", device: str = "cuda", mock: bool = False, **kwargs):
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.backend = "mock"
        self.processor = None
        self.tokenizer = None
        self.model = None
        if mock:
            logger.info("ARK-ASR: MOCK mode")
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Loading ARK-ASR {model_id} on {device}...")
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, dtype=dtype,
                attn_implementation="sdpa",
                device_map="auto" if device=="cuda" else None,
            )
            self.model.eval()
            self.backend = "ark-asr-0.6b"
            logger.info(f"ARK-ASR loaded ✓ {model_id} on {device}")
        except Exception as e:
            logger.error(f"ARK-ASR load failed {e}, fallback to mock")
            import traceback; traceback.print_exc()
            self.backend = "mock"
            self.mock = True

    def _is_speech(self, pcm: np.ndarray) -> bool:
        return np.sqrt(np.mean(pcm.astype(np.float32)**2)) > 0.01

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event) -> AsyncGenerator[dict, None]:
        if self.mock or self.backend == "mock":
            # Mock as before
            mock_words = ["hello","how","are","you","today","ark","asr","streaming","audio8","multilingual"]
            mock_idx = 0
            utterance_buffer = np.zeros(0, dtype=np.float32)
            last_partial_t = time.time()
            last_speech_t = time.time()
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
                            continue
                        pcm_f32 = item.astype(np.float32) / 32768.0
                        utterance_buffer = np.concatenate([utterance_buffer, pcm_f32])
                        if self._is_speech(pcm_f32):
                            last_speech_t = now
                    if len(utterance_buffer) > 0 and (now - last_partial_t)*1000 >= 300:
                        last_partial_t = now
                        if mock_idx < len(mock_words):
                            partial = " ".join(mock_words[:mock_idx+1])
                            mock_idx = min(mock_idx+1, len(mock_words)-1)
                        yield {"type": "stt_partial", "text": partial, "latency_ms": 20}
                    if len(utterance_buffer) > 0.8*SAMPLE_RATE and (now - last_speech_t)*1000 > VAD_SILENCE_MS:
                        text = " ".join(mock_words[:min(mock_idx+3, len(mock_words))])
                        if text.strip():
                            yield {"type": "stt_final", "text": text.strip(), "latency_ms": 20}
                        utterance_buffer = np.zeros(0, dtype=np.float32)
                        mock_idx = 0
                except asyncio.CancelledError:
                    break
            return

        # Real ARK-ASR streaming - buffer and run model.generate for partials
        utterance_buffer = np.zeros(0, dtype=np.float32)
        last_partial_t = time.time()
        last_speech_t = time.time()
        last_partial = ""
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
                            text = await self._transcribe_buffer(utterance_buffer)
                            if text.strip():
                                yield {"type": "stt_final", "text": text.strip(), "latency_ms": 80}
                            utterance_buffer = np.zeros(0, dtype=np.float32)
                            last_partial = ""
                        continue
                    pcm_f32 = item.astype(np.float32) / 32768.0
                    utterance_buffer = np.concatenate([utterance_buffer, pcm_f32])
                    if self._is_speech(pcm_f32):
                        last_speech_t = now
                # Partial every 300ms if we have >0.6s audio
                if len(utterance_buffer) > 0.6*SAMPLE_RATE and (now - last_partial_t)*1000 >= 300:
                    last_partial_t = now
                    recent = utterance_buffer[-int(2*SAMPLE_RATE):]
                    text = await self._transcribe_buffer(recent)
                    if text and text != last_partial:
                        last_partial = text
                        yield {"type": "stt_partial", "text": text, "latency_ms": 80}
                # Endpoint
                if len(utterance_buffer) > 0.8*SAMPLE_RATE and (now - last_speech_t)*1000 > VAD_SILENCE_MS:
                    text = await self._transcribe_buffer(utterance_buffer)
                    if text.strip():
                        yield {"type": "stt_final", "text": text.strip(), "latency_ms": 80}
                    utterance_buffer = np.zeros(0, dtype=np.float32)
                    last_partial = ""
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"ARK-ASR stream error {e}")
                await asyncio.sleep(0.05)

    async def _transcribe_buffer(self, pcm_f32: np.ndarray) -> str:
        if self.mock or self.backend == "mock":
            return "hello"
        try:
            import torch
            # ARK-ASR expects conversation with audio + text
            # We need to create a temp wav in memory and use processor
            # For streaming, we use the processor's chat template
            def _infer():
                # Save pcm to temp file for processor (it expects path)
                import tempfile, soundfile as sf, os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, pcm_f32, SAMPLE_RATE)
                    tmp_path = f.name
                try:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "path": tmp_path},
                                {"type": "text", "text": "Please transcribe this audio."},
                            ],
                        }
                    ]
                    inputs = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        sampling_rate=SAMPLE_RATE,
                        audio_max_length=30*SAMPLE_RATE,
                    )
                    # Move to device
                    device = next(self.model.parameters()).device
                    # Use bfloat16 for cuda to match model (fixes Input type float vs Half)
                    target_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    if "audios" in inputs and isinstance(inputs["audios"], torch.Tensor):
                        inputs["audios"] = inputs["audios"].to(device=device, dtype=target_dtype)
                    # Also ensure input_ids is long
                    if "input_ids" in inputs:
                        inputs["input_ids"] = inputs["input_ids"].to(device)
                    # Bad words for cleaner output
                    bad_words_ids = None
                    try:
                        eos_ids = self.tokenizer.eos_token_id
                        keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
                        bad_ids = set(self.tokenizer.all_special_ids) - keep_ids
                        bad_ids.update(token_id for token, token_id in self.tokenizer.get_added_vocab().items() if token.startswith("<") and token.endswith(">") and token_id not in keep_ids)
                        bad_words_ids = [[tid] for tid in sorted(bad_ids)]
                    except:
                        pass
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=128,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            bad_words_ids=bad_words_ids,
                        )
                    # Decode only new tokens
                    input_len = inputs["input_ids"].shape[1]
                    decoded = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
                    # Cleanup
                    text = decoded[0].strip() if decoded else ""
                    # ARK-ASR may include thinking? Clean up
                    text = text.replace("<|im_end|>", "").strip()
                    return text
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            text = await asyncio.to_thread(_infer)
            return text.strip()
        except Exception as e:
            logger.warning(f"ARK-ASR transcribe failed {e}")
            import traceback; traceback.print_exc()
            return ""

    async def transcribe_once(self, pcm_f32: np.ndarray) -> str:
        return await self._transcribe_buffer(pcm_f32)
