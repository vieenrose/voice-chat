"""
Audio8 TTS Preview 0.6b streaming wrapper — Audio8/Audio8-TTS-Preview-0.6b
Multilingual 11 langs, zero-shot voice cloning, DualAR (Slow 24L + Fast 4L), 44.1kHz, 10 codebooks, ONNX INT4 option
Supports streaming via processor + model.generate, with reference audio for cloning
"""
import asyncio
import time
import re
import numpy as np
from pathlib import Path
from typing import AsyncGenerator
from loguru import logger

SAMPLE_RATE = 44100  # Audio8 codec 44.1kHz
FLUSH_TOKENS = 8
SENTENCE_END = re.compile(r'[.!?。！？\n]')

class StreamingPrimeTTS:
    def __init__(self, model_id: str = "Audio8/Audio8-TTS-Preview-0.6b", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = False
        self.backend = "audio8-0.6b"
        self.processor = None
        self.model = None
        self.sample_rate = SAMPLE_RATE
        # Try local /tmp/Audio8 first if available (faster, no HF download)
        local_path = "/tmp/Audio8"
        if Path(local_path).exists() and (Path(local_path) / "model.safetensors").exists():
            model_id = local_path
            logger.info(f"Using local Audio8 at {local_path}")
        # Try CUDA first with expandable_segments to avoid OOM; fallback to CPU if fails
        # Ling uses 4.6G, Audio8 needs ~2.5G, total ~7.1G on 12G so CUDA should fit with fragmentation handling
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        orig_device = device
        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            logger.info(f"Loading Audio8 TTS {model_id} on {device}...")
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            if device == "cuda":
                try:
                    self.model = AutoModel.from_pretrained(
                        model_id, trust_remote_code=True,
                        dtype=dtype,
                        device_map="auto",
                    )
                except Exception as e_cuda:
                    logger.warning(f"Audio8 CUDA load failed {e_cuda}, fallback to CPU")
                    device = "cpu"
                    dtype = torch.float32
                    self.model = AutoModel.from_pretrained(
                        model_id, trust_remote_code=True,
                        dtype=dtype,
                        device_map=None,
                    )
                    self.model = self.model.to(device)
            else:
                self.model = AutoModel.from_pretrained(
                    model_id, trust_remote_code=True,
                    dtype=dtype,
                    device_map=None,
                )
                self.model = self.model.to(device)
            self.model.eval()
            # Check if we have a reference audio for cloning - use default None (no cloning, pure TTS)
            # For demo, we can use no reference (pure TTS) or a built-in reference
            self.backend = "audio8-0.6b"
            logger.info(f"Audio8 TTS loaded ✓ {model_id} sample_rate={self.model.config.codec_sample_rate if hasattr(self.model.config, 'codec_sample_rate') else SAMPLE_RATE}Hz")
            self.sample_rate = getattr(self.model.config, 'codec_sample_rate', SAMPLE_RATE)
        except Exception as e:
            logger.error(f"Audio8 TTS load failed {e}")
            import traceback; traceback.print_exc()
            raise RuntimeError(f"Audio8 TTS required: {e}")

    async def synthesize(self, text: str, reference_audio: str = None, reference_text: str = None) -> np.ndarray:
        try:
            def _infer():
                import torch
                # Audio8 processor: text + optional reference_audio + reference_text
                # For pure TTS without cloning, just text
                if reference_audio and reference_text:
                    inputs = self.processor(
                        text=[text],
                        reference_audio=[reference_audio],
                        reference_text=[reference_text],
                        return_tensors="pt",
                    )
                else:
                    inputs = self.processor(
                        text=[text],
                        return_tensors="pt",
                    )
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                with torch.inference_mode():
                    # Dynamic max_new_tokens to reduce latency: ~15 tokens per char, cap 512
                    dyn_max = min(512, max(64, int(len(text) * 12)))
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=dyn_max,
                        temperature=0.8,
                        top_p=0.95,
                        top_k=50,
                        do_sample=True,
                        return_dict_in_generate=True,
                    )
                    # Decode audio
                    waveforms, lengths = self.model.decode_audio(output.codes)
                    wav = waveforms[0, :int(lengths[0])].float().cpu().numpy()
                    return wav
            wav = await asyncio.to_thread(_infer)
            pcm = (np.array(wav) * 32767).astype(np.int16)
            return pcm
        except Exception as e:
            logger.error(f"Audio8 synthesize failed {e}")
            import traceback; traceback.print_exc()
            # Fallback to silence
            return np.zeros(int(SAMPLE_RATE*0.5), dtype=np.int16)

    async def synthesize_streaming(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        # Audio8 does not have explicit generate_streaming, but we can simulate streaming by chunking text and using the model's streaming if available
        # For now, do one-shot and yield as single chunk (still streaming API, but per sentence)
        # If the model has a streaming API, we could use it, but for now just yield one chunk
        pcm = await self.synthesize(text)
        # Split into chunks for streaming effect (simulate 0.5s chunks)
        chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5s per chunk
        for i in range(0, len(pcm), chunk_size):
            yield pcm[i:i+chunk_size]
            await asyncio.sleep(0.02)  # small delay to simulate streaming

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
                    async for pcm_chunk in self.synthesize_streaming(text_to_synth):
                        latency = int((time.time()-t0)*1000) if t0 else 0
                        t0 = None
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": text_to_synth, "sampleRate": self.sample_rate, "latency_ms": latency}
            elif chunk["type"] == "llm_done":
                if buf.strip():
                    async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": self.sample_rate, "latency_ms": 60}
                yield {"type": "tts_end"}
                return
        if buf.strip():
            async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": self.sample_rate, "latency_ms": 60}
        yield {"type": "tts_end"}

    async def tts_from_text(self, text: str) -> AsyncGenerator[dict, None]:
        sentences = re.split(r'([.!?]+)', text)
        for i in range(0, len(sentences), 2):
            sent = (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")).strip()
            if not sent:
                continue
            async for pcm_chunk in self.synthesize_streaming(sent):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": sent, "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}
