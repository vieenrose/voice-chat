"""
Qwen3-TTS 12Hz 0.6B streaming wrapper — GGML backend via qwentts-cpp-python (CUDA 12.4)
Follows https://github.com/huggingface/speech-to-speech#cuda-note-for-qwen3-tts:
  pip install "qwentts-cpp-python==0.3.1+cu124" -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu124
Loads GGUF talker + codec directly from /tmp/qwen3_tts (downloaded via curl to avoid HF xet hang).
Output: 24 kHz mono int16 (Qwen3-TTS 12Hz -> 24000 Hz).
"""
import asyncio
import time
import re
import threading
from pathlib import Path
from typing import AsyncGenerator
import numpy as np
from loguru import logger

SAMPLE_RATE = 24000  # Qwen3-TTS 12Hz * 2000
FLUSH_TOKENS = 26   # flush only at real sentence ends (was 8 -> fragmented sentences => pauses INSIDE a sentence)
SENTENCE_END = re.compile(r'[.!?。！？\n]')

class StreamingPrimeTTS:
    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = False
        self.backend = "qwen3-tts-0.6b"
        self.runtime = None
        self.sample_rate = SAMPLE_RATE
        self.speaker = "Aiden"
        # --- load GGML model (CustomVoice Q8_0, 924MB -> STABLE + 9 built-in speakers; Q4_K_M + Base-clone unstable on this qwentts build) ---
        talker = Path("/tmp/qwen3_tts/talker_cv_q8.gguf")
        codec = Path("/tmp/qwen3_tts/codec.gguf")
        if not talker.exists() or not codec.exists():
            raise RuntimeError(f"Qwen3-TTS GGUF missing: {talker.exists()} {codec.exists()} — run /tmp/dl_curl.sh")
        try:
            from faster_qwen3_tts.ggml_backend import _require_qwentts_cpp
            QwenTTS, _ = _require_qwentts_cpp()
            logger.info(f"Loading Qwen3-TTS GGML {talker} + {codec}...")
            self.runtime = QwenTTS(str(talker), str(codec))
            from faster_qwen3_tts import GGMLQwen3TTS
            self.model = GGMLQwen3TTS(self.runtime)
            try:
                self.sample_rate = int(self.runtime.sample_rate)
            except Exception:
                self.sample_rate = SAMPLE_RATE
            try:
                self.runtime.warmup(prefill_len=16)
            except Exception as e:
                logger.warning(f"Qwen warmup skipped {e}")
            logger.info(f"Qwen3-TTS loaded ✓ {model_id} sample_rate={self.sample_rate}Hz (TRUE streaming enabled)")
        except Exception as e:
            logger.error(f"Qwen3-TTS load failed {e}")
            import traceback; traceback.print_exc()
            raise RuntimeError(f"Qwen3-TTS required: {e}")
        # --- voice presets: CustomVoice built-in speakers (stable; clone refs are Base-only & unstable here) ---
        # speakers: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan
        self.VOICE_PRESETS = {
            "Aiden": {"type": "speaker", "name": "Aiden"},
            "中文女声": {"type": "speaker", "name": "vivian"},
            "中文男声": {"type": "speaker", "name": "uncle_fu"},
        }
        self.VOICE_PRESETS["台湾腔"] = {"type": "speaker", "name": "vivian"}
        self.voice_refs: dict[str, dict] = {}
        self._vv = "Aiden"
    @property
    def voice(self) -> str:
        return self._vv
    def voices(self) -> list[str]:
        return list(self.VOICE_PRESETS.keys())
    def set_voice(self, name: str):
        if name not in self.VOICE_PRESETS:
            raise KeyError(f"unknown voice {name}; available {self.voices()}")
        self._vv = name
        self.speaker = self.VOICE_PRESETS[name].get("name", name)
    async def _ensure_voice_ref(self, preset: dict) -> dict:
        """Extract & cache .spk/.rvq from reference audio (24k mono f32) — like MOSS voice_clone."""
        key = preset["ref_audio"]
        if key in self.voice_refs or Path(preset["ref_spk"]).exists():
            self.voice_refs[key] = preset
            return preset
        logger.info(f"Extracting voice clone ref from {key} (台湾腔)...")
        import soundfile as sf, scipy.signal as sp
        wav, sr = sf.read(preset["ref_audio"], dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 24000:
            wav = sp.resample_poly(wav, 24000, sr).astype(np.float32)
        def _extract():
            vr = self.runtime.extract_voice_ref(wav)
            vr.save(preset["ref_spk"], preset["ref_rvq"])
        await asyncio.to_thread(_extract)
        self.voice_refs[key] = preset
        logger.info(f"Saved 台湾腔 voice ref -> {preset['ref_spk']}/{preset['ref_rvq']}")
        return preset

    async def synthesize(self, text: str, reference_audio: str = None) -> np.ndarray:
        preset = self.VOICE_PRESETS[self._vv]
        try:
            if preset.get("type") == "clone":
                # 台湾腔 via reference-audio cloning (same as MOSS voice_clone with zh_4.wav)
                await self._ensure_voice_ref(preset)
                def _infer_clone():
                    import qwentts_cpp as qc
                    ref_spk = qc.load_speaker_embedding(preset["ref_spk"])
                    ref_codes = qc.load_rvq_codes(preset["ref_rvq"], self.runtime.num_codebooks())
                    audio, sr = self.runtime.synthesize(
                        text=text,
                        lang="auto",
                        ref_spk_emb=ref_spk,
                        ref_codes=ref_codes,
                        ref_text=preset["ref_text"],
                        max_new_tokens=512,
                        temperature=0.9,
                        top_k=50,
                        top_p=1.0,
                    )
                    return audio, sr
                audio, sr = await asyncio.to_thread(_infer_clone)
                arr = np.array(audio, dtype=np.float32).squeeze()
                arr = np.nan_to_num(arr); arr = np.clip(arr, -1.0, 1.0)
                pcm = (arr * 32767).astype(np.int16)
                return pcm
            def _infer():
                audio, sr = self.runtime.synthesize(
                    text=text,
                    lang="auto",
                    speaker=self.speaker,
                    max_new_tokens=512,
                    temperature=0.9,
                    top_k=50,
                    top_p=1.0,
                )
                arr = np.array(audio, dtype=np.float32).squeeze()
                # Audio likely float -1..1; clamp and convert to int16
                arr = np.nan_to_num(arr)
                arr = np.clip(arr, -1.0, 1.0)
                pcm = (arr * 32767).astype(np.int16)
                return pcm
            pcm = await asyncio.to_thread(_infer)
            return pcm
        except Exception as e:
            logger.error(f"Qwen3-TTS synthesize failed {e}")
            import traceback; traceback.print_exc()
            return np.zeros(int(SAMPLE_RATE*0.4), dtype=np.int16)

    @staticmethod
    def _compress_silence(pcm16: np.ndarray, max_gap_s: float = 0.26, keep_s: float = 0.16) -> np.ndarray:
        """Shrink long intra-audio silence runs to keep_s. Qwen3-TTS prosody inserts
        0.2-0.45s pauses inside sentences (after exclamations, at commas) — cap them."""
        sr = SAMPLE_RATE
        thresh = 40  # int16 level (~ -58dB) counts as silence
        abs_w = np.abs(pcm16.astype(np.int32))
        quiet = abs_w < thresh
        max_n = int(max_gap_s * sr); keep_n = int(keep_s * sr)
        # trim edge padding: leading/trailing silence of a chunk -> tiny 30ms
        edge_n = int(0.03 * sr)
        nz = np.nonzero(~quiet)[0]
        if nz.size:
            first, last = int(nz[0]), int(nz[-1])
            if first > edge_n:
                pcm16 = pcm16[first - edge_n:]
                quiet = np.abs(pcm16.astype(np.int32)) < thresh
            if len(pcm16) - 1 - last > edge_n:  # recompute last on (possibly) trimmed array
                nz2 = np.nonzero(~quiet)[0]
                if nz2.size:
                    last2 = int(nz2[-1])
                    if len(pcm16) - 1 - last2 > edge_n:
                        pcm16 = pcm16[:last2 + edge_n]
                        quiet = np.abs(pcm16.astype(np.int32)) < thresh
        out = []
        i = 0; n = len(pcm16)
        while i < n:
            if quiet[i]:
                j = i
                while j < n and quiet[j]:
                    j += 1
                run = j - i
                if run > max_n:
                    out.append(pcm16[i:i + keep_n])      # keep a short natural pause
                    i = j
                else:
                    out.append(pcm16[i:j]); i = j
            else:
                j = i
                while j < n and not quiet[j]:
                    j += 1
                out.append(pcm16[i:j]); i = j
        return np.concatenate(out) if out else pcm16

    async def synthesize_streaming(self, text: str, chunk_frames: int = 24) -> AsyncGenerator[np.ndarray, None]:
        """TRUE token-streaming via generate_custom_voice_streaming (TTFA ~20ms, like HF framework).
        chunk_frames=24 -> ~1.7s audio per chunk (bigger chunks = fewer jitter boundaries)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        loop = asyncio.get_running_loop()

        def _run():
            try:
                preset = self.VOICE_PRESETS[self._vv]
                if preset.get("type") == "clone":
                    # 台湾腔 streaming via voice-clone generator
                    import asyncio as _a
                    fut = asyncio.run_coroutine_threadsafe(self._ensure_voice_ref(preset), loop)
                    fut.result(timeout=120)
                    gen = self.model.generate_voice_clone_streaming(
                        text=text,
                        language="auto",
                        ref_audio=preset["ref_audio"],
                        ref_text=preset["ref_text"],
                        chunk_size=max(4, int(chunk_frames)),
                    )
                else:
                    gen = self.model.generate_custom_voice_streaming(
                        text=text,
                        speaker=self.speaker,
                        language="auto",
                        chunk_size=max(4, int(chunk_frames)),
                    )
                for chunk, sr, meta in gen:
                    arr = np.array(chunk, dtype=np.float32).squeeze()
                    arr = np.nan_to_num(arr)
                    arr = np.clip(arr, -1.0, 1.0)
                    pcm16 = (arr * 32767).astype(np.int16)
                    asyncio.run_coroutine_threadsafe(q.put(("pcm", pcm16, meta)), loop).result()
            except Exception as e:
                logger.error(f"Qwen streaming error {e}")
                asyncio.run_coroutine_threadsafe(q.put(("err", e, {})), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(("done", None, {})), loop).result()

        threading.Thread(target=_run, daemon=True).start()
        # POOL the runtime's tiny ramp chunks (0.08-0.64s) into stable ~1.2s chunks:
        # the runtime yields powers-of-two frame batches regardless of chunk_size,
        # and on iOS's 0.25s-quantized AudioContext clock those micro-boundaries = audible pauses.
        MIN_CHUNK_S = 1.2
        _buf = np.zeros(0, dtype=np.int16)
        _ttfa_logged = False
        while True:
            kind, payload, meta = await q.get()
            if kind == "done":
                break
            if kind == "err":
                raise payload
            if meta and meta.get("chunk_index") == 0 and not _ttfa_logged:
                ttfa = float(meta.get("total_ms", 0) or 0)
                logger.info(f"Qwen3-TTS TTFA {ttfa:.0f}ms (first frame @{self.sample_rate}Hz)")
                _ttfa_logged = True
            _buf = np.concatenate([_buf, payload])
            if len(_buf) / self.sample_rate >= MIN_CHUNK_S:
                yield self._compress_silence(_buf)
                _buf = np.zeros(0, dtype=np.int16)
        if len(_buf):
            yield self._compress_silence(_buf)
        _buf = np.zeros(0, dtype=np.int16)

    async def stream_tts(self, token_stream: AsyncGenerator[dict, None]) -> AsyncGenerator[dict, None]:
        buf = ""; token_count = 0
        async for chunk in token_stream:
            if chunk["type"] == "llm_token":
                token = chunk["token"]; buf += token; token_count += 1
                should_flush = False
                if SENTENCE_END.search(token): should_flush = True
                elif token_count >= FLUSH_TOKENS and buf and buf[-1] in " ,": should_flush = True
                if should_flush and buf.strip():
                    txt = buf.strip(); buf = ""; token_count = 0
                    async for pcm_chunk in self.synthesize_streaming(txt):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": txt, "sampleRate": self.sample_rate, "latency_ms": 40}
            elif chunk["type"] == "llm_done":
                if buf.strip():
                    async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                        yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": self.sample_rate, "latency_ms": 40}
                yield {"type": "tts_end"}; return
        if buf.strip():
            async for pcm_chunk in self.synthesize_streaming(buf.strip()):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": buf.strip(), "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def tts_from_text(self, text: str) -> AsyncGenerator[dict, None]:
        sentences = re.split(r'([.!?]+)', text)
        for i in range(0, len(sentences), 2):
            sent = (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")).strip()
            if not sent: continue
            async for pcm_chunk in self.synthesize_streaming(sent):
                yield {"type": "tts_chunk", "pcm": pcm_chunk, "text": sent, "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}
