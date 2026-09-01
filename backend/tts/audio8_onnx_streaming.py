"""
Audio8 TTS Preview 0.1B — ONNX INT8, CPU, true streaming, Apache-2.0.

Why this engine exists in the ladder
------------------------------------
The previous default (Qwen3-TTS 0.6B, GGML/CUDA) holds the GPU and mis pronounces
mixed-script text. This one is:
  * ~100M params, INT8 dynamic-quantized ONNX graphs, CPUExecutionProvider only
    (20-thread box: no GPU VRAM at all, so the STT + LLM keep the whole card),
  * streaming by design: `ArkTtsRuntime.stream()` decodes a sliding window of codec
    frames and keeps a 1-frame guard, so boundary samples are never emitted twice,
  * `stop_event` on the generation loop -> barge-in actually stops the CPU work,
  * 11 languages incl. Cantonese/Japanese/Korean, which is the code-switch class the
    Qwen3 path handled by splitting text into per-language segments.

Integration notes
-----------------
Runtime code is NOT vendored: it lives in the Audio8_TTS checkout as `arktts_runtime`
(Apache-2.0). Point at it with AUDIO8_RUNTIME_DIR. Model files come from
`Audio8/audio8-TTS-0.1B-ONNX-INT8` (AUDIO8_MODEL_DIR); the 414 MB voice-registration
encoder is optional and only needed to clone new voices.

Output is 44.1 kHz mono; every adapter in this repo yields **int16** PCM and the
frontend resamples per chunk using the `sampleRate` it is told, so no other component
changes.

Env knobs: AUDIO8_MODEL_DIR, AUDIO8_RUNTIME_DIR, AUDIO8_VOICES_DIR, AUDIO8_THREADS,
AUDIO8_CHUNK_FRAMES, AUDIO8_TEMPERATURE/TOP_P/TOP_K/SEED, AUDIO8_MAX_FRAMES.

Measured on this box (i7-12700, 20 threads, RTX 3060 12G) — kept as an *option*, not the
default, because these numbers are worse than the incumbent on every axis that matters:
  * slow AR       30 ms/frame at 1-8 threads; 69 ms/frame at 20 (oversubscription; the
                  default `cpu_count()` thread count was measured to be 2x slower)
  * codec decode  22-35 ms/frame; upstream stream() re-decodes
                  (stream_context_frames + chunk_frames) per emitted chunk, so small
                  chunks cost 4-14x the minimum audio time (RTF 9-12 measured at
                  chunk_frames=10)
  * prefill       ~2.9 s per call: the exported graph is one-token, and the prompt replays
                  the 110-frame reference voice every time (cacheable in principle, the
                  reference block sits before the target text)
  * TTFA          5.2-6.8 s measured end to end; Qwen3-TTS is ~0.2-0.5 s
  * pronunciation WORSE, not better: see test_tts_asr_roundtrip.py / README "Pronunciation"
                  (plain Chinese 29% CER vs Qwen3-TTS 9%, long sentences 30% vs 3%)
  * onnxruntime   upstream pins >=1.22,<1.24; this box runs 1.29.0 (int8 graphs load and
                  produce audio), so the pin is not enforced here - verify after any bump.
"""
import asyncio
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from loguru import logger

DEFAULT_MODEL_DIR = os.getenv("AUDIO8_MODEL_DIR", "/home/user/models/audio8")
DEFAULT_RUNTIME_DIR = os.getenv("AUDIO8_RUNTIME_DIR", "/home/user/models/Audio8_TTS/onnx_runtime_0_1b_int8")
DEFAULT_VOICES_DIR = os.getenv("AUDIO8_VOICES_DIR", "")          # default: <model_dir>/voices

# Frames are 2048 samples at 44.1 kHz -> 21.5 frames/s of audio. The model stops at its
# own EOS, so over-budgeting only wastes CPU on an early stop; under-budgeting cuts the
# sentence off mid-word, which is the exact bug class we are trying to kill (measured:
# 4.8 frames/char truncated "...攝氏三十四度" at "三十"). Bias high, cap for runaway loops.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u00b7]")
_MD = [(re.compile(r"\*\*(.+?)\*\*"), r"\1"), (re.compile(r"\*(.+?)\*"), r"\1"),
       (re.compile(r"`{1,3}([^`]*)`{1,3}"), r"\1"), (re.compile(r"\[(.+?)\]\([^)]*\)"), r"\1"),
       (re.compile(r"^#{1,6}\s*", re.M), ""), (re.compile(r"^\s*[-*]\s+", re.M), "")]


def plain_text(text: str) -> str:
    """Strip markdown wrappers that the LLM emits and that TTS reads aloud as symbols."""
    out = text or ""
    for pat, rep in _MD:
        out = pat.sub(rep, out)
    return re.sub(r"\s+", " ", out).strip()


def budget_frames(text: str, hard_max: int = 1000) -> int:
    cjk = len(_CJK.findall(text))
    other = max(0, len(text) - cjk)
    return max(48, min(hard_max, 32 + int(6.5 * cjk) + int(2.2 * other)))


class StreamingPrimeTTS:
    """Interface-compatible wrapper around `arktts_runtime.runtime.ArkTtsRuntime`."""

    def __init__(self, model_id: str = "Audio8/audio8-TTS-0.1B-ONNX-INT8", device: str = "cpu",
                 mock: bool = False, model_dir: str | None = None, voices_dir: str | None = None,
                 threads: int | None = None):
        self.model_id = model_id
        self.mock = False
        self.backend = "audio8-tts-0.1b-onnx-int8"
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        runtime_dir = Path(DEFAULT_RUNTIME_DIR)
        self.voices_dir = Path(voices_dir or DEFAULT_VOICES_DIR or (self.model_dir / "voices"))
        self.threads = int(threads or os.getenv("AUDIO8_THREADS") or max(4, (os.cpu_count() or 8)))
        self.chunk_frames = int(os.getenv("AUDIO8_CHUNK_FRAMES") or 10)
        self.temperature = float(os.getenv("AUDIO8_TEMPERATURE") or 0.7)
        self.top_p = float(os.getenv("AUDIO8_TOP_P") or 0.9)
        self.top_k = int(os.getenv("AUDIO8_TOP_K") or 50)
        self.seed = int(os.getenv("AUDIO8_SEED") or 42)
        self.last_stats: dict = {}

        if mock:
            raise ImportError("audio8-onnx has no mock mode; the pipeline falls back to tts.mock_streaming")

        # Everything below is an ImportError on purpose: the pipeline's engine ladder and
        # its construct-then-fallback path both key on ImportError, and a missing model
        # directory is exactly that kind of problem (not a bug to surface as a 500).
        if not (self.model_dir / "runtime_manifest.json").is_file():
            raise ImportError(f"Audio8 model not found at {self.model_dir} "
                              "(hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir ...)")
        if not (runtime_dir / "arktts_runtime").is_dir():
            raise ImportError(f"arktts_runtime not found under {runtime_dir} "
                              "(git clone https://github.com/Audio8-AI/Audio8_TTS)")
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))
        try:
            from arktts_runtime.runtime import ArkTtsRuntime
        except Exception as e:                            # onnxruntime missing/incompatible, etc.
            raise ImportError(f"arktts_runtime not importable: {e}") from e

        self._ensure_default_voice()
        logger.info(f"Audio8-TTS: loading INT8 graphs (CPU EP, {self.threads} threads) from {self.model_dir}")
        t0 = __import__("time").time()
        self.rt = ArkTtsRuntime(model_dir=self.model_dir, voices_dir=self.voices_dir, threads=self.threads)
        self.sample_rate = int(self.rt.manifest["sample_rate"])
        self.frame_samples = int(self.rt.manifest["codec_hop_length"])
        self.frames_per_s = self.sample_rate / self.frame_samples
        self._vv = self._pick_default_voice()
        logger.info(f"Audio8-TTS ready ✓ ({__import__('time').time()-t0:.1f}s) sr={self.sample_rate} "
                    f"voice={self._vv} voices={self.voices}")

    # ------------------------------------------------------------------ voices
    def _ensure_default_voice(self):
        """Build the packaged `default` voice from the shipped reference codes.

        Same shape as upstream scripts/register_default_voice.py, but automatic: the
        service must not need a separate setup command to have a working voice.
        """
        target = self.voices_dir / "default"
        if (target / "codes.npy").is_file() and (target / "meta.json").is_file():
            return
        manifest = json.loads((self.model_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
        codes = np.load(self.model_dir / manifest.get("reference_codes", "reference_codes.npy"),
                        allow_pickle=False)
        if codes.ndim != 2 or codes.shape[0] != int(manifest["num_codebooks"]) or codes.shape[1] == 0:
            raise ImportError(f"packaged reference codes have unusable shape {codes.shape}")
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "codes.npy", codes.astype(np.uint16, copy=False))
        (target / "meta.json").write_text(json.dumps({
            "name": "default",
            "reference_text": str(manifest["reference_text"]),
            "shape": list(codes.shape),
            "dtype": "uint16",
            "sample_rate": int(manifest["sample_rate"]),
            "model_fingerprint": manifest.get("model_fingerprint", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": "model_reference_codes",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"Audio8-TTS: registered packaged voice 'default' {tuple(codes.shape)} "
                    f"(≈{codes.shape[1] / 21.5:.1f}s reference -> that many prefill tokens per call)")

    @property
    def voices(self) -> list[str]:
        try:
            names = [v.get("name") for v in self.rt.voices.list() if v.get("name")]
        except Exception:
            names = []
        return names or ["default"]

    @property
    def VOICE_PRESETS(self) -> dict:
        return {v: {"type": "voice", "name": v} for v in self.voices}

    @property
    def voice(self) -> str:
        return self._vv

    def _pick_default_voice(self) -> str:
        for cand in ("default", "Default"):
            if cand in self.voices:
                return cand
        return (self.voices or ["default"])[0]

    def set_voice(self, name: str):
        if not name:
            return
        if name in self.voices:
            self._vv = name
        else:
            logger.warning(f"Audio8-TTS: unknown voice {name!r}, keeping {self._vv} (have {self.voices})")

    # ------------------------------------------------------------- synthesis
    def _gen_args(self, text: str, voice: str | None) -> dict:
        txt = plain_text(text)
        return dict(text=txt, voice=voice or self._vv,
                    max_new_tokens=budget_frames(txt),
                    temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, seed=self.seed)

    async def synthesize_streaming(self, text: str, chunk_frames: int | None = None,
                                   voice: str | None = None) -> AsyncGenerator[np.ndarray, None]:
        """Yield int16 PCM chunks. Generation runs in a worker thread (ONNX is blocking).

        The worker gets a `stop_event`: when the consumer abandons this generator
        (barge-in, disconnect) the AR loop exits at its next frame instead of burning
        20 CPU cores on audio nobody will hear.
        """
        txt = plain_text(text)
        if not txt:
            return
        kwargs = self._gen_args(text, voice)
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        t_start = __import__("time").time()
        first_at: list[float] = []

        def worker():
            n_chunks = n_frames = 0
            try:
                for ev in self.rt.stream(chunk_frames=int(chunk_frames or self.chunk_frames), **kwargs):
                    if stop.is_set():
                        break
                    if ev.get("type") != "audio_chunk":
                        continue
                    n_chunks += 1
                    n_frames = int(ev.get("frame_count") or n_frames)
                    loop.call_soon_threadsafe(out.put_nowait, ("chunk", ev["audio"], n_frames))
            except Exception as e:                                   # pragma: no cover - runtime specific
                loop.call_soon_threadsafe(out.put_nowait, ("err", e, n_frames))
            finally:
                loop.call_soon_threadsafe(out.put_nowait, ("done", None, n_frames))

        threading.Thread(target=worker, daemon=True, name="audio8-tts").start()
        n_frames_final = 0
        try:
            while True:
                kind, payload, n_frames = await out.get()
                if kind == "err":
                    raise payload
                if kind == "done":
                    n_frames_final = n_frames
                    break
                if payload is None or len(payload) == 0:
                    continue
                if not first_at:
                    first_at.append(__import__("time").time() - t_start)
                n_frames_final = n_frames
                # int16 is this repo's wire format (pcm_to_base64 bytes it verbatim and the
                # client decodes Int16Array); the codec decoder hands us float32 in [-1, 1].
                yield (np.clip(payload, -1.0, 1.0) * 32767.0).astype(np.int16)
        finally:
            stop.set()
            secs = n_frames_final * self.frame_samples / self.sample_rate
            self.last_stats = {
                "ttfa_s": round(first_at[0], 3) if first_at else None,
                "audio_s": round(secs, 2),
                "wall_s": round(__import__("time").time() - t_start, 2),
                "rtf": round((__import__("time").time() - t_start) / secs, 3) if secs > 0 else None,
                "frames": n_frames_final, "chars": len(txt),
            }
            if first_at:
                logger.info(f"Audio8-TTS ttfa={first_at[0]:.2f}s audio={secs:.2f}s "
                            f"wall={self.last_stats['wall_s']}s rtf={self.last_stats['rtf']}")

    async def synthesize(self, text: str, reference_audio: str = None, voice: str = None,
                         chunk_frames: int | None = None) -> np.ndarray:
        # chunk_frames is exposed here because the upstream stream() re-decodes a
        # (stream_context_frames + chunk_frames)-frame window per emitted chunk: one call
        # with a big chunk is ~4x cheaper than many small ones. Streaming callers pick
        # their own latency/chunk trade-off instead.
        chunks = [c async for c in self.synthesize_streaming(text, chunk_frames=chunk_frames, voice=voice)]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)

    def stream_tts(self, *a, **k):                       # kept for pipeline compat helpers
        raise NotImplementedError("use synthesize_streaming(); app.tts_chunks owns chunk framing")
