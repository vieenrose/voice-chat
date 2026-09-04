"""Qwen3-ASR as the STT stage: one blocking transcription per utterance, via
transformers directly rather than through the framework's own STT stage.

huggingface/speech-to-speech added a stock ``--stt qwen3-asr`` backend for this
exact model (commit db9b7f9, "Add Qwen3-ASR STT backend"), but it lives on
``main``, not in the 0.2.12 release this project is pinned to -- and ``main``
has since removed ``get_llm_handler`` / ``get_tts_handler`` /
``_build_pipeline_handlers``, the exact hooks ``s2s/serve.py`` depends on to
keep this project's tool loop and TTS text normalisation (the README's OmniVoice
note is the same problem, for the same reason: adopting a `main`-only feature
means porting to whatever replaced those hooks, not changing one CLI flag).

So this calls the same underlying transformers APIs the upstream handler uses
(``AutoProcessor.apply_transcription_request``, ``generate``, ``decode``)
directly, from this project's own STT step (``s2s/agent_handler.py
._transcribe`` -> here) instead of through a pipeline stage. Same model, same
call shape, none of the upstream plumbing.

Replaces Gemma 4 E4B doing STT (``s2s/stt_gemma.py``, now unused): faster (no
reasoning pass to work around -- a dedicated transcription model has nothing to
narrate), no JSON-schema workaround needed (there is no prose to constrain
against), and measured more accurate on this project's own test clips: heard
陳怡君 correctly where E4B's chat-completions route misheard it as 陳一軍.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("STT_MODEL", "Qwen/Qwen3-ASR-0.6B-hf")
_DEVICE = os.getenv("STT_DEVICE", "cuda")
# None = detect per utterance. A forced language would be faster and this demo
# is zh-TW throughout, but the extension-lookup eval also tests English
# nicknames (README: "test also search by english nickname"), and auto-detect
# measured no meaningful latency cost (0.1-0.7 s either way on short clips).
_LANGUAGE = os.getenv("STT_LANGUAGE") or None

_lock = threading.Lock()
_processor = None
_model = None


def _load() -> None:
    global _processor, _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:      # another thread won the race while this one waited
            return
        import time

        import torch
        import transformers

        logger.info("loading Qwen3-ASR STT model: %s", _MODEL_NAME)
        t0 = time.monotonic()
        dtype = torch.bfloat16 if _DEVICE.startswith("cuda") else torch.float32

        def _cached_or_download(cls, **kw):
            # from_pretrained's default revalidates the Hub on every call (a HEAD
            # per config/tokenizer/weights file) even when the exact revision is
            # already cached -- measured at ~4s of sequential round-trips before
            # this fix, once per process start. The weights are pinned to a
            # commit hash in the cache either way, so a stale local copy is not a
            # real risk here; try local-only first and only reach the network if
            # this is genuinely the first run on this machine.
            try:
                return cls.from_pretrained(_MODEL_NAME, local_files_only=True, **kw)
            except Exception:
                logger.info("Qwen3-ASR not cached locally; downloading %s", _MODEL_NAME)
                return cls.from_pretrained(_MODEL_NAME, **kw)

        processor = _cached_or_download(transformers.AutoProcessor)
        model = (_cached_or_download(transformers.AutoModelForMultimodalLM, dtype=dtype)
                 .to(_DEVICE).eval())
        _processor, _model = processor, model
        logger.info("Qwen3-ASR ready in %.1fs", time.monotonic() - t0)


def transcribe(pcm16: bytes, sample_rate: int = 16000) -> str:
    """Transcribe one complete utterance. Returns "" on any failure.

    Raises nothing: a load failure or a bad generation must not crash the
    turn, it should just produce no transcript, the same way an empty typed
    message would.
    """
    import time
    t0 = time.monotonic()
    try:
        _load()
        import torch

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            g = np.gcd(16000, sample_rate)
            audio = resample_poly(audio, 16000 // g, sample_rate // g).astype(np.float32)

        dtype = torch.bfloat16 if _DEVICE.startswith("cuda") else torch.float32
        inputs = _processor.apply_transcription_request(audio=audio, language=_LANGUAGE)
        inputs = inputs.to(_DEVICE, dtype)
        with torch.inference_mode():
            output_ids = _model.generate(**inputs, max_new_tokens=128)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = _processor.decode(generated, return_format="transcription_only")[0].strip()
        logger.info("Qwen3-ASR: %.2fs for %.2fs of audio -> %r",
                    time.monotonic() - t0, len(audio) / 16000, text)
        return text
    except Exception:
        logger.exception("Qwen3-ASR STT call failed after %.2fs", time.monotonic() - t0)
        return ""
