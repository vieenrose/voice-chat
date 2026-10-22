"""Run `speech-to-speech` with Qwen3-ASR on MLX instead of Transformers/MPS.

Usage (see run.sh): python launch.py serve <speech-to-speech options>

On an 8 GB Apple Silicon Mac the pipeline must stay resident: once the weights spill to
swap, every stage reloads them and a turn goes from ~2 s to over a minute. The built-in
Qwen3-ASR handler runs through Transformers on MPS and costs ~2.7 GB for the 0.6B model;
the MLX 4-bit checkpoint costs ~1.1 GB and transcribes twice as fast.

With S2S_ASR_BACKEND=mlx (the default here) the built-in handler class is replaced before
the pipeline is built. The replacement keeps all of its turn and language logic and only
swaps model loading and inference. `--qwen3_asr_model_name` then takes an MLX checkpoint.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _install_mlx_asr() -> None:
    import speech_to_speech.STT.qwen3_asr_handler as builtin
    from speech_to_speech.utils.mlx_lock import MLXLockContext

    class MLXQwen3ASRSTTHandler(builtin.Qwen3ASRSTTHandler):
        """Built-in Qwen3-ASR handler with an mlx-audio model underneath."""

        def setup(
            self,
            model_name: str = "mlx-community/Qwen3-ASR-0.6B-4bit",
            device: str = "auto",
            torch_dtype: str = "auto",
            language: Optional[str] = "auto",
            prompt: Optional[str] = None,
            gen_kwargs: Optional[dict[str, Any]] = None,
        ) -> None:
            from mlx_audio.stt.utils import load_model

            logger.info("Loading Qwen3-ASR STT model on MLX: %s", model_name)
            self.prompt = prompt or None
            self.gen_kwargs = dict(gen_kwargs or {})
            if "max_new_tokens" in self.gen_kwargs:
                self.gen_kwargs["max_tokens"] = self.gen_kwargs.pop("max_new_tokens")
            self.configure_language(language)
            self.model = load_model(model_name)
            self.warmup()

        def _transcribe(self, audio: np.ndarray, language: Optional[str]) -> tuple[str, Optional[str]]:
            kwargs = dict(self.gen_kwargs)
            if language is not None:
                # mlx-audio matches language names ("Chinese"), not ISO codes.
                kwargs["language"] = builtin.LANGUAGE_CODE_TO_NAME.get(language, language)
            if self.prompt:
                kwargs["system_prompt"] = self.prompt
            # LLM and TTS also run on MLX; all of them share one Metal queue.
            with MLXLockContext(handler_name="Qwen3ASR-MLX"):
                result = self.model.generate(audio, verbose=False, **kwargs)
            text = result.text.strip()
            detected = None if language is not None else getattr(result, "language", None)
            return text, detected

    builtin.Qwen3ASRSTTHandler = MLXQwen3ASRSTTHandler


def _install_compact_voice_prompt() -> None:
    """Drop the framework's ~300-token English voice rules from every turn.

    Bonsai prefills at ~45 tokens/s on an M1, so those rules alone cost ~6 s per turn.
    Our session prompt already asks for short spoken zh-TW answers; tools and the
    language hint are kept when present.
    """
    from speech_to_speech.LLM import language_model
    from speech_to_speech.LLM.utils import format_language_instruction

    def build_compact_voice_system_prompt(
        session_prompt: str, *, tool_section: str = "", language_name: Optional[str] = None
    ) -> str:
        parts = [session_prompt.strip()]
        if tool_section.strip():
            parts.append(tool_section.strip())
        if language_name:
            parts.append(format_language_instruction(language_name))
        return "\n\n".join(p for p in parts if p)

    language_model.build_voice_system_prompt = build_compact_voice_system_prompt


def _install_prompt_log() -> None:
    """Log the exact chat messages sent to the LLM (debugging prompt size)."""
    from speech_to_speech.LLM import language_model

    original = language_model.LanguageModelHandler._generate

    def _generate(self, chat, *args, **kwargs):
        logger.warning("LLM prompt messages: %r", chat.to_transformers_chat())
        return original(self, chat, *args, **kwargs)

    language_model.LanguageModelHandler._generate = _generate


if os.environ.get("S2S_ASR_BACKEND", "mlx") == "mlx":
    _install_mlx_asr()
if os.environ.get("S2S_COMPACT_VOICE_PROMPT", "1") == "1":
    _install_compact_voice_prompt()
if os.environ.get("S2S_LOG_PROMPT") == "1":
    _install_prompt_log()

from speech_to_speech.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["speech-to-speech", *sys.argv[1:]]
    sys.exit(main())
