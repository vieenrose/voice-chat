"""Launch the HF speech-to-speech pipeline with Qwen-Agent as the LLM stage.

s2s selects its LLM stage through ``s2s_pipeline.get_llm_handler``. Substituting
that one function is enough to keep every other stage -- VAD, Smart Turn, STT,
TTS, the Realtime transport and the CancelScope barge-in machinery -- exactly as
upstream ships it, with no fork to rebase.

    python -m s2s.serve --mode realtime --ws_port 8765 [upstream flags...]

``--llm_backend`` is forced to a value whose argument dataclass carries the
fields we need (base_url / api_key / model_name), and cancel_scope plus
speculative_turns are injected into those kwargs by the pipeline builder, so the
handler receives them without any extra plumbing.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speech_to_speech import s2s_pipeline

from s2s.qwen_agent_handler import QwenAgentLanguageModelHandler

logger = logging.getLogger(__name__)

# Which llama-server the harness talks to. Defaults match llm_manager's Bonsai 8B.
LLM_API_BASE = os.getenv("S2S_LLM_API_BASE", "http://127.0.0.1:11435/v1")
LLM_MODEL_NAME = os.getenv("S2S_LLM_MODEL_NAME", "bonsai-8b")

_upstream_get_llm_handler = s2s_pipeline.get_llm_handler


def _get_llm_handler(
    module_kwargs,
    stop_event,
    text_prompt_queue,
    lm_response_queue,
    language_model_handler_kwargs,
    responses_api_language_model_handler_kwargs,
):
    if os.getenv("S2S_USE_UPSTREAM_LLM") == "1":
        logger.info("S2S_USE_UPSTREAM_LLM=1 -- using the stock %s stage", module_kwargs.llm_backend)
        return _upstream_get_llm_handler(
            module_kwargs,
            stop_event,
            text_prompt_queue,
            lm_response_queue,
            language_model_handler_kwargs,
            responses_api_language_model_handler_kwargs,
        )

    kw = vars(responses_api_language_model_handler_kwargs)
    logger.info("LLM stage: Qwen-Agent harness -> %s (%s)", LLM_API_BASE, LLM_MODEL_NAME)
    return QwenAgentLanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs={
            "api_base": LLM_API_BASE,
            "model_name": LLM_MODEL_NAME,
            # Injected per pipeline unit by _build_realtime_pipeline_unit.
            "cancel_scope": kw.get("cancel_scope"),
            "speculative_turns": kw.get("speculative_turns"),
        },
    )


def main() -> None:
    s2s_pipeline.get_llm_handler = _get_llm_handler
    # chat-completions' argument dataclass is the one carrying base_url/api_key,
    # and it is what the builder stamps cancel_scope onto.
    if not any(a.startswith("--llm_backend") for a in sys.argv[1:]):
        sys.argv.extend(["--llm_backend", "chat-completions"])
    s2s_pipeline.main()


if __name__ == "__main__":
    main()
