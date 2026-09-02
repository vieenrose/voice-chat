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

def _neutralize_mps_empty_cache() -> None:
    """Work around an upstream bug that breaks Chinese voice input on non-Mac.

    speech_to_speech 0.2.12's ParaformerSTTHandler.process() calls
    torch.mps.empty_cache() unconditionally (STT/paraformer_handler.py:59). Every
    other call site in the package guards it with `if self.device == "mps"`; this
    one does not. On CUDA/Linux it raises

        RuntimeError: Cannot execute emptyCache() without MPS backend

    *after* a successful transcription, so the STT stage dies on every utterance
    and no voice turn ever reaches the LLM. Paraformer is the only Chinese STT
    backend, so zh voice input cannot work on Linux without this.

    Where MPS is unavailable the call can never do anything but raise, so making
    it a no-op is what it should have been. Reported upstream; delete this when a
    release carries the guard.
    """
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return
    try:
        torch.mps.empty_cache = lambda *a, **k: None
        torch.mps.synchronize = lambda *a, **k: None
    except Exception:
        logger.debug("could not neutralize torch.mps helpers", exc_info=True)


def _install_vram_route() -> None:
    """Expose GPU memory on the Realtime server, for the UI's VRAM readout.

    The browser client speaks only the Realtime protocol and has no backend of its
    own, so a host-level number has to come from the one server it already talks to.
    Wrapping create_app() adds the route without forking the framework.

    Two details:
      * torch.cuda.mem_get_info() reports the DRIVER's free/total, so it counts every
        process on the card -- llama-server included -- which is what "VRAM used"
        should mean here. torch.cuda.memory_allocated() would only see this process.
      * CORS has to be added too. The framework ships none, which is fine for
        WebSockets (not subject to CORS) but would block a cross-origin fetch, and
        the page is served from a different origin than :8765 whenever it is not
        opened on this box.
    """
    from speech_to_speech.api.openai_realtime import websocket_router as _wsr

    _upstream_create_app = _wsr.create_app

    def create_app(*args, **kwargs):
        app = _upstream_create_app(*args, **kwargs)
        try:
            from fastapi.middleware.cors import CORSMiddleware

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],      # read-only, no credentials, no secrets
                allow_methods=["GET"],
                allow_headers=["*"],
            )

            @app.get("/v1/vram")
            def vram() -> dict:
                try:
                    import torch

                    if not torch.cuda.is_available():
                        return {"available": False, "reason": "no CUDA device"}
                    free, total = torch.cuda.mem_get_info()
                    return {
                        "available": True,
                        "device": torch.cuda.get_device_name(0),
                        "total_mib": total // (1 << 20),
                        "used_mib": (total - free) // (1 << 20),
                        "free_mib": free // (1 << 20),
                    }
                except Exception as e:  # never let a readout break the server
                    return {"available": False, "reason": f"{type(e).__name__}: {e}"}

            logger.info("GET /v1/vram registered")
        except Exception:
            logger.exception("could not register /v1/vram (continuing without it)")
        return app

    _wsr.create_app = create_app


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
    _neutralize_mps_empty_cache()
    _install_vram_route()
    s2s_pipeline.get_llm_handler = _get_llm_handler
    # chat-completions' argument dataclass is the one carrying base_url/api_key,
    # and it is what the builder stamps cancel_scope onto.
    if not any(a.startswith("--llm_backend") for a in sys.argv[1:]):
        sys.argv.extend(["--llm_backend", "chat-completions"])
    s2s_pipeline.main()


if __name__ == "__main__":
    main()
