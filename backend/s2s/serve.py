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

from s2s.agent_handler import AgentLanguageModelHandler

logger = logging.getLogger(__name__)

# Which endpoint the LLM stage talks to. LLM_API_BASE / LLM_MODEL_ID are what the
# harness itself reads, so they are the primary names here too -- one source of
# truth, rather than an S2S_* copy that can disagree with the agent's own config.
# The S2S_* names remain accepted for compatibility.
LLM_API_BASE = os.getenv("LLM_API_BASE") or os.getenv("S2S_LLM_API_BASE", "http://127.0.0.1:11435/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_ID") or os.getenv("S2S_LLM_MODEL_NAME", "qwen3.5-9b")

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


# Providers the UI may select. Keeping this a fixed map rather than accepting an
# arbitrary base URL means a request from the page cannot aim the agent (and its
# API key) at a host of the caller's choosing.
def _install_llm_config_route(app) -> None:
    """GET /v1/llm-config: which model is serving, for the UI's pipeline card.

    Read-only. The endpoint is fixed at startup (LLM_API_BASE / LLM_MODEL_ID) and
    is always local, so there is nothing for the page to set -- it only needs to
    know what to display, because a hardcoded label went stale the moment the
    model changed.
    """

    @app.get("/v1/llm-config")
    def get_llm_config() -> dict:
        # The tool list is reported, not hardcoded in the page: the card read
        # "3 工具" for a while after a fourth was added, which is the same way the
        # model label went stale before it was derived from here.
        try:
            from agent.qwen_harness import _tools
            tools = sorted(_tools())
        except Exception:
            logger.exception("could not read the tool list")
            tools = []
        return {
            "model": os.getenv("LLM_MODEL_ID", LLM_MODEL_NAME),
            "api_base": os.getenv("LLM_API_BASE", LLM_API_BASE),
            "tools": tools,
        }


def _install_tool_trace_route(app) -> None:
    """GET /v1/tool-trace: what the tools returned this session, newest last."""

    @app.get("/v1/tool-trace")
    def get_tool_trace() -> dict:
        from s2s.tool_trace import snapshot

        return {"trace": snapshot()}


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
                allow_origins=["*"],
                # POST and OPTIONS are required, not optional: /v1/llm-config is a
                # POST carrying JSON, which makes it a non-simple request, so the
                # browser sends a preflight first. With GET alone the preflight was
                # rejected -- "OPTIONS /v1/llm-config 400" in the log -- and the page
                # reported only "Failed to fetch".
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
                # No credentials: the page sends the API key in the body, never a
                # cookie, so there is nothing for a third-party origin to ride on.
                allow_credentials=False,
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

            _install_llm_config_route(app)
            _install_tool_trace_route(app)
            logger.info("GET /v1/vram and GET|POST /v1/llm-config registered")
        except Exception:
            logger.exception("could not register /v1/vram (continuing without it)")
        return app

    _wsr.create_app = create_app


_caption_stream = None


def _tee_audio_to_captions() -> None:
    """Copy inbound audio to the caption stream, without touching the audio path.

    AudioHandler.append_pcm is the single funnel every inbound chunk passes
    through, for both the WebSocket and WebRTC transports, so teeing here inserts
    no handler and adds no queue hop. The tee is one non-blocking put per chunk;
    the decoding happens on the caption thread.

    Deliberately not a pipeline stage: a stage would sit in the chain and its cost,
    however small, would be on the turn.
    """
    from speech_to_speech.api.openai_realtime.handlers.audio import AudioHandler

    upstream = AudioHandler.append_pcm

    def append_pcm(self, conn_id, pcm_bytes, src_rate):
        if _caption_stream is not None and pcm_bytes:
            _caption_stream.feed(pcm_bytes, src_rate)
        return upstream(self, conn_id, pcm_bytes, src_rate)

    AudioHandler.append_pcm = append_pcm
    logger.info("inbound audio teed to the caption stream")


def _give_stage_the_event_queue() -> None:
    """Let our LLM stage publish transcription events.

    With --stt none there is no STT stage, so nothing produces the user transcript
    the UI shows. Our stage can produce one -- it has the audio -- but publishing it
    needs `text_output_queue`, which the builder passes only to LMOutputProcessor.
    Wrapping the builder to hand the same queue to our stage is smaller than
    threading it through get_llm_handler, whose signature is upstream's.
    """
    upstream = s2s_pipeline._build_pipeline_handlers

    def build(*args, **kwargs):
        handlers = upstream(*args, **kwargs)
        q = kwargs.get("text_output_queue")
        for h in handlers:
            if isinstance(h, AgentLanguageModelHandler):
                h.text_output_queue = q
        # Captions stream from the audio tee rather than from the LLM stage, so they
        # appear while the user is still speaking instead of after the segment closes.
        global _caption_stream
        if q is not None and os.getenv("S2S_CAPTION", "1").strip().lower() not in ("0", "false", "no"):
            from s2s.caption import CaptionStream

            _caption_stream = CaptionStream(q)
            _align_captions_to_vad(handlers, _caption_stream)
        return handlers

    s2s_pipeline._build_pipeline_handlers = build


def _align_captions_to_vad(handlers, caption) -> None:
    """Drive the caption's utterance boundaries from the pipeline's own VAD.

    The alternative -- X-ASR's endpoint detection, or an idle timer -- drifts from
    the pipeline, so a caption ends up describing a different span of audio than the
    answer does. Silero plus Smart Turn already decide where a turn begins and ends,
    and the VAD handler announces both on text_output_queue.

    So the queue is proxied rather than the handler subclassed: the VAD publishes
    those events from six or more places, and a proxy catches all of them without
    reimplementing any.
    """
    from speech_to_speech.VAD.vad_handler import VADHandler

    class _Tap:
        def __init__(self, real):
            self._real = real

        def put(self, item, *a, **kw):
            kind = getattr(item, "type", None)
            if kind == "speech_started":
                caption.begin()
            elif kind == "speech_stopped":
                caption.end()
            return self._real.put(item, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

    for h in handlers:
        if isinstance(h, VADHandler) and getattr(h, "text_output_queue", None) is not None:
            h.text_output_queue = _Tap(h.text_output_queue)
            logger.info("caption boundaries aligned to the pipeline VAD")


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
    return AgentLanguageModelHandler(
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


def _speak_simplified() -> None:
    """Normalise the text the TTS reads; the transcript on screen is untouched.

    The rules live in s2s/tts_text.py -- how a string should be pronounced is a
    TTS concern, not something for the tools to encode in their output.
    """
    from s2s.tts_text import normalize

    upstream = s2s_pipeline.get_tts_handler

    def get_tts_handler(*args, **kwargs):
        handler = upstream(*args, **kwargs)
        original = handler.process

        def process(tts_input):
            text = getattr(tts_input, "text", None)
            if text:
                spoken = normalize(text)
                if spoken != text:
                    tts_input = tts_input.model_copy(update={"text": spoken})
            yield from original(tts_input)

        handler.process = process
        logger.info("TTS text normalised (glyphs + extensions); transcript unchanged")
        return handler

    s2s_pipeline.get_tts_handler = get_tts_handler


def main() -> None:
    _neutralize_mps_empty_cache()
    _give_stage_the_event_queue()
    _tee_audio_to_captions()
    _install_vram_route()
    _speak_simplified()
    logger.info("LLM endpoint: %s (%s)", LLM_API_BASE, LLM_MODEL_NAME)
    s2s_pipeline.get_llm_handler = _get_llm_handler
    # chat-completions' argument dataclass is the one carrying base_url/api_key,
    # and it is what the builder stamps cancel_scope onto.
    if not any(a.startswith("--llm_backend") for a in sys.argv[1:]):
        sys.argv.extend(["--llm_backend", "chat-completions"])
    s2s_pipeline.main()


if __name__ == "__main__":
    main()
