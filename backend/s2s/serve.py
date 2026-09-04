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


# OpenCode Go is the only answering provider on this branch: a hosted,
# OpenAI-compatible endpoint (https://opencode.ai/docs/go/). Fixed here rather
# than accepted from the page as an arbitrary base URL, so a request cannot aim
# the agent -- and the API key it is about to be given -- at a host of the
# caller's choosing; a caller-supplied MODEL is just a string forwarded to this
# one fixed base, which is safe because it cannot redirect the request anywhere.
OPENCODE_GO_BASE = "https://opencode.ai/zen/go/v1"
# First entry is the default (index 0, used when a POST omits "model"). Not an
# exhaustive list -- see GET /v1/llm-models for the live ~35-model catalogue --
# just what a fresh process or an offline dropdown falls back to. Both entries
# are chat/completions-family (agent.qwen_harness._RESPONSES_API_PREFIXES),
# the wire format native_loop.py has always spoken.
OPENCODE_GO_MODELS = ("mimo-v2.5", "mimo-v2.5-pro")

# The live LLM stage, so /v1/llm-config can repoint it without a restart. Only
# the reachability probe cares (see LingStreaming._reachable): the tool-calling
# turn itself always reads LLM_API_BASE/LLM_MODEL_ID/LLM_API_KEY fresh from the
# environment (agent/qwen_harness.py._endpoint()), so setting those three env
# vars alone is already enough to change which model the NEXT turn talks to.
_llm_stage: AgentLanguageModelHandler | None = None


# GET /v1/llm-models cache: (fetched_at, [model ids]). Module-level and shared
# across requests/tests on purpose -- a 35-entry list from a listing endpoint
# that needs no auth changes rarely, so there is no reason to refetch on every
# page load, and NOT fetching it inside GET /v1/llm-config keeps that route
# offline-safe (it always answers from OPENCODE_GO_MODELS, no network) for
# tests and for a page that has not asked for the live catalogue yet.
_MODEL_CACHE: list = [0.0, []]
_MODEL_CACHE_TTL = 900.0   # seconds


def _install_llm_config_route(app) -> None:
    """GET/POST /v1/llm-config: the OpenCode Go API key and model, from the UI.
    GET /v1/llm-models: the live catalogue, for the model dropdown.

    The browser client has no backend of its own, and the LLM stage runs
    server-side, so a key pasted into the UI has to be handed over somewhere.
    This is that seam.

    The key is held in this process only (os.environ, for the harness to read)
    and is never logged, never written to disk, and never returned -- GET
    reports a masked fingerprint so the UI can show whether one is set.
    """
    from fastapi import Body, HTTPException

    def _state() -> dict:
        key = os.getenv("LLM_API_KEY", "")
        key_set = bool(key and key != "none")
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
            "provider": "opencode-go",
            "model": os.getenv("LLM_MODEL_ID", LLM_MODEL_NAME),
            "models": list(OPENCODE_GO_MODELS),
            # Enough to tell two keys apart, not enough to use one.
            "key_set": key_set,
            "key_hint": (key[:4] + "…" + key[-4:]) if key_set and len(key) > 9 else "",
            "tools": tools,
        }

    @app.get("/v1/llm-config")
    def get_llm_config() -> dict:
        return _state()

    @app.get("/v1/llm-models")
    def list_llm_models() -> dict:
        """OpenCode Go's live model catalogue, filtered to what actually works
        here, for the dropdown.

        Proxied rather than fetched by the browser so the page needs no
        third-party origin, and cached because the list changes rarely. On any
        failure (no network, endpoint down) this falls back to the small fixed
        OPENCODE_GO_MODELS list rather than erroring, so the dropdown is never
        empty -- a stale-but-plausible list beats a broken page.

        Filtered to model ids native_loop.py can actually speak to: OpenCode Go
        serves some families over /v1/responses (agent.qwen_harness
        ._RESPONSES_API_PREFIXES, now supported) or /v1/messages, Anthropic-style
        (MiniMax, Qwen3.x -- not supported by either wire format this project
        has). Picking an unsupported model is not a clean 400 at selection time
        -- the gateway accepts the key and the model, then it fails at
        generation with a bare "抱歉，模型供應者暫時故障". Confirmed live with
        muse-spark-1.3-contributor before Responses API support existed here.
        Unlisting the remainder beats a dropdown item that silently breaks
        every turn.
        """
        import time

        import httpx

        # Both wire formats native_loop.py speaks are supported now (chat/completions
        # and Responses API), so only the third family -- Anthropic Messages-shaped
        # (MiniMax, Qwen3.x) -- is excluded. Named rather than enumerating the
        # supported prefixes, so a new chat/completions- or Responses-family model
        # added to the catalogue tomorrow shows up without this filter needing an edit.
        _messages_shaped = ("minimax-", "qwen3.")

        fetched_at, models = _MODEL_CACHE
        if not models or (time.time() - fetched_at) >= _MODEL_CACHE_TTL:
            try:
                r = httpx.get(f"{OPENCODE_GO_BASE}/models", timeout=10.0)
                r.raise_for_status()
                ids = [m["id"] for m in (r.json().get("data") or []) if m.get("id")]
                models = sorted(m for m in ids if not m.startswith(_messages_shaped))
                _MODEL_CACHE[:] = [time.time(), models]
            except Exception as e:
                logger.warning("could not fetch OpenCode Go's model list: %s", e)
                models = models or list(OPENCODE_GO_MODELS)
        return {"models": models}

    @app.post("/v1/llm-config")
    def set_llm_config(body: dict = Body(...)) -> dict:
        key = str(body.get("api_key") or "").strip()
        if not key:
            raise HTTPException(status_code=400, detail="OpenCode Go needs an API key")
        model = str(body.get("model") or "").strip() or OPENCODE_GO_MODELS[0]
        # Not checked against the catalogue: a caller-supplied model is just a
        # string forwarded to this one FIXED base URL (OPENCODE_GO_BASE above),
        # so it cannot redirect the request anywhere -- an unrecognised id
        # simply gets a 404 from OpenCode Go itself on the next turn, the same
        # as picking a real model that gets discontinued between the dropdown
        # loading and the click.
        os.environ["LLM_API_BASE"] = OPENCODE_GO_BASE
        os.environ["LLM_MODEL_ID"] = model
        os.environ["LLM_API_KEY"] = key
        if _llm_stage is not None:
            _llm_stage.reconfigure(OPENCODE_GO_BASE, model, key)
        logger.info("LLM stage -> OpenCode Go (%s), key_set=True", model)
        return _state()


def _install_tool_trace_route(app) -> None:
    """GET /v1/tool-trace: what the tools returned this session, newest last.
    GET /v1/turn-trace: reasoning + tool calls + token usage for the turn IN
    PROGRESS, for the UI's live "thinking" panel -- see s2s/turn_trace.py."""

    @app.get("/v1/tool-trace")
    def get_tool_trace() -> dict:
        from s2s.tool_trace import snapshot

        return {"trace": snapshot()}

    @app.get("/v1/turn-trace")
    def get_turn_trace() -> dict:
        from s2s.turn_trace import snapshot

        return snapshot()


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


def _give_stage_the_event_queue() -> None:
    """Let our LLM stage publish the transcript it produces.

    With --stt none there is no STT stage, so nothing produces the user transcript
    the UI shows. Our stage produces one itself now -- see AgentLanguageModelHandler
    ._transcribe(), which asks Gemma 4 E4B to transcribe the turn's audio before
    handing the text to the (cloud) answering model -- but publishing it needs
    `text_output_queue`, which the builder passes only to LMOutputProcessor.
    Wrapping the builder to hand the same queue to our stage is smaller than
    threading it through get_llm_handler, whose signature is upstream's.

    An earlier version of this ran a SEPARATE model on a side channel purely to
    caption the screen live, because the answering model heard the audio directly
    and nothing else needed a transcript at all. Now that E4B's transcript is
    itself the thing the answering model reasons over, it is already the most
    accurate text available -- captioning from a second, independent guess would
    be strictly worse, so the stage simply publishes what it already produced.
    """
    upstream = s2s_pipeline._build_pipeline_handlers

    def build(*args, **kwargs):
        handlers = upstream(*args, **kwargs)
        q = kwargs.get("text_output_queue")
        for h in handlers:
            if isinstance(h, AgentLanguageModelHandler):
                h.text_output_queue = q
        return handlers

    s2s_pipeline._build_pipeline_handlers = build


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
    global _llm_stage
    _llm_stage = AgentLanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs={
            "api_base": LLM_API_BASE,
            "model_name": LLM_MODEL_NAME,
            "api_key": os.getenv("LLM_API_KEY", "none"),
            # Injected per pipeline unit by _build_realtime_pipeline_unit.
            "cancel_scope": kw.get("cancel_scope"),
            "speculative_turns": kw.get("speculative_turns"),
        },
    )
    return _llm_stage


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
