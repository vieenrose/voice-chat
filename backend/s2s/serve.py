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


# The live LLM stage, so /v1/llm-config can repoint it without a restart.
_llm_stage = None

# Providers the UI may select. Keeping this a fixed map rather than accepting an
# arbitrary base URL means a request from the page cannot aim the agent (and its
# API key) at a host of the caller's choosing.
PROVIDERS = {
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",      # the auto-router; any catalogue id may replace it
        "needs_key": True,
        "label": "OpenRouter",
        "catalogue": "https://openrouter.ai/api/v1/models",
    },
    # Kept as the offline fallback: it is the only option that works with no key and
    # no network. base/model None means "whatever this process was started with" --
    # see _provider_local(), so it follows the CLI rather than a copy of it.
    "local": {"base": None, "model": None, "needs_key": False, "label": "本機 llama-server"},
}


def _startup_key() -> str:
    """An OpenRouter key from the environment, if the operator supplied one.

    OPENROUTER_API_KEY is the conventional name; LLM_API_KEY is what the harness
    itself reads. Either starts the demo on OpenRouter instead of the local server.
    """
    return (os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()


def _install_llm_config_route(app) -> None:
    """GET/POST /v1/llm-config: let the page choose the LLM endpoint.

    The browser client has no backend of its own, and the LLM stage runs
    server-side, so a key pasted into the UI has to be handed over somewhere. This
    is that seam.

    The key is held in this process only (os.environ for the harness to read) and
    is never logged, never written to disk, and never returned -- GET reports a
    masked fingerprint so the UI can show whether a key is set.
    """
    import time

    from fastapi import Body, HTTPException

    def _state() -> dict:
        key = os.getenv("LLM_API_KEY", "")
        base = os.getenv("LLM_API_BASE", LLM_API_BASE)
        provider = next((n for n, p in PROVIDERS.items()
                         if p["base"] and p["base"] == base), "local")
        return {
            "provider": provider,
            "model": os.getenv("LLM_MODEL_ID", LLM_MODEL_NAME),
            # Enough to tell two keys apart, not enough to use one.
            "key_set": bool(key and key != "none"),
            "key_hint": (key[:7] + "…" + key[-4:]) if key and key != "none" else "",
            "providers": {n: {"label": p["label"], "needs_key": p["needs_key"]}
                          for n, p in PROVIDERS.items()},
        }

    _catalogue_cache: dict = {}

    @app.get("/v1/llm-models")
    def list_llm_models(provider: str = "openrouter") -> dict:
        """The provider's text models, for the UI's dropdown.

        Proxied rather than fetched by the browser so the page needs no third-party
        origin, and cached because it is a 400-entry list that changes rarely.

        Only models that are text-capable in and out, and can call tools, are returned. This demo's agent calls three
        tools, and a model that cannot call them is never a valid choice here -- it
        answers weather and news questions from its weights instead, which is the
        fabrication failure this project spends most of its effort avoiding. 66 of
        the 421 text models are filtered out on that basis.
        """
        spec = PROVIDERS.get(provider)
        if spec is None or not spec.get("catalogue"):
            raise HTTPException(status_code=400, detail=f"no catalogue for {provider!r}")
        cached = _catalogue_cache.get(provider)
        if cached and (time.time() - cached[0]) < 900:
            return cached[1]
        try:
            import httpx

            r = httpx.get(spec["catalogue"], timeout=15.0)
            r.raise_for_status()
            raw = r.json().get("data") or []
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e

        out = []
        for m in raw:
            arch = m.get("architecture") or {}
            # Text in AND text out. "Capable of", not "exclusively": a model that also
            # accepts images still takes text, and requiring text-ONLY input would drop
            # Claude Opus 5, GPT-5.6, Gemini 3.7 and Grok 4.6 -- 353 candidates down to
            # 116 -- for no benefit to a voice pipeline that only ever sends text.
            if "text" not in (arch.get("input_modalities") or []):
                continue
            if "text" not in (arch.get("output_modalities") or []):
                continue
            params = m.get("supported_parameters") or []
            if "tools" not in params:
                continue
            price = str((m.get("pricing") or {}).get("prompt", "1"))
            out.append({
                "id": m.get("id"),
                "name": m.get("name") or m.get("id"),
                "context": m.get("context_length") or 0,
                "tools": "tools" in params,
                "free": price in ("0", "0.0", "0.00"),
            })
        out.sort(key=lambda x: (not x["free"], x["id"]))
        payload = {"provider": provider, "count": len(out), "models": out}
        _catalogue_cache[provider] = (time.time(), payload)
        return payload

    @app.get("/v1/llm-config")
    def get_llm_config() -> dict:
        return _state()

    @app.post("/v1/llm-config")
    def set_llm_config(body: dict = Body(...)) -> dict:
        name = str(body.get("provider") or "local")
        spec = PROVIDERS.get(name)
        if spec is None:
            raise HTTPException(status_code=400, detail=f"unknown provider {name!r}")
        key = str(body.get("api_key") or "").strip()
        if spec["needs_key"] and not key:
            raise HTTPException(status_code=400, detail=f"{spec['label']} needs an API key")

        base = spec["base"] or _provider_local()["base"]
        model = spec["model"] or _provider_local()["model"]
        # A caller-supplied model is just a string forwarded to a FIXED base URL, so
        # it cannot redirect the request anywhere; only providers with a catalogue
        # accept one, and the local server's model comes from its own registry.
        want = str(body.get("model") or "").strip()
        if want and spec.get("catalogue"):
            model = want
        os.environ["LLM_API_BASE"] = base
        os.environ["LLM_MODEL_ID"] = model
        os.environ["LLM_API_KEY"] = key or "none"

        # The Assistant bakes model and key in at construction, so it must be dropped.
        try:
            from agent.qwen_harness import reset_agent

            reset_agent()
        except Exception:
            logger.exception("could not reset the agent after a config change")
        if _llm_stage is not None:
            _llm_stage.reconfigure(base, model)
        logger.info("LLM stage -> %s (%s), key_set=%s", base, model, bool(key))
        return _state()


def _provider_local() -> dict:
    """The local endpoint as configured at startup."""
    return {"base": LLM_API_BASE, "model": LLM_MODEL_NAME}


def _fix_tool_call_id() -> None:
    """Work around a qwen-agent bug that breaks tool calling on strict providers.

    Converting its internal messages to OpenAI form, qwen-agent labels a tool result
    with `id` (llm/base.py:446):

        new_msg['role'] = 'tool'
        new_msg['id'] = msg.get('extra', {}).get('function_id', '1')

    but the OpenAI schema requires `tool_call_id` on a tool message. llama-server
    accepts either, so this is invisible locally; strict providers reject the whole
    request. Observed through OpenRouter:

        Cohere: invalid tool message at messages[3]: tool_call_id is a required field
        Nvidia: missing field `tool_call_id`

    Because the failure lands on the follow-up call -- after the tool has already
    run -- a search or weather turn would call its tool, then die instead of
    answering. Single-step turns were unaffected, which is why it looked like
    flaky model quality rather than a protocol error.
    """
    try:
        from qwen_agent.llm.base import BaseChatModel

        original = BaseChatModel._conv_qwen_agent_messages_to_oai

        def patched(messages):
            out = original(messages)
            for m in out:
                if isinstance(m, dict) and m.get("role") == "tool" and "tool_call_id" not in m:
                    m["tool_call_id"] = m.get("id") or "1"
            return out

        BaseChatModel._conv_qwen_agent_messages_to_oai = staticmethod(patched)
        logger.info("patched qwen-agent tool messages to carry tool_call_id")
    except Exception:
        logger.exception("could not patch tool_call_id (tool turns may fail on strict providers)")


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

            _install_llm_config_route(app)
            logger.info("GET /v1/vram and GET|POST /v1/llm-config registered")
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
    global _llm_stage
    _llm_stage = QwenAgentLanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs={
            "api_base": os.getenv("LLM_API_BASE", LLM_API_BASE),
            "model_name": os.getenv("LLM_MODEL_ID", LLM_MODEL_NAME),
            # Injected per pipeline unit by _build_realtime_pipeline_unit.
            "cancel_scope": kw.get("cancel_scope"),
            "speculative_turns": kw.get("speculative_turns"),
        },
    )
    return _llm_stage


def main() -> None:
    _neutralize_mps_empty_cache()
    _fix_tool_call_id()
    _install_vram_route()
    # A key in the environment means start on OpenRouter; without one the demo would
    # have no working model at all, so it falls back to the local server and the UI
    # asks for a key.
    key = _startup_key()
    if key:
        spec = PROVIDERS["openrouter"]
        os.environ.setdefault("LLM_API_BASE", spec["base"])
        os.environ.setdefault("LLM_MODEL_ID", spec["model"])
        os.environ["LLM_API_KEY"] = key
        logger.info("starting on %s (%s) -- key supplied by the environment",
                    spec["label"], os.environ["LLM_MODEL_ID"])
    else:
        logger.info("no OPENROUTER_API_KEY -- starting on the local server; "
                    "the UI can switch once a key is pasted")
    s2s_pipeline.get_llm_handler = _get_llm_handler
    # chat-completions' argument dataclass is the one carrying base_url/api_key,
    # and it is what the builder stamps cancel_scope onto.
    if not any(a.startswith("--llm_backend") for a in sys.argv[1:]):
        sys.argv.extend(["--llm_backend", "chat-completions"])
    s2s_pipeline.main()


if __name__ == "__main__":
    main()
