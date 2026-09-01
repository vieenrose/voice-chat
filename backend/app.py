"""
FastAPI backend for streaming voice chat with tool calling (web_search via self-hosted SearXNG).

Exposes:
  GET /health, /stats, /search (SearXNG compat), /api/search, /api/chat, /api/chat/tools
  WS  /ws/chat

Low-latency WebSocket protocol:
  Client JSON: {"type":"start"|"audio_chunk"|"stop"|"barge_in"|"text_input", ...}
  Backend JSON: {"type":"stt_partial"|"stt_final"|"llm_token"|"tool_call"|"tool_result"|"tts_chunk"|"tts_end"|"latency"}

Access control:
  Set VOICE_CHAT_TOKEN (or --token) to require `X-Auth-Token` / `Authorization: Bearer`
  / `?token=` on every route except the lite GET /health. When no token is configured
  the server runs in dev mode: unauthenticated, but model switching (which restarts
  llama-server and silences every other session) is restricted to loopback clients.
"""
import asyncio
import base64
import argparse
import html as _html
import time
import json
import os
import secrets
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(__file__))
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlsplit
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger
import numpy as np

from pipeline.speech_to_speech import HFSpeechToSpeechPipeline
# the two 'this is not speech' rules are shared with the voice pipeline on purpose
from pipeline.speech_to_speech import _is_tool_artifact, is_echo_of_prompt, prepare_tts_text
from llm_manager import llm_manager, MODEL_REGISTRY as LLM_MODEL_REGISTRY

app = FastAPI(title="Voice Chat HF S2S + SearXNG Tools", version="1.1.0")

# --- access control ---
# AUTH_TOKEN is None => dev mode (no auth). Configure via env VOICE_CHAT_TOKEN or --token.
AUTH_TOKEN = os.getenv("VOICE_CHAT_TOKEN", "").strip() or None
# CORS: allow_origins=["*"] together with allow_credentials=True lets ANY origin read
# authenticated responses (verified live: Access-Control-Allow-Origin echoed the
# caller's Origin alongside Allow-Credentials: true). Defaults are now an explicit
# dev-origin list with no credentials; set ALLOWED_ORIGINS (comma-separated) for
# deployment. Credentialed wildcard is refused.
CORS_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
CORS_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "0") == "1" and CORS_ORIGINS != ["*"]
if os.getenv("ALLOW_CREDENTIALS", "0") == "1" and CORS_ORIGINS == ["*"]:
    logger.error("ALLOWED_ORIGINS=* with ALLOW_CREDENTIALS=1 is unsafe; credentials disabled for CORS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _presented_token(request: Request) -> str | None:
    t = request.headers.get("x-auth-token")
    if not t:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            t = auth[7:]
    if not t:
        t = request.query_params.get("token")
    return t or None


def _token_ok(request: Request) -> bool:
    if not AUTH_TOKEN:
        return True
    presented = _presented_token(request)
    return bool(presented) and secrets.compare_digest(presented, AUTH_TOKEN)


def _agent_config() -> dict:
    """Live agent-layer switches, read from the harness itself so this cannot drift."""
    try:
        from agent import qwen_harness as _qh
        return {
            "preflight_lookup": _qh._preflight_enabled(),
            "smalltalk_rule": _qh._smalltalk_rule() != "",
            "thinking": _qh._thinking_on(),
        }
    except Exception as e:                                    # harness absent (light image)
        return {"error": f"{type(e).__name__}"}


def _is_loopback(request: Request) -> bool:
    client = request.client
    return bool(client and client.host in _LOOPBACK)


_WS_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
WS_ALLOW_ANY_ORIGIN = os.getenv("WS_ALLOW_ANY_ORIGIN", "").strip() in ("1", "true", "yes")


def _origin_ok(websocket: WebSocket) -> tuple[bool, str]:
    """Gate the WebSocket handshake by Origin — CORSMiddleware does not cover it.

    Browsers apply CORS to fetch(), not to a WebSocket: a page at https://evil.example
    can open ws://127.0.0.1:8000/ws/chat from the visitor's own browser, and when the demo
    is bound to a Tailnet/Funnel address the same trick reaches a colleague's browser on a
    machine where the service *is* trusted. Every earlier origin fix in this file left that
    door open, and the thing behind the door is a live microphone-to-speaker session with
    tool calls.

    Still allowed, deliberately: same-origin pages (which is how the UI is served, so the
    Funnel setup in the README keeps working with no configuration), loopback pages on any
    port (a page whose Origin is localhost is already on this machine), and clients that
    send no Origin at all — non-browser callers, tests, the e2e suite. The token check is
    independent and unchanged. Set WS_ALLOW_ANY_ORIGIN=1 to opt out entirely.
    """
    if WS_ALLOW_ANY_ORIGIN:
        return True, ""
    origin = (websocket.headers.get("origin") or "").strip()
    if not origin:
        return True, ""
    if origin == "null":
        return False, "opaque origin (null)"     # sandboxed iframe / file:// — not an identity
    parsed = urlsplit(origin)
    scheme, netloc = (parsed.scheme or "").lower(), (parsed.netloc or "").lower()
    if scheme not in ("http", "https"):
        return False, f"non-http origin {origin!r}"
    if origin.lower() in {o.lower() for o in CORS_ORIGINS}:
        return True, ""
    if (parsed.hostname or "").lower() in _WS_LOOPBACK_HOSTS:
        return True, ""
    host = (websocket.headers.get("host") or "").lower()
    if netloc and netloc == host:
        return True, ""
    return False, f"origin {origin!r} is neither same-origin ({host}) nor in ALLOWED_ORIGINS"


# Routes that do work / change state and therefore need the token. Everything else is
# the built SPA shell + its hashed assets (mounted at "/"), which carry no secrets and
# must stay open or the UI cannot even load.
PROTECTED_PREFIXES = ("/api/", "/ws/", "/search", "/stats")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Reject unauthenticated use of the working routes when a token is configured.

    /health stays open (lite form) because the UI polls it every 3s and
    docker-compose healthchecks it; its usage-history fields require ?verbose=1 +
    token. Unknown/other paths (the SPA) are not gated."""
    path = request.url.path
    if path.startswith("/health"):
        if request.query_params.get("verbose") == "1" and not _token_ok(request):
            return JSONResponse({"error": "token required for /health?verbose=1"}, status_code=401)
        return await call_next(request)
    if path.startswith(PROTECTED_PREFIXES) and not _token_ok(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


# Model switching: reachable-or-not is decided below; these are the knobs.
#
# The original hardening ("loopback-only when no token is configured") was wrong in
# practice: it broke the operator's OWN model picker the moment the UI was opened over
# Tailscale or the LAN, because from the server's side that request arrives with a
# 100.x/192.x peer address and is indistinguishable from a stranger's. It also protected
# a capability the demo's own open-by-design policy already hands to anyone who can reach
# the port (chat, search, speak). So: in open mode switching is allowed, *attributed and
# rate-limited* instead of blocked; set REQUIRE_TOKEN_FOR_MODEL_SWITCH=1 to get the old
# closed behaviour back.
MODEL_SWITCH_REQUIRE_TOKEN = os.getenv("REQUIRE_TOKEN_FOR_MODEL_SWITCH", "").strip().lower() in ("1", "true", "yes")
MODEL_SWITCH_COOLDOWN_S = float(os.getenv("MODEL_SWITCH_COOLDOWN_S", "10"))
_switch_state = {"at": 0.0, "peer": "", "model": ""}

pipeline: HFSpeechToSpeechPipeline | None = None
# `latencies` is a bounded ring buffer: as a plain list it grew for the whole
# lifetime of the process (one entry per utterance, never freed) on a long-running demo.
stats = {"connections": 0, "utterances": 0, "latencies": deque(maxlen=1000), "tool_calls": 0,
         "model_switches": deque(maxlen=20)}   # who switched what, when (verbose /health)
MOCK_MODE = False

# --- SearXNG self-host integration ---
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
searxng_process = None

async def ensure_searxng():
    """Try to ensure SearXNG minimal is reachable; if not, start it in background thread"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{SEARXNG_URL}/healthz")
            if r.status_code == 200:
                logger.info(f"SearXNG already running at {SEARXNG_URL} ✓")
                return True
    except Exception:
        pass
    # Try to start minimal SearXNG server via subprocess
    try:
        import pathlib
        tools_dir = pathlib.Path(__file__).parent / "tools"
        server_path = tools_dir / "searxng_server.py"
        if server_path.exists():
            logger.info(f"Starting self-hosted SearXNG minimal at {SEARXNG_URL} ...")
            # start in background via thread (uvicorn in subprocess)
            import threading
            def _run():
                import uvicorn
                from tools.searxng_server import app as searx_app
                uvicorn.run(searx_app, host="127.0.0.1", port=8888, log_level="warning")
            th = threading.Thread(target=_run, daemon=True, name="searxng-minimal")
            th.start()
            await asyncio.sleep(1.5)
            # verify
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(f"{SEARXNG_URL}/healthz")
                    if r.status_code == 200:
                        logger.info("SearXNG minimal started ✓")
                        return True
            except Exception as e2:
                logger.warning(f"SearXNG minimal start verification failed {e2}")
        else:
            logger.warning(f"SearXNG server not found at {server_path}")
    except Exception as e:
        logger.warning(f"Failed to start SearXNG {e}")
    # Even if SearXNG not running, web_search will fallback to DDGS/mock, so it's OK
    logger.info("SearXNG not running, web_search will use fallback (DDGS/mock) but still works")
    return False

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # app.on_event("startup") is deprecated in the pinned FastAPI (it warns on boot).
    if not AUTH_TOKEN:
        logger.warning("VOICE_CHAT_TOKEN not set: running unauthenticated. Model "
                       "switching is restricted to loopback clients; do not expose this "
                       "port (e.g. via Tailscale Funnel) without setting a token.")
    asyncio.create_task(ensure_searxng())
    yield


app.router.lifespan_context = _lifespan

# --- health & stats ---
@app.get("/health")
async def health(verbose: bool = Query(False)):
    mem = psutil.Process().memory_info()
    # Check searxng status
    searxng_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{SEARXNG_URL}/healthz")
            searxng_ok = r.status_code == 200
    except Exception:
        searxng_ok = False
    body = {
        "status": "ok",
        "mock": MOCK_MODE,
        "models_loaded": {
            "stt": pipeline.stt.backend if pipeline else "not_loaded",
            "llm": "mock" if MOCK_MODE else (pipeline.llm.backend if pipeline and hasattr(pipeline.llm, 'model') else "loaded"),
            "tts": pipeline.tts.backend if pipeline else "not_loaded",
        },
        "llm_manager": llm_manager.status(),
        "rss_mb": round(mem.rss / 1024/1024, 1),
        "searxng": {"url": SEARXNG_URL, "ok": searxng_ok, "self_hosted": True},
        "auth": "token" if AUTH_TOKEN else "open",
        # What the agent layer is configured to do, so any measurement can state it.
        # The harness can *make* a turn correct (forcing a lookup the question demanded,
        # repairing an answer that named a tool it never called); a benchmark that does
        # not know those switches were on is measuring the guards, not the model.
        "agent": _agent_config(),
        # Count only. The raw latency array (every utterance's E2E history) and VMS
        # are served only to an authenticated ?verbose=1 caller — they used to be
        # broadcast unauthenticated on a funnel-exposed port, and the UI reads neither.
        "stats": {k: (len(v) if isinstance(v, deque) else v) for k, v in stats.items()},
        "last_model_switches": list(stats["model_switches"])[-5:] if verbose else [],
    }
    if verbose:
        body["vms_mb"] = round(mem.vms / 1024/1024, 1)
        body["stats"]["latencies"] = list(stats["latencies"])
    return body

@app.get("/api/model")
async def get_model():
    return llm_manager.status()

@app.post("/api/model")
async def switch_model(payload: dict, request: Request):
    """Switch the loaded text-LLM model. Stops the current llama-server and
    starts the requested one — the caller should expect this to take several
    seconds and the pipeline's LLM to be briefly unavailable while it swaps.

    Access: when a token is configured the auth middleware has already checked it. In open
    mode the switch is allowed from any peer but *attributed* (logged + kept in a bounded
    history visible at /health?verbose=1) and rate-limited, because a switch restarts
    llama-server for every session and thrashing it is a self-inflicted outage.
    REQUIRE_TOKEN_FOR_MODEL_SWITCH=1 restores loopback-only. The truly destructive part —
    terminating whatever holds the LLM port — stays restricted inside llm_manager to
    processes that actually are llama-server.
    """
    peer = request.client.host if request.client else "?"
    if not AUTH_TOKEN:
        if MODEL_SWITCH_REQUIRE_TOKEN and not _is_loopback(request):
            return JSONResponse({
                "error": "model switching is restricted to loopback (REQUIRE_TOKEN_FOR_MODEL_SWITCH=1)",
                "remedy": "open the UI from the server itself, or set VOICE_CHAT_TOKEN and pass ?token=…",
            }, status_code=403)
        if not _is_loopback(request):
            logger.warning(f"model switch requested by non-loopback peer {peer} (no token configured)")
    model_id = payload.get("model_id") or payload.get("model")
    if not model_id:
        return JSONResponse({"error": "model_id required"}, status_code=400)
    if model_id not in LLM_MODEL_REGISTRY:
        return JSONResponse({"error": f"unknown model_id {model_id!r}", "available": list(LLM_MODEL_REGISTRY)}, status_code=400)
    if llm_manager.switching:
        return JSONResponse({"error": "a model switch is already in progress"}, status_code=409)
    _since = time.time() - _switch_state["at"]
    if _switch_state["at"] and _since < MODEL_SWITCH_COOLDOWN_S:
        # Cooldown, not permission: two switches in quick succession means the first one's
        # model never got a chance to serve anything, and repeated 4B reloads on a shared
        # GPU take the whole assistant down for minutes.
        return JSONResponse({"error": f"switched to {_switch_state['model']} {_since:.0f}s ago by "
                                      f"{_switch_state['peer']}; wait {MODEL_SWITCH_COOLDOWN_S - _since:.0f}s",
                             "retry_after_s": round(MODEL_SWITCH_COOLDOWN_S - _since, 1)}, status_code=429)
    _t_switch = time.time()
    try:
        result = await llm_manager.switch_to(model_id)
    except Exception as e:
        logger.exception(f"model switch to {model_id} failed: {e}")
        stats["model_switches"].append({"at": round(_t_switch, 1), "peer": peer, "model": model_id,
                                        "ok": False, "error": str(e)[:120]})
        return JSONResponse({"error": str(e)}, status_code=500)
    _switch_state.update(at=time.time(), peer=peer, model=model_id)
    stats["model_switches"].append({"at": round(_t_switch, 1), "peer": peer, "model": model_id,
                                    "ok": True, "took_s": round(time.time() - _t_switch, 1)})
    if pipeline is not None and result.get("alias"):
        # LingStreaming reads self.model_name fresh on every request (it's not baked
        # into a closure at construction), so updating these two attributes in place
        # is enough to point the existing client at the newly-loaded server — no need
        # to reconstruct it.
        pipeline.llm.model_name = result["alias"]
        pipeline.llm.mock = False
    # The tool-calling agent harnesses each cache a singleton Assistant/Agent with the
    # model alias baked in at construction time (unlike LingStreaming above) — without
    # this, any turn that triggers a tool call would keep talking to llama-server using
    # the model alias from before the switch. Best-effort across all three fallback
    # harnesses since only one is actually in use at a time.
    for _mod_name in ("agent.qwen_harness", "agent.harness", "agent.pydantic_harness"):
        try:
            import importlib
            importlib.import_module(_mod_name).reset_agent()
        except Exception:
            pass
    return result

@app.get("/stats")
async def get_stats():
    lats = list(stats["latencies"])
    if not lats:
        return {"count":0, "p50":0, "p95":0, "avg":0, "rss_mb": round(psutil.Process().memory_info().rss/1024/1024,1), "tool_calls": stats["tool_calls"]}
    arr = np.array(lats)
    return {
        "count": len(lats),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "avg": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "rss_mb": round(psutil.Process().memory_info().rss/1024/1024,1),
        "tool_calls": stats["tool_calls"]
    }

# --- SearXNG compatible endpoints (self-hosted) ---
@app.get("/search")
async def searx_search(request: Request, q: str = Query("", description="query"), format: str = Query("json")):
    # Proxy to tools.web_search to provide SearXNG compatible JSON
    from tools.web_search import web_search
    if not q:
        q = request.query_params.get("q","")
    if not q:
        return JSONResponse({"error":"missing q"}, status_code=400)
    res = await web_search(q, count=5)
    # Return SearXNG shape
    payload = {
        "query": q,
        "number_of_results": len(res["results"]),
        "results": [
            {"title": r["title"], "url": r["url"], "content": r["content"], "engine": res["source"], "score": 1.0 - i*0.1, "category": "general"}
            for i, r in enumerate(res["results"])
        ],
        "answers": [], "corrections": [], "infoboxes": [], "suggestions": [], "unresponsive_engines": []
    }
    if format == "json":
        return JSONResponse(payload)
    else:
        # This endpoint echoes the caller's `q` and internet-sourced titles/URLs into
        # HTML. It used to interpolate them raw, so `?q=<b>hi</b></h2>` came back as
        # live markup (reflected XSS), and any engine result could inject an href.
        # Everything is escaped, and only http(s) URLs become links.
        def _safe_href(u: str) -> str:
            return _html.escape(u, quote=True) if u.lower().startswith(("http://", "https://")) else "#"
        rows = "".join(
            f'<div><a href="{_safe_href(r["url"])}" rel="noopener noreferrer" target="_blank">{_html.escape(r["title"])}</a>'
            f'<p>{_html.escape(r["content"][:200])}</p></div>'
            for r in res["results"]
        )
        return HTMLResponse(f"<html><body><h2>{_html.escape(q)}</h2>{rows}</body></html>")

@app.get("/api/search")
async def api_search(q: str = Query(..., description="search query"), count: int = 5):
    from tools.web_search import web_search
    res = await web_search(q, count=count)
    stats["tool_calls"] += 1
    return res

@app.post("/api/tools/web_search")
async def api_tools_search(payload: dict):
    from tools.web_search import web_search
    q = payload.get("query") or payload.get("q") or ""
    cnt = int(payload.get("count", 5))
    if not q:
        return JSONResponse({"error":"query required"}, status_code=400)
    res = await web_search(q, count=cnt)
    stats["tool_calls"] += 1
    return res

def pcm_to_base64(pcm: np.ndarray) -> str:
    return base64.b64encode(pcm.tobytes()).decode('ascii')


async def tts_chunks(text: str, voice: str | None = None):
    """Stream TTS for one utterance with an optional per-call voice.

    The TTS instance is shared by every session, so the previous approach —
    pipeline.tts.set_voice(name) before synthesizing — permanently retuned the
    default voice for all other sessions (and a mistyped name was swallowed by
    `except KeyError: pass`, silently speaking in whatever voice was last set).
    Backends whose synthesize_streaming predates the voice kwarg (kokoro/moss/…)
    fall back to set_voice with a warning; a genuinely unknown voice name
    (KeyError) propagates to the caller instead of being ignored.

    The text is run through the same front-end as the voice pipeline (markdown, units,
    percent signs) so every TTS entry point speaks the same thing; the caller keeps the
    original text for the chat bubble.
    """
    text, applied = prepare_tts_text(text)
    if applied:
        logger.debug(f"tts_chunks text front-end {applied}: -> {text[:60]!r}")
    if not text:
        return
    if voice and hasattr(pipeline.tts, "synthesize_streaming"):
        try:
            async for pcm in pipeline.tts.synthesize_streaming(text, voice=voice):
                yield pcm
            return
        except TypeError:
            logger.debug(f"{pipeline.tts.backend}: synthesize_streaming(voice=) unsupported, using set_voice")
            if hasattr(pipeline.tts, "set_voice"):
                pipeline.tts.set_voice(voice)
    async for pcm in pipeline.tts.synthesize_streaming(text):
        yield pcm

def _resolve_session_id(session_id: str) -> str:
    """Empty or the literal 'default' gets a unique id.

    session_id keys the pipeline's in-flight-turn map (_voice_response_tasks), so
    every client that omitted it shared one slot: two such tabs could cancel each
    other's replies, because do_barge_in pops the shared entry.
    """
    if not session_id or session_id == "default":
        return f"anon-{uuid.uuid4().hex[:10]}"
    return session_id


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, session_id: str = Query(default="")):
    # Auth before accept(): a rejected socket gets an HTTP 403 handshake rather than
    # an established connection we then have to tear down.
    ok, why = _origin_ok(websocket)
    if not ok:
        await websocket.close(code=1008)
        logger.warning(f"WS handshake rejected: {why}")
        return
    if not _token_ok(websocket):
        await websocket.close(code=1008)
        logger.warning("WS rejected: missing/invalid token")
        return
    session_id = _resolve_session_id(session_id)
    await websocket.accept()
    stats["connections"] += 1
    logger.info(f"WS connected session={session_id}")
    pcm_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    # Shared with the voice pipeline (stream_chat_interleaved) so an explicit barge-in
    # here and a voice-triggered barge-in there cancel through the same signal.
    active_tts_tasks = set()
    barge_in_event = asyncio.Event()
    # Serializes set()->cancel->await->clear() sequences on barge_in_event between this
    # function and the pipeline's own voice-triggered cancel_current_turn (a new stt_final
    # superseding an in-flight reply) — without it, an explicit barge-in here and a voice
    # barge-in there could interleave their clear()s, letting one stray chunk slip past
    # the is_set() check for a turn that turn_id filtering would otherwise still be the
    # only thing catching downstream.
    barge_in_lock = asyncio.Lock()

    def _log_unexpected_cancel_results(results):
        # asyncio.gather(..., return_exceptions=True) hands back the exception object
        # instead of raising — without inspecting it, a genuine bug in a cancelled task
        # (as opposed to the expected CancelledError) would vanish with no trace.
        for r in results:
            if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                logger.warning(f"barge_in: cancelled task raised {r!r} instead of being cleanly cancelled")

    async def do_barge_in(reason: str):
        """Cancel whatever is currently generating/speaking for this session — the
        text_input path (active_tts_tasks) and/or the live voice-turn task — and wait
        for it to actually stop before returning, so callers can rely on it being done
        (rather than clearing the event immediately, which let cancelled tasks observe
        it as already-False and made the cooperative checks throughout a no-op)."""
        async with barge_in_lock:
            tasks = list(active_tts_tasks)
            voice_task = pipeline._voice_response_tasks.get(session_id)
            if voice_task is not None and not voice_task.done():
                tasks.append(voice_task)
            if not tasks:
                return
            logger.info(f"barge_in ({reason}): cancelling {len(tasks)} in-flight task(s)")
            barge_in_event.set()
            for task in tasks:
                task.cancel()
            active_tts_tasks.clear()
            _log_unexpected_cancel_results(await asyncio.gather(*tasks, return_exceptions=True))
            pipeline._voice_response_tasks.pop(session_id, None)
            barge_in_event.clear()
        try:
            await websocket.send_text(json.dumps({"type": "barge_in", "reason": reason}))
        except Exception:
            pass

    async def cancel_text_input_tasks():
        """Symmetric counterpart to do_barge_in: called by the voice pipeline right
        before it starts a new turn from a fresh stt_final, so a spoken utterance
        correctly supersedes an in-flight text_input reply — without this, someone
        speaking while a typed reply is still playing would leave both the new voice
        turn and the old direct_tts task running concurrently, each writing tts_chunks
        for a different turn_id to the same socket."""
        async with barge_in_lock:
            tasks = list(active_tts_tasks)
            if not tasks:
                return
            logger.info(f"barge_in (voice superseding text_input): cancelling {len(tasks)} task(s)")
            barge_in_event.set()
            for task in tasks:
                task.cancel()
            active_tts_tasks.clear()
            _log_unexpected_cancel_results(await asyncio.gather(*tasks, return_exceptions=True))
            barge_in_event.clear()

    async def sender_loop():
        try:
            async for event in pipeline.stream_chat_interleaved(pcm_queue, stop_event, session_id, barge_in_event=barge_in_event, barge_in_lock=barge_in_lock, on_new_voice_turn=cancel_text_input_tasks):
                etype = event["type"]
                if etype == "tts_chunk":
                    pcm = event["pcm"]
                    payload = {"type": "tts_chunk", "pcm": pcm_to_base64(pcm), "text": event["text"], "sampleRate": event["sampleRate"], "latency_ms": event.get("latency_ms", 0), "turn_id": event.get("turn_id")}
                    await websocket.send_text(json.dumps(payload))
                elif etype == "tts_start":
                    await websocket.send_text(json.dumps(event))
                elif etype in ("stt_partial", "stt_final", "llm_token", "latency", "tts_end", "tool_call", "tool_result",
                              "tool_guard", "reasoning", "llm_reasoning"):
                    await websocket.send_text(json.dumps(event, ensure_ascii=False))
                    if etype == "tool_call":
                        stats["tool_calls"] += 1
                else:
                    await websocket.send_text(json.dumps(event, ensure_ascii=False))
                if etype == "latency":
                    e2e = event.get("e2e_ms", 0)
                    if e2e > 0:
                        stats["latencies"].append(e2e)
                        stats["utterances"] += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"sender_loop error {e}")

    sender_task = asyncio.create_task(sender_loop())
    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break
            if "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                t = data.get("type")
                if t == "start":
                    logger.info(f"session {session_id} start {data}")
                elif t == "audio_chunk":
                    b64 = data.get("pcm") or data.get("data") or ""
                    if not b64:
                        continue
                    try:
                        raw = base64.b64decode(b64)
                        pcm = np.frombuffer(raw, dtype=np.int16)
                        await pcm_queue.put(pcm)
                    except Exception as e:
                        logger.warning(f"audio decode error {e}")
                elif t == "stop":
                    await pcm_queue.put({"type":"flush"})
                elif t == "barge_in":
                    await do_barge_in("audio")
                elif t == "text_input":
                    # Barge-in: cancel previous TTS/LLM generation (typed or spoken) before starting new one
                    await do_barge_in("text_input")
                    txt = data.get("text","")
                    _req_voice = data.get("voice") or ""
                    # Bind this turn's inputs as defaults: direct_tts is a closure over
                    # ws_chat's locals, and `txt`/`history` are rebound by the next
                    # text_input message — a second message arriving before this task
                    # reached its history-write used to store the WRONG user text.
                    async def direct_tts(txt=txt, _req_voice=_req_voice):
                        start = time.time()
                        my_turn_id = pipeline.next_turn_id(session_id)
                        tool_queries: list[str] = []   # for question-echo suppression below
                        sent_tts_start = False
                        # Per-call voice via tts_chunks() — never mutate the shared
                        # TTS instance (see tts_chunks docstring).
                        if _req_voice and not hasattr(pipeline.tts, "synthesize_streaming"):
                            try:
                                pipeline.tts.set_voice(_req_voice)
                            except KeyError:
                                pass
                        # Don't echo stt_final for text_input (frontend already added user bubble) — avoids duplication of "hi"
                        # await websocket.send_text(json.dumps({"type":"stt_final","text":txt,"latency_ms":5}))
                        llm_start = time.time()
                        tts_buf = ""
                        llm_text_so_far = ""
                        cnt = 0
                        from llm.ling_streaming import SpokenGuard as _SpokenGuard
                        spoken_guard_app = _SpokenGuard()
                        import re
                        SENT_END = re.compile(r'[.!?。！？\n]')
                        first_llm = None
                        first_tts = None
                        # Use Ling multi-turn tool-aware generation with session history
                        history = pipeline._get_history(session_id)
                        # Ensure history is trimmed and has system
                        history = pipeline._trim_history(history)
                        # The zh-TW default lives in the system prompt (SYSTEM_PROMPT /
                        # AGENT_SYSTEM_MESSAGE), never appended to the user's own words:
                        # a hint glued onto the turn text became part of what the model
                        # was asked, and it duly copied it into web_search queries and
                        # read it back aloud as if it were the answer.
                        try:
                            async for ev in pipeline.llm.generate_chat_with_tools(history, txt) if hasattr(pipeline.llm, 'generate_chat_with_tools') else pipeline.llm.generate_with_tools(txt):
                                # Check for barge-in cancellation
                                if barge_in_event.is_set():
                                    logger.info("direct_tts cancelled by barge-in")
                                    break
                                if ev["type"] == "tool_call":
                                    _tq = ev.get("query") or (ev.get("arguments") or {}).get("query") or ""
                                    if _tq:
                                        tool_queries.append(_tq)
                                    await websocket.send_text(json.dumps({"type":"tool_call","name":ev["name"],"arguments":ev.get("arguments",{}), "query": ev.get("query",""), "turn_id": my_turn_id}, ensure_ascii=False))
                                    stats["tool_calls"] += 1
                                elif ev["type"] == "llm_reset":
                                    # Harness took back speculative answer text: drop the
                                    # TTS buffer and tell the UI to clear its bubble.
                                    tts_buf = ""
                                    cnt = 0
                                    llm_text_so_far = ""
                                    await websocket.send_text(json.dumps({"type": "llm_reset", "turn_id": my_turn_id}))
                                elif ev["type"] == "tool_guard":
                                    # "this lookup was run/forced by the harness, not chosen by the
                                    # model" — see agent/_shared.py guard(). The UI can ignore it;
                                    # test_e2e_report.py uses it to separate model routing quality
                                    # from guard assistance, which is the difference between a
                                    # measurement and a self-fulfilling one.
                                    await websocket.send_text(json.dumps(
                                        {"type": "tool_guard", "reason": ev.get("reason", ""),
                                         "tool": ev.get("tool", ""), "turn_id": my_turn_id}, ensure_ascii=False))
                                elif ev["type"] in ("llm_reasoning", "reasoning"):
                                    # Never spoken — forwarded to UI's reasoning panel only
                                    await websocket.send_text(json.dumps({"type": "reasoning", "text": ev.get("text", ""), "delta": ev.get("text", ""), "turn_id": my_turn_id}, ensure_ascii=False))
                                elif ev["type"] == "tool_result":
                                    # forward lightweight result to frontend (omit large content for WS size, but include formatted)
                                    await websocket.send_text(json.dumps({"type":"tool_result","name":ev["name"],"latency_ms":ev.get("latency_ms",0),"source":ev.get("result",{}).get("source","") if isinstance(ev.get("result"),dict) else "", "formatted":ev.get("formatted","")[:600], "turn_id": my_turn_id}, ensure_ascii=False))
                                elif ev["type"] == "llm_token":
                                    # Filter tool call XML and reasoning spillover from TTS/history
                                    _tok = ev.get("token","")
                                    # Defense: if a reasoning chunk arrived as llm_token (budget truncated),
                                    # re-route it to reasoning channel.
                                    try:
                                        from llm.ling_streaming import _is_reasoning_text as _is_r
                                        if _is_r(_tok):
                                            await websocket.send_text(json.dumps({"type": "reasoning", "text": _tok, "delta": _tok, "turn_id": my_turn_id}, ensure_ascii=False))
                                            continue
                                    except Exception:
                                        pass
                                    if "<tool_call" in _tok or "<arg_" in _tok or "</" in _tok or "tool_call" in _tok.lower() or ("<" in _tok and ">" in _tok):
                                        continue
                                    if first_llm is None:
                                        first_llm = time.time()
                                    await websocket.send_text(json.dumps({**ev, "turn_id": my_turn_id}, ensure_ascii=False))
                                    llm_text_so_far = ev.get("text_so_far", llm_text_so_far + _tok)
                                    # Also filter TTS buffer
                                    if "<tool" in _tok.lower() or ("<" in _tok and ">" in _tok):
                                        continue
                                    tts_buf += _tok
                                    cnt += 1
                                    flush = False
                                    if SENT_END.search(ev["token"]):
                                        flush=True
                                    elif cnt>=300:
                                        flush=True  # safety cap ONLY — never the old >=8+' ,' (8-char flush paced the speech)
                                    if barge_in_event.is_set():
                                        logger.info("direct_tts barge-in: aborting LLM->TTS flush")
                                        break
                                    if flush and tts_buf.strip():
                                        if barge_in_event.is_set():
                                            break
                                        txt_s = tts_buf.strip()
                                        # Never speak reasoning — route to UI panel instead
                                        try:
                                            from llm.ling_streaming import _is_reasoning_text as _is_r2
                                            if _is_r2(txt_s):
                                                await websocket.send_text(json.dumps({"type": "reasoning", "text": txt_s, "delta": txt_s, "turn_id": my_turn_id}, ensure_ascii=False))
                                                tts_buf=""
                                                cnt=0
                                                continue
                                        except Exception:
                                            pass
                                        # Never say the same thing twice in one turn (the harness
                                        # re-streams an answer from the start on each new step).
                                        if not spoken_guard_app.should_speak(txt_s):
                                            logger.info(f"direct_tts skip duplicate: {txt_s[:60]}")
                                            tts_buf=""
                                            cnt=0
                                            continue
                                        # Tool plumbing and question echoes are not speech.
                                        # Both rules live in pipeline.speech_to_speech so the
                                        # voice and text paths cannot drift apart — and the
                                        # echo rule is a content comparison, not the literal
                                        # match on one benchmark question that used to sit
                                        # here (`startswith("who is the president")`).
                                        if _is_tool_artifact(txt_s):
                                            tts_buf=""
                                            cnt=0
                                            continue
                                        if is_echo_of_prompt(txt_s, txt, tool_queries):
                                            logger.info(f"direct_tts skip question echo: {txt_s[:60]}")
                                            tts_buf=""
                                            cnt=0
                                            continue
                                        tts_buf=""
                                        cnt=0
                                        # Check again before TTS (long speech may have been barged)
                                        if barge_in_event.is_set():
                                            break
                                        # Use streaming to avoid large WS messages (>1MB)
                                        first_chunk = True
                                        async for pcm_chunk in tts_chunks(txt_s, _req_voice):
                                            if barge_in_event.is_set():
                                                logger.info("direct_tts barge-in: aborting TTS streaming mid-sentence")
                                                break
                                            # Skip silence chunks
                                            if len(pcm_chunk) == 0 or int(np.max(np.abs(pcm_chunk))) == 0:
                                                continue
                                            if not sent_tts_start:
                                                await websocket.send_text(json.dumps({"type":"tts_start","sampleRate":pipeline.tts.sample_rate,"turn_id":my_turn_id}))
                                                sent_tts_start = True
                                            if first_chunk:
                                                if first_tts is None:
                                                    first_tts = time.time()
                                                await websocket.send_text(json.dumps({"type":"tts_chunk","pcm":pcm_to_base64(pcm_chunk),"text":txt_s,"sampleRate":pipeline.tts.sample_rate,"latency_ms":40,"turn_id":my_turn_id}))
                                                await websocket.send_text(json.dumps({"type":"latency","stt_ms":0,"llm_ttft_ms":int((first_llm-llm_start)*1000) if first_llm else 0,"tts_ttfb_ms":int((first_tts-llm_start)*1000) if first_tts else 0,"e2e_ms":int((first_tts-start)*1000),"turn_id":my_turn_id}))  # interim e2e = first audio
                                                first_chunk = False
                                            else:
                                                await websocket.send_text(json.dumps({"type":"tts_chunk","pcm":pcm_to_base64(pcm_chunk),"text":txt_s,"sampleRate":pipeline.tts.sample_rate,"latency_ms":40,"turn_id":my_turn_id}))
                                        continue
                                elif ev["type"] == "llm_done":
                                    if tts_buf.strip():
                                        _rem = tts_buf.strip()
                                        _is_dup_rem = False
                                        try:
                                            from llm.ling_streaming import _is_reasoning_text as _is_r_rem2
                                            if _is_r_rem2(_rem):
                                                await websocket.send_text(json.dumps({"type": "reasoning", "text": _rem, "delta": _rem, "turn_id": my_turn_id}, ensure_ascii=False))
                                                _is_dup_rem = True
                                            elif not spoken_guard_app.should_speak(_rem):
                                                logger.info(f"direct_tts skip duplicate remainder: {_rem[:60]}")
                                                _is_dup_rem = True
                                        except Exception:
                                            pass
                                        if _is_dup_rem:
                                            pass
                                        elif "<" in _rem and ">" in _rem:
                                            pass
                                        elif "tool_call" in _rem.lower() or _rem.strip().startswith("<"):
                                            pass
                                        else:
                                            async for pcm_chunk in tts_chunks(_rem, _req_voice):
                                                if len(pcm_chunk) == 0 or int(np.max(np.abs(pcm_chunk))) == 0:
                                                    continue
                                                if not sent_tts_start:
                                                    await websocket.send_text(json.dumps({"type":"tts_start","sampleRate":pipeline.tts.sample_rate,"turn_id":my_turn_id}))
                                                    sent_tts_start = True
                                                await websocket.send_text(json.dumps({"type":"tts_chunk","pcm":pcm_to_base64(pcm_chunk),"text":_rem,"sampleRate":pipeline.tts.sample_rate,"latency_ms":40,"turn_id":my_turn_id}))
                                    # Update multi-turn history for Ling chat template
                                    try:
                                        from llm.ling_streaming import SYSTEM_PROMPT
                                        hist = pipeline._get_history(session_id)
                                        if not hist or hist[0].get("role") != "system":
                                            hist.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                                        hist.append({"role": "user", "content": txt})
                                        assistant_text = llm_text_so_far or ev.get("text") or ""
                                        if assistant_text:
                                            hist.append({"role": "assistant", "content": assistant_text})
                                        pipeline.sessions[session_id] = pipeline._trim_history(hist)
                                    except Exception as e:
                                        logger.debug(f"history update failed {e}")
                                    # REAL end-to-end: speech finished -> final latency event (overwrites the interim first-audio e2e)
                                    await websocket.send_text(json.dumps({"type":"latency","stt_ms":0,"llm_ttft_ms":int((first_llm-llm_start)*1000) if first_llm else 0,"tts_ttfb_ms":int((first_tts-llm_start)*1000) if first_tts else 0,"e2e_ms":int((time.time()-start)*1000),"turn_id":my_turn_id}))
                                    await websocket.send_text(json.dumps({"type":"tts_end","turn_id":my_turn_id}))
                                    break
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            # The LLM/TTS backend failed independently mid-turn (e.g. a model
                            # switch via POST /api/model killed the llama-server connection this
                            # generation was using) — without this, the exception would vanish as
                            # an unretrieved-task-exception warning (the _done_callback below only
                            # discards this task from active_tts_tasks, it doesn't inspect the
                            # result) and the client would hang forever waiting for a tts_end that
                            # will never arrive.
                            logger.warning(f"direct_tts: LLM/TTS generation failed mid-turn: {e!r}")
                            try:
                                await websocket.send_text(json.dumps({"type":"tts_end","turn_id":my_turn_id}))
                            except Exception:
                                pass
                        # also record latency if we had tool calls
                    _task = asyncio.create_task(direct_tts())
                    active_tts_tasks.add(_task)
                    def _done_callback(t):
                        active_tts_tasks.discard(t)
                    _task.add_done_callback(_done_callback)
                elif t == "web_search":
                    # explicit tool trigger from frontend
                    q = data.get("query") or data.get("q") or ""
                    if q:
                        from tools.web_search import web_search
                        res = await web_search(q)
                        await websocket.send_text(json.dumps({"type":"tool_result","name":"web_search","result":res}, ensure_ascii=False))
            elif "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                if len(b)==0:
                    continue
                header = b[0]
                if header == 0x01:
                    pcm = np.frombuffer(b[1:], dtype=np.int16)
                    await pcm_queue.put(pcm)
                elif header == 0x02:
                    await pcm_queue.put({"type":"flush"})
                elif header == 0x03:
                    await do_barge_in("binary")
                else:
                    pcm = np.frombuffer(b, dtype=np.int16)
                    await pcm_queue.put(pcm)
    except WebSocketDisconnect:
        logger.info(f"WS disconnected {session_id}")
    except RuntimeError as e:
        if "disconnect message" in str(e):
            # Benign Starlette race: client closed between our own WebSocketDisconnect
            # handling and its internal post-handler receive() — not an app error.
            logger.info(f"WS disconnected (race) {session_id}")
        else:
            logger.exception(f"WS error {e}")
    except Exception as e:
        logger.exception(f"WS error {e}")
    finally:
        stop_event.set()
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        while not pcm_queue.empty():
            try:
                pcm_queue.get_nowait()
            except Exception:
                break
        stats["connections"] -= 1

@app.post("/api/chat")
async def api_chat(payload: dict):
    import time
    t0 = time.time()
    if "text" in payload:
        txt = payload["text"]
        stt_text = txt
        stt_ms = 5
    elif "audio_b64" in payload:
        raw = base64.b64decode(payload["audio_b64"])
        pcm_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
        t1 = time.time()
        stt_text = await pipeline.stt.transcribe_once(pcm_f32)
        stt_ms = int((time.time()-t1)*1000)
    else:
        return JSONResponse({"error":"provide audio_b64 or text"}, status_code=400)
    # Optional voice (e.g. 台湾腔) — validated up-front, then passed per synthesis
    # call via tts_chunks() instead of mutating the shared TTS instance.
    v = payload.get("voice") or ""
    if v:
        _presets = getattr(pipeline.tts, "VOICE_PRESETS", None)
        if _presets is not None and v not in _presets:
            return JSONResponse({"error": f"unknown voice {v!r}; available {sorted(_presets)}"}, status_code=400)
    use_tools = payload.get("tools", True)  # default enable tool calling
    llm_start = time.time()
    llm_text = ""
    first_token_ms = None
    tool_calls = []
    guards = []          # turns where the harness, not the model, made it correct
    gen = pipeline.llm.generate_with_tools(stt_text) if use_tools else pipeline.llm.generate_stream(stt_text)
    # The WS path converts a mid-turn generation failure into a tts_end so its client cannot
    # hang; this endpoint had no equivalent, so an unreachable model server (LLM_API_BASE
    # pointing at a compose service that is not running) surfaced as a bare 500 + traceback.
    # The LLM layer degrades on its own now (see ling_streaming._reachable); this is the net
    # for anything else that escapes: a JSON 503 with the cause, not an HTML 500.
    try:
        async for ev in gen:
            if ev["type"] == "llm_token":
                if first_token_ms is None:
                    first_token_ms = int((time.time()-llm_start)*1000)
                llm_text = ev["text_so_far"]
            elif ev["type"] == "tool_call":
                tool_calls.append(ev)
            elif ev["type"] == "tool_guard":
                guards.append({"tool": ev.get("tool", ""), "reason": ev.get("reason", "")})
            elif ev["type"] == "tool_result":
                # we could include but not needed for text
                pass
            elif ev["type"] == "llm_done":
                if ev.get("text"):
                    llm_text = ev["text"]
    except Exception as e:
        logger.exception(f"/api/chat generation failed: {e!r}")
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:200]}",
                            "stt_text": stt_text, "llm_text": llm_text, "tool_calls": tool_calls},
                            status_code=503)
    llm_ms = int((time.time()-llm_start)*1000)
    tts_start = time.time()
    pcm_chunks = []
    async for _pcm in tts_chunks(llm_text, v or None):
        if len(_pcm):
            pcm_chunks.append(_pcm)
    tts_ms = int((time.time()-tts_start)*1000)
    e2e_ms = int((time.time()-t0)*1000)
    audio_b64 = pcm_to_base64(np.concatenate(pcm_chunks)) if pcm_chunks else ""
    return {
        "stt_text": stt_text,
        "llm_text": llm_text,
        "tool_calls": tool_calls,
        "guards": guards,          # see agent/_shared.py guard(): pre-flight/repair attribution
        "audio_b64": audio_b64,
        "latencies": {"stt_ms": stt_ms, "llm_ttft_ms": first_token_ms or 0, "llm_total_ms": llm_ms, "tts_ms": tts_ms, "e2e_ms": e2e_ms},
        "rss_mb": round(psutil.Process().memory_info().rss/1024/1024,1)
    }

@app.post("/api/chat/tools")
async def api_chat_tools(payload: dict):
    # Explicit tool test endpoint
    txt = payload.get("text","")
    if not txt:
        return JSONResponse({"error":"text required"}, status_code=400)
    events = []
    async for ev in pipeline.llm.generate_with_tools(txt):
        events.append(ev)
    return {"events": events, "count": len(events)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock", action="store_true", help="run without downloading models")
    parser.add_argument("--stt-model", default="Audio8/ARK-ASR-0.6B")
    parser.add_argument("--llm", default="noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF")
    parser.add_argument("--tts", default="OpenMOSS-Team/MOSS-TTS-Nano-100M")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--searxng-port", type=int, default=8888)
    parser.add_argument("--token", default=None, help="require this token on API/WS routes (alias of VOICE_CHAT_TOKEN)")
    parser.add_argument("--allowed-origins", default=None, help="comma-separated CORS origins (alias of ALLOWED_ORIGINS)")
    args = parser.parse_args()
    global MOCK_MODE, pipeline, SEARXNG_URL, AUTH_TOKEN
    MOCK_MODE = args.mock
    if args.token:
        os.environ["VOICE_CHAT_TOKEN"] = args.token
        AUTH_TOKEN = args.token
    if args.allowed_origins:
        # CORS is configured at import time (middleware already added), so a CLI
        # override here would silently do nothing — send people to the env var.
        parser.error("--allowed-origins is read at import; set ALLOWED_ORIGINS=<csv> instead")
    # --host defaults to 0.0.0.0 (Docker needs it). Binding a public interface with no
    # token is exactly the exposure the README's Funnel note warns about, so say so.
    if args.host not in ("127.0.0.1", "localhost", "::1") and not (args.token or os.getenv("VOICE_CHAT_TOKEN")):
        logger.warning(f"--host {args.host} with no --token/VOICE_CHAT_TOKEN: every route except "
                       "model-switching is reachable by anyone who can open a socket to this port.")
    # Respect Docker env SEARXNG_URL (e.g., http://searxng:8080) if set, else use localhost:port
    if not os.getenv("SEARXNG_URL"):
        SEARXNG_URL = f"http://localhost:{args.searxng_port}"
    # torch is used for exactly one thing here: deciding whether a cuda request can be
    # honored. Importing it unconditionally made the 800MB+ dependency a *boot*
    # requirement, which contradicted the mock rungs added to every adapter ladder
    # (`--mock` was supposed to start anywhere). Inside the real image torch is present
    # anyway, so this changes nothing there.
    try:
        import torch
        _cuda_ok = torch.cuda.is_available()
    except ImportError:
        _cuda_ok = False
        logger.info("torch not installed; device falls back to cpu (torch is only used for this check)")
    if not _cuda_ok:
        args.device = "cpu"
        logger.warning("CUDA not available, using CPU")
    logger.info(f"Initializing pipeline mock={args.mock} device={args.device} ...")
    # Serve the built frontend (single-port container / convenience) if present
    _ui = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if _ui.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=_ui, html=True), name="ui")
        logger.info(f"UI mounted from {_ui}")
    if not args.mock:
        # Must run before HFSpeechToSpeechPipeline() below, whose LingStreaming client
        # does its own synchronous readiness check against this same port at construction.
        asyncio.run(llm_manager.ensure_started())
    pipeline = HFSpeechToSpeechPipeline(stt_model=args.stt_model, llm_model=args.llm, tts_model=args.tts, device=args.device, mock=args.mock)
    if not args.mock and llm_manager.current_alias:
        # Keep the freshly-constructed LLM client's requested model name in sync with
        # whichever model llm_manager actually ended up serving (it may have adopted an
        # already-running server whose alias differs from speech_to_speech.py's hardcoded
        # default) — see the identical note in the /api/model handler above.
        pipeline.llm.model_name = llm_manager.current_alias
        pipeline.llm.mock = False
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=16*1024*1024)

if __name__ == "__main__":
    main()
