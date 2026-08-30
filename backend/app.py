"""
FastAPI backend for streaming voice chat with tool calling (web_search via self-hosted SearXNG).

Exposes:
  GET /health, /stats, /search (SearXNG compat), /api/search, /api/chat, /api/chat/tools
  WS  /ws/chat

Low-latency WebSocket protocol:
  Client JSON: {"type":"start"|"audio_chunk"|"stop"|"barge_in"|"text_input", ...}
  Backend JSON: {"type":"stt_partial"|"stt_final"|"llm_token"|"tool_call"|"tool_result"|"tts_chunk"|"tts_end"|"latency"}
"""
import asyncio
import base64
import argparse
import time
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(__file__))
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger
import numpy as np

from pipeline.speech_to_speech import HFSpeechToSpeechPipeline
from llm_manager import llm_manager, MODEL_REGISTRY as LLM_MODEL_REGISTRY

app = FastAPI(title="Voice Chat HF S2S + SearXNG Tools", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: HFSpeechToSpeechPipeline | None = None
stats = {"connections": 0, "utterances": 0, "latencies": [], "tool_calls": 0}
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
    except:
        pass
    # Try to start minimal SearXNG server via subprocess
    try:
        import subprocess, pathlib
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

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ensure_searxng())

# --- health & stats ---
@app.get("/health")
async def health():
    mem = psutil.Process().memory_info()
    # Check searxng status
    searxng_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{SEARXNG_URL}/healthz")
            searxng_ok = r.status_code == 200
    except:
        searxng_ok = False
    return {
        "status": "ok",
        "mock": MOCK_MODE,
        "models_loaded": {
            "stt": pipeline.stt.backend if pipeline else "not_loaded",
            "llm": "mock" if MOCK_MODE else (pipeline.llm.backend if pipeline and hasattr(pipeline.llm, 'model') else "loaded"),
            "tts": pipeline.tts.backend if pipeline else "not_loaded",
        },
        "llm_manager": llm_manager.status(),
        "rss_mb": round(mem.rss / 1024/1024, 1),
        "vms_mb": round(mem.vms / 1024/1024, 1),
        "searxng": {"url": SEARXNG_URL, "ok": searxng_ok, "self_hosted": True},
        "stats": stats
    }

@app.get("/api/model")
async def get_model():
    return llm_manager.status()

@app.post("/api/model")
async def switch_model(payload: dict):
    """Switch the loaded text-LLM model. Stops the current llama-server and
    starts the requested one — the caller should expect this to take several
    seconds and the pipeline's LLM to be briefly unavailable while it swaps."""
    model_id = payload.get("model_id") or payload.get("model")
    if not model_id:
        return JSONResponse({"error": "model_id required"}, status_code=400)
    if model_id not in LLM_MODEL_REGISTRY:
        return JSONResponse({"error": f"unknown model_id {model_id!r}", "available": list(LLM_MODEL_REGISTRY)}, status_code=400)
    if llm_manager.switching:
        return JSONResponse({"error": "a model switch is already in progress"}, status_code=409)
    try:
        result = await llm_manager.switch_to(model_id)
    except Exception as e:
        logger.exception(f"model switch to {model_id} failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
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
    lats = stats["latencies"]
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
        html = "".join([f'<div><a href="{r["url"]}">{r["title"]}</a><p>{r["content"][:200]}</p></div>' for r in res["results"]])
        return HTMLResponse(f"<html><body><h2>{q}</h2>{html}</body></html>")

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

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, session_id: str = Query(default="default")):
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
                elif etype in ("stt_partial", "stt_final", "llm_token", "latency", "tts_end", "tool_call", "tool_result"):
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
                except:
                    continue
                t = data.get("type")
                if t == "start":
                    logger.info(f"session {session_id} start {data}")
                elif t == "audio_chunk":
                    b64 = data.get("pcm") or data.get("data") or ""
                    if not b64: continue
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
                    async def direct_tts():
                        start = time.time()
                        my_turn_id = pipeline.next_turn_id(session_id)
                        sent_tts_start = False
                        if _req_voice and hasattr(pipeline.tts, "set_voice"):
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
                        import re
                        SENT_END = re.compile(r'[.!?。！？\n]')
                        first_llm = None
                        first_tts = None
                        # Use Ling multi-turn tool-aware generation with session history
                        history = pipeline._get_history(session_id)
                        # Ensure history is trimmed and has system
                        history = pipeline._trim_history(history)
                        # See llm/ling_streaming.py:LANG_HINT for why this is appended every turn
                        # unconditionally (zh-TW default, English only for untranslatable terms).
                        from llm.ling_streaming import LANG_HINT as _lang_hint
                        try:
                            async for ev in pipeline.llm.generate_chat_with_tools(history, txt + _lang_hint) if hasattr(pipeline.llm, 'generate_chat_with_tools') else pipeline.llm.generate_with_tools(txt):
                                # Check for barge-in cancellation
                                if barge_in_event.is_set():
                                    logger.info("direct_tts cancelled by barge-in")
                                    break
                                if ev["type"] == "tool_call":
                                    await websocket.send_text(json.dumps({"type":"tool_call","name":ev["name"],"arguments":ev.get("arguments",{}), "query": ev.get("query",""), "turn_id": my_turn_id}, ensure_ascii=False))
                                    stats["tool_calls"] += 1
                                elif ev["type"] == "tool_result":
                                    # forward lightweight result to frontend (omit large content for WS size, but include formatted)
                                    await websocket.send_text(json.dumps({"type":"tool_result","name":ev["name"],"latency_ms":ev.get("latency_ms",0),"source":ev.get("result",{}).get("source","") if isinstance(ev.get("result"),dict) else "", "formatted":ev.get("formatted","")[:600], "turn_id": my_turn_id}, ensure_ascii=False))
                                elif ev["type"] == "llm_token":
                                    # Filter tool call XML from TTS/history
                                    _tok = ev.get("token","")
                                    if "<tool_call" in _tok or "<arg_" in _tok or "</" in _tok or "tool_call" in _tok.lower() or ("<" in _tok and ">" in _tok):
                                        continue
                                    if first_llm is None: first_llm = time.time()
                                    await websocket.send_text(json.dumps({**ev, "turn_id": my_turn_id}, ensure_ascii=False))
                                    llm_text_so_far = ev.get("text_so_far", llm_text_so_far + _tok)
                                    # Also filter TTS buffer
                                    if "<tool" in _tok.lower() or ("<" in _tok and ">" in _tok):
                                        continue
                                    tts_buf += _tok
                                    cnt += 1
                                    flush = False
                                    if SENT_END.search(ev["token"]): flush=True
                                    elif cnt>=300: flush=True  # safety cap ONLY — never the old >=8+' ,' (8-char flush paced the speech)
                                    if barge_in_event.is_set():
                                        logger.info("direct_tts barge-in: aborting LLM->TTS flush")
                                        break
                                    if flush and tts_buf.strip():
                                        if barge_in_event.is_set():
                                            break
                                        txt_s = tts_buf.strip()
                                        # Skip tool call artifacts and silence - more aggressive
                                        _low = txt_s.lower()
                                        if ("<" in txt_s and ">" in txt_s) or "tool_call" in _low or "arg_key" in _low or "arg_value" in _low or txt_s.strip().lower() in ["web_search", "query"] or ("web_search" in _low and len(txt_s.split()) < 4):
                                            tts_buf=""; cnt=0
                                            continue
                                        if txt_s.strip().startswith("<") or "tool_call" in _low:
                                            tts_buf=""; cnt=0
                                            continue
                                        # Skip tool query like "Who is the president of France 2024" when in tool context
                                        if txt_s.strip().lower().startswith("who is the president") and len(txt_s.split()) < 10:
                                            # Check if recent history had tool call
                                            try:
                                                hist_str = str(history).lower()
                                                if "tool" in hist_str and "web_search" in hist_str:
                                                    tts_buf=""; cnt=0
                                                    continue
                                            except: pass
                                        tts_buf=""; cnt=0
                                        # Check again before TTS (long speech may have been barged)
                                        if barge_in_event.is_set():
                                            break
                                        # Use streaming to avoid large WS messages (>1MB)
                                        first_chunk = True
                                        async for pcm_chunk in pipeline.tts.synthesize_streaming(txt_s):
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
                                                if first_tts is None: first_tts = time.time()
                                                await websocket.send_text(json.dumps({"type":"tts_chunk","pcm":pcm_to_base64(pcm_chunk),"text":txt_s,"sampleRate":pipeline.tts.sample_rate,"latency_ms":40,"turn_id":my_turn_id}))
                                                await websocket.send_text(json.dumps({"type":"latency","stt_ms":0,"llm_ttft_ms":int((first_llm-llm_start)*1000) if first_llm else 0,"tts_ttfb_ms":int((first_tts-llm_start)*1000) if first_tts else 0,"e2e_ms":int((first_tts-start)*1000),"turn_id":my_turn_id}))  # interim e2e = first audio
                                                first_chunk = False
                                            else:
                                                await websocket.send_text(json.dumps({"type":"tts_chunk","pcm":pcm_to_base64(pcm_chunk),"text":txt_s,"sampleRate":pipeline.tts.sample_rate,"latency_ms":40,"turn_id":my_turn_id}))
                                        continue
                                elif ev["type"] == "llm_done":
                                    if tts_buf.strip():
                                        _rem = tts_buf.strip()
                                        if "<" in _rem and ">" in _rem:
                                            pass
                                        elif "tool_call" in _rem.lower() or _rem.strip().startswith("<"):
                                            pass
                                        else:
                                            async for pcm_chunk in pipeline.tts.synthesize_streaming(_rem):
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
                if len(b)==0: continue
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
        try: await sender_task
        except asyncio.CancelledError: pass
        while not pcm_queue.empty():
            try: pcm_queue.get_nowait()
            except: break
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
    # Optional voice (e.g. 台湾腔) — set on TTS before synthesis
    v = payload.get("voice") or ""
    if v and hasattr(pipeline.tts, "set_voice"):
        try:
            pipeline.tts.set_voice(v)
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    use_tools = payload.get("tools", True)  # default enable tool calling
    llm_start = time.time()
    llm_text = ""
    first_token_ms = None
    tool_calls = []
    gen = pipeline.llm.generate_with_tools(stt_text) if use_tools else pipeline.llm.generate_stream(stt_text)
    async for ev in gen:
        if ev["type"] == "llm_token":
            if first_token_ms is None: first_token_ms = int((time.time()-llm_start)*1000)
            llm_text = ev["text_so_far"]
        elif ev["type"] == "tool_call":
            tool_calls.append(ev)
        elif ev["type"] == "tool_result":
            # we could include but not needed for text
            pass
        elif ev["type"] == "llm_done":
            if ev.get("text"): llm_text = ev["text"]
    llm_ms = int((time.time()-llm_start)*1000)
    tts_start = time.time()
    pcm_chunks = []
    async for ev in pipeline.tts.tts_from_text(llm_text):
        if ev["type"] == "tts_chunk":
            pcm_chunks.append(ev["pcm"])
    tts_ms = int((time.time()-tts_start)*1000)
    e2e_ms = int((time.time()-t0)*1000)
    audio_b64 = pcm_to_base64(np.concatenate(pcm_chunks)) if pcm_chunks else ""
    return {
        "stt_text": stt_text,
        "llm_text": llm_text,
        "tool_calls": tool_calls,
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
    args = parser.parse_args()
    global MOCK_MODE, pipeline, SEARXNG_URL
    MOCK_MODE = args.mock
    # Respect Docker env SEARXNG_URL (e.g., http://searxng:8080) if set, else use localhost:port
    if not os.getenv("SEARXNG_URL"):
        SEARXNG_URL = f"http://localhost:{args.searxng_port}"
    import torch
    if not torch.cuda.is_available():
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
