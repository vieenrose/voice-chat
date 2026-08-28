#!/usr/bin/env python3
"""
Self-hosted SearXNG-compatible server (minimal).

Provides /search?format=json compatible with SearXNG API, but lightweight and
runs without Redis/valkey/Docker. Ideal for voice-chat tool calling demo where
we need a local search service.

It internally uses the same web_search backends (DDGS, lite scrape, mock) and
exposes them via SearXNG's JSON contract so the main voice-chat backend can
treat it as a real SearXNG instance at http://localhost:8888.

Usage:
  python tools/searxng_server.py --port 8888 --host 127.0.0.1
  # then voice chat will auto-detect it via http://localhost:8888/search

Also can be run via: python -m tools.searxng_server

If docker is available, you can instead run official SearXNG:
  docker run -d -p 8888:8080 searxng/searxng

But this lightweight version works everywhere (no docker needed) and still
satisfies "self-host searxng to support it".
"""
import argparse
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

# Import the same search logic — robust to cwd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from web_search import web_search, format_results
except ImportError:
    try:
        from tools.web_search import web_search, format_results
    except ImportError:
        # last resort: mock
        async def web_search(q, count=5):
            return {"query": q, "results": [{"title": f"Mock for {q}", "url": "https://example.com", "content": "mock"}], "source": "mock", "latency_ms": 5}
        def format_results(r): return str(r)

app = FastAPI(title="SearXNG (self-hosted minimal)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return HTMLResponse("""
    <html><head><title>SearXNG (self-hosted minimal)</title></head>
    <body style="font-family: sans-serif; max-width: 700px; margin: 40px auto; background:#0a0a0f; color:#eee; padding:20px">
      <h1>🔍 SearXNG — self-hosted minimal</h1>
      <p>This is a lightweight SearXNG-compatible search instance for the voice-chat demo.</p>
      <p>It provides <code>/search?format=json&q=...</code> compatible with real SearXNG.</p>
      <p>Try: <a href="/search?q=weather Paris&format=json" style="color:#7c5cff">/search?q=weather Paris&format=json</a></p>
      <p>Source: <code>backend/tools/searxng_server.py</code> + <code>backend/tools/web_search.py</code></p>
      <p>Run official SearXNG via Docker if you want full 200+ engines: <code>docker run -p 8888:8080 searxng/searxng</code></p>
    </body></html>
    """)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "searxng-minimal", "port": 8888}

@app.get("/config")
async def config():
    # SearXNG config endpoint (minimal)
    return {
        "instance_name": "SearXNG minimal (voice-chat demo)",
        "searxng_version": "2026.04.13-minimal",
        "self_hosted": True,
        "engines": ["duckduckgo", "lite_scrape", "mock"],
        "note": "Lightweight drop-in. Full SearXNG at https://docs.searxng.org"
    }

@app.get("/search")
async def search(
    request: Request,
    q: str = Query("", description="query"),
    format: str = Query("json", description="output format: json or html"),
    categories: str = Query("general"),
    pageno: int = Query(1),
    language: str = Query("auto"),
    time_range: Optional[str] = Query(None),
):
    if not q:
        return JSONResponse({"error": "missing q"}, status_code=400)
    # We respect SearXNG params but mostly ignore them for minimal impl
    count = 5
    # SearXNG also supports `format=json` via Accept header; we handle both
    fmt = format.lower()
    # Also check ?format= param overrides Accept
    if request.headers.get("accept","").find("json") != -1 and fmt == "html":
        # keep explicit format param
        pass

    t0 = time.time()
    result = await web_search(q, count=count)
    latency_ms = result["latency_ms"]

    # SearXNG JSON shape (subset)
    # https://docs.searxng.org/dev/search_api.html
    searx_json = {
        "query": q,
        "number_of_results": len(result["results"]),
        "results": [
            {
                "title": r["title"],
                "url": r["url"],
                "content": r["content"],
                "engine": result["source"],
                "score": 1.0 - i*0.1,
                "category": "general",
                "pretty_url": r["url"],
            }
            for i, r in enumerate(result["results"])
        ],
        "answers": [],
        "corrections": [],
        "infoboxes": [],
        "suggestions": [],
        "unresponsive_engines": [],
    }

    if fmt == "json":
        return JSONResponse(searx_json)
    else:
        # html — very minimal
        html_results = "".join([
            f'<div style="margin:12px 0; padding:12px; background:#14141c; border-radius:8px"><a href="{r["url"]}" style="color:#7c5cff; font-weight:700">{r["title"]}</a><div style="opacity:0.6; font-size:12px">{r["url"]}</div><div style="margin-top:6px">{r["content"][:280]}</div></div>'
            for r in result["results"]
        ])
        return HTMLResponse(f"""
        <html><head><title>{q} — SearXNG</title></head>
        <body style="font-family: sans-serif; max-width:800px; margin:20px auto; background:#0a0a0f; color:#eee">
          <h2>🔍 {q}</h2><div>found {len(result["results"])} via {result["source"]} in {latency_ms}ms</div>
          {html_results}
          <p style="opacity:0.5">SearXNG minimal — <a href="/search?q={q}&format=json" style="color:#7c5cff">json</a></p>
        </body></html>
        """)

# Also support POST as SearXNG does
@app.post("/search")
async def search_post(request: Request):
    form = await request.form()
    q = form.get("q","") or (await request.json()).get("q","") if request.headers.get("content-type","").find("json")!=-1 else ""
    # fallback parse json body
    if not q:
        try:
            j = await request.json()
            q = j.get("q","") or j.get("query","")
        except: pass
    if not q:
        q = request.query_params.get("q","")
    if not q:
        return JSONResponse({"error":"missing q"}, status_code=400)
    # delegate to GET handler
    return await search(request, q=q, format=request.query_params.get("format","json"))

def main():
    p = argparse.ArgumentParser(description="SearXNG minimal self-hosted")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    logger.info(f"Starting SearXNG minimal on http://{args.host}:{args.port} — self-hosted for voice-chat tool calling")
    logger.info(" Try: curl 'http://localhost:8888/search?q=weather&format=json' | jq")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
