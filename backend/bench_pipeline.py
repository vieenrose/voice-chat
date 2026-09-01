#!/usr/bin/env python3
"""Benchmark current pipeline: e2e latency (WS streaming + audio STT), peak RSS, search accuracy."""
import asyncio
import json
import base64
import time
import numpy as np
import httpx

BASE = "http://127.0.0.1:8000"
WS = "ws://127.0.0.1:8000/ws/chat"

SEARCH_Q = [
    "What is the weather in Paris today?",
    "Who is the president of France?",
    "Python 3.14 new features",
    "latest AI news 2026",
    "台灣今日天氣",
]

async def ws_e2e(q, sid):
    import websockets
    uri = f"{WS}?session_id={sid}"
    async with websockets.connect(uri, max_size=16*1024*1024) as ws:
        await ws.send(json.dumps({"type": "text_input", "text": q}))
        t_send = time.time()
        first=None
        t_end=None
        nch=0
        maxamp=0
        toks=0
        async for m in ws:
            d = json.loads(m)
            t = d.get("type")
            if t == "llm_token":
                toks += 1
            if t == "tts_chunk":
                if first is None:
                    first = (time.time()-t_send)*1000
                pcm = np.frombuffer(base64.b64decode(d["pcm"]), dtype=np.int16)
                nch += 1
                maxamp = max(maxamp, int(np.max(np.abs(pcm))) if len(pcm) else 0)
            if t == "tts_end":
                t_end = (time.time()-t_send)*1000
                break
    return {"ttfa_ms": round(first or 0), "e2e_ms": round(t_end or 0), "chunks": nch, "peak_amp": maxamp, "llm_tokens": toks}

async def api_audio_e2e():
    """Real audio STT path: /api/chat with audio_b64 (16k speech clip)"""
    import soundfile as sf
    import scipy.signal as sp
    w, sr = sf.read("/tmp/MOSS-TTS-Nano/assets/audio/zh_4.wav", dtype="float32")
    if w.ndim == 2:
        w = w.mean(axis=1)
    w16 = sp.resample_poly(w, 16000, sr).astype(np.float32)
    pcm16 = (w16*32767).astype(np.int16)
    b64 = base64.b64encode(pcm16[:16000*3].tobytes()).decode()
    async with httpx.AsyncClient(timeout=120) as c:
        t0 = time.time()
        r = await c.post(f"{BASE}/api/chat", json={"audio_b64": b64, "tools": False})
        dt = (time.time()-t0)*1000
        d = r.json()
        return {"wall_ms": round(dt), "stt_ms": d["latencies"]["stt_ms"], "llm_ms": d["latencies"]["llm_total_ms"],
                "tts_ms": d["latencies"]["tts_ms"], "e2e_ms": d["latencies"]["e2e_ms"],
                "text": d["stt_text"][:40]}

async def rss(current, peak, sample):
    peak[0] = max(peak[0], current)
    return sample

async def main():
    print("="*60)
    print("BENCHMARK: E2E (X-ASR → Qwen3.5-2B-MTP → Qwen3-TTS true streaming)")
    print("="*60)

    # --- 1) WS streaming e2e (text input) + RSS peak sampling ---
    peak_rss = [0]
    async def rss_watcher():
        while True:
            try:
                async with httpx.AsyncClient(timeout=3) as c:
                    r = await c.get(f"{BASE}/health")
                    peak_rss[0] = max(peak_rss[0], r.json()["rss_mb"])
            except Exception:
                pass
            await asyncio.sleep(0.4)
    w = asyncio.create_task(rss_watcher())
    print("\n## WS streaming (text_input) — time to first audio + full e2e")
    rows = []
    for i, q in enumerate(["Hello, how are you doing today?", "The quick brown fox jumps over the lazy dog."]):
        r = await ws_e2e(q, f"bm{i}")
        rows.append(r)
        print(f"  '{q[:28]}' -> TTFA {r['ttfa_ms']}ms | e2e {r['e2e_ms']}ms | {r['chunks']} chunks | peak_amp {r['peak_amp']}")
    if rows:
        print(f"  AVG TTFA {np.mean([r['ttfa_ms'] for r in rows]):.0f}ms | AVG e2e {np.mean([r['e2e_ms'] for r in rows]):.0f}ms")
    w.cancel()

    # --- 2) Real audio STT e2e ---
    print("\n## Audio STT path (X-ASR, 3s zh speech via /api/chat audio_b64)")
    ar = await api_audio_e2e()
    print(f"  stt {ar['stt_ms']}ms | llm {ar['llm_ms']}ms | tts {ar['tts_ms']}ms | e2e {ar['e2e_ms']}ms (wall {ar['wall_ms']}ms)")
    print(f"  stt_text: '{ar['text']}'")

    # --- 3) Peak RSS ---
    print("\n## Peak RSS")
    async with httpx.AsyncClient(timeout=5) as c:
        h = (await c.get(f"{BASE}/health")).json()
    # qwen llama-server + bge rss + backend current
    import subprocess
    def rss_of(pattern):
        out = subprocess.run(["ps", "-eo", "pid,rss,cmd"], capture_output=True, text=True).stdout
        tot = 0
        for line in out.splitlines():
            if pattern in line and "grep" not in line:
                try:
                    tot += int(line.split()[1])
                except Exception:
                    pass
        return tot/1024  # MB
    qwen_mb = rss_of("Qwen3.5-2B-MTP")
    ling_mb = rss_of("Ling-3.0-tiny") or 0
    bge_mb = rss_of("bge-small")
    back = h["rss_mb"]
    print(f"  backend python      : {back:.0f} MB (peak during chat {peak_rss[0]:.0f} MB)")
    print(f"  llama-server Qwen3  : {qwen_mb:.0f} MB")
    print(f"  llama-server bge    : {bge_mb:.0f} MB")
    if ling_mb:
        print(f"  llama-server Ling   : {ling_mb:.0f} MB")   # only present when that model is loaded
    print(f"  TOTAL (this box)    : {back+qwen_mb+bge_mb+(ling_mb or 0):.0f} MB RAM")
    v = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"  GPU VRAM used       : {v}")

    # --- 4) Search accuracy ---
    print("\n## Search accuracy (via /api/search)")
    ok = 0
    total = len(SEARCH_Q)
    async with httpx.AsyncClient(timeout=60) as c:
        for q in SEARCH_Q:
            r = (await c.get(f"{BASE}/api/search", params={"q": q})).json()
            src = r.get("source","")
            res = r.get("results", [])
            real = src.startswith("searxng") or src in ("duckduckgo","lite_scrape")
            kw = [w for w in ["paris","macron","python","ai","news","台灣","天氣"] if w.lower() in q.lower()]
            hit = any(any(k.lower() in (x.get("title","")+x.get("content","")+x.get("url","")).lower() for x in res[:3]) for k in kw) if kw else False
            acc = 1.0 if (real and hit) else (0.7 if real else 0.2)
            if real:
                ok += 1
            print(f"  '{q[:34]}' -> src={src:<14} n={len(res):<2} real={real} kw_hit={hit} (acc {acc:.2f})")
    print(f"  REAL-source rate: {ok}/{total} ({ok/total*100:.0f}%)")

asyncio.run(main())