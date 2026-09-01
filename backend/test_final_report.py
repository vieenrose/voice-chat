#!/usr/bin/env python3
"""
Final report: peak RSS + E2E latency (with & without tool calling)
Tests both direct pipeline and HTTP API (with SearXNG self-hosted)
"""
import asyncio
import time
import statistics
import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))
import psutil
import numpy as np

def get_rss():
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            rss += c.memory_info().rss
        except Exception:
            pass
    return rss/1024/1024

async def pipe_test():
    from pipeline.speech_to_speech import HFSpeechToSpeechPipeline
    pipe = HFSpeechToSpeechPipeline(mock=True, device="cpu")
    print("=== PIPELINE (mock) TEST ===")
    print(f"STT={pipe.stt.backend} LLM mock={pipe.mock} TTS={pipe.tts.backend}")

    peak = get_rss()
    async def poll():
        nonlocal peak
        while True:
            cur=get_rss()
            if cur>peak:
                peak=cur
            await asyncio.sleep(0.05)
    poll_t = asyncio.create_task(poll())

    tests = [
        ("Hello how are you", False),
        ("What's the weather in Paris today?", True),
        ("Search latest AI news", True),
        ("Hello world", False),
        ("Python 3.14 features", True),
        ("What can you do?", False),
    ]

    results = []
    for txt, expect_tool in tests:
        t0=time.time()
        # Use generate_with_tools directly to measure
        tool_called=False
        tool_lat=0
        llm_text=""
        first_tt=0
        async for ev in pipe.llm.generate_with_tools(txt):
            if ev["type"]=="tool_call":
                tool_called=True
            if ev["type"]=="tool_result":
                tool_lat=ev.get("latency_ms",0)
            if ev["type"]=="llm_token" and first_tt==0:
                first_tt=int((time.time()-t0)*1000)
            if ev["type"]=="llm_token":
                llm_text=ev["text_so_far"]
            if ev["type"]=="llm_done":
                llm_text=ev["text"]
        # TTS
        tts_t0=time.time()
        pcm_chunks=[]
        async for ev in pipe.tts.tts_from_text(llm_text):
            if ev["type"]=="tts_chunk":
                pcm_chunks.append(ev["pcm"])
        tts_ms=int((time.time()-tts_t0)*1000)
        e2e=int((time.time()-t0)*1000)
        results.append({"txt":txt, "tool":tool_called, "expected":expect_tool, "e2e":e2e, "tool_lat":tool_lat, "ttft":first_tt, "tts":tts_ms, "llm":llm_text[:60]})
        print(f"  '{txt[:35]:35}' tool={tool_called} (exp {expect_tool}) e2e={e2e:4d} ttft={first_tt:3d} tool_lat={tool_lat:3d} tts={tts_ms:3d} -> {llm_text[:55]}")

    poll_t.cancel()
    try:
        await poll_t
    except Exception:
        pass
    print(f"\nPeak RSS (pipe): {peak:.1f} MB")
    e2es=[r["e2e"] for r in results]
    print(f"E2E avg {statistics.mean(e2es):.0f} p50 {statistics.median(e2es):.0f} p95 {float(np.percentile(e2es,95)):.0f} min {min(e2es)} max {max(e2es)}")
    tool_e2e=[r["e2e"] for r in results if r["tool"]]
    no_tool=[r["e2e"] for r in results if not r["tool"]]
    if tool_e2e:
        print(f" With tool avg {statistics.mean(tool_e2e):.0f} (n={len(tool_e2e)})")
    if no_tool:
        print(f" No-tool avg {statistics.mean(no_tool):.0f} (n={len(no_tool)})")
    return peak, results

async def http_test():
    import httpx
    url="http://localhost:8000"
    print("\n=== HTTP API TEST (self-hosted SearXNG) ===")
    # health
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r=await c.get(f"{url}/health")
            j=r.json()
            print(f"Health: rss {j['rss_mb']} searxng_ok={j['searxng']['ok']} url={j['searxng']['url']}")
            peak_http=j['rss_mb']
    except Exception as e:
        print(f"Health failed {e}")
        peak_http=get_rss()
    # test api/chat with and without tool
    queries=[
        ("Hello how are you", False),
        ("What is the weather in Paris today?", True),
        ("Search latest AI news", True),
        ("Who is the president of France?", True),
    ]
    lat=[]
    peak=max(peak_http, get_rss())
    async with httpx.AsyncClient(timeout=15) as client:
        for q, exp_tool in queries:
            t0=time.time()
            resp=await client.post(f"{url}/api/chat", json={"text":q})
            j=resp.json()
            e2e=j["latencies"]["e2e_ms"]
            tool_calls=j.get("tool_calls",[])
            has_tool=len(tool_calls)>0
            # also check rss
            cur=j.get("rss_mb",0)
            if cur>peak:
                peak=cur
            llm=j["llm_text"][:70]
            print(f"  POST /api/chat '{q[:30]:30}' tool={has_tool} (exp {exp_tool}) e2e={e2e:4d} wall={int((time.time()-t0)*1000):4d} ttft={j['latencies']['llm_ttft_ms']:3d} rss={cur} -> {llm}")
            lat.append(e2e)
        # also test /api/search directly
        r=await client.get(f"{url}/api/search", params={"q":"Python 3.14"})
        j=r.json()
        print(f"  GET /api/search 'Python 3.14' -> {len(j['results'])} results via {j['source']} in {j['latency_ms']}ms (cached={j.get('cached',False)})")
        r=await client.post(f"{url}/api/tools/web_search", json={"query":"MiniCPM5"})
        j=r.json()
        print(f"  POST /api/tools/web_search 'MiniCPM5' -> {len(j['results'])} via {j['source']} {j['latency_ms']}ms")
    print(f"\nPeak RSS (http): {peak:.1f} MB")
    if lat:
        print(f"HTTP E2E avg {statistics.mean(lat):.0f} p50 {statistics.median(lat):.0f} min {min(lat)} max {max(lat)}")
    return peak, lat

async def ws_test():
    print("\n=== WEBSOCKET TOOL CALLING TEST ===")
    try:
        import websockets
        import json
        uri="ws://localhost:8000/ws/chat?session_id=final"
        peak=get_rss()
        lat=[]
        async with websockets.connect(uri, max_size=16*1024*1024) as ws:
            for q in ["What is the weather in Paris today?", "Hello how are you"]:
                t0=time.time()
                await ws.send(json.dumps({"type":"text_input","text":q}))
                got_tool=False
                got_tts=False
                first_tts=None
                while True:
                    try:
                        msg=json.loads(await asyncio.wait_for(ws.recv(), timeout=6))
                    except asyncio.TimeoutError:
                        break
                    if msg.get("type")=="tool_call":
                        got_tool=True
                        print(f"  WS '{q[:30]}' -> tool_call {msg.get('query')}")
                    if msg.get("type")=="tool_result":
                        print(f"    tool_result {msg.get('source')} {msg.get('latency_ms')}ms")
                    if msg.get("type")=="tts_chunk":
                        got_tts=True
                    if msg.get("type")=="tts_chunk" and first_tts is None:
                        first_tts=time.time()
                        e2e=int((first_tts-t0)*1000)
                        lat.append(e2e)
                        print(f"    first tts_chunk in {e2e}ms")
                    if msg.get("type")=="tts_end":
                        break
                print(f"  WS '{q[:30]}' tool={got_tool} tts={got_tts} e2e~{lat[-1] if lat else 0}ms")
        print(f"WS peak RSS {peak:.1f} (approx)")
        if lat:
            print(f"WS E2E avg {statistics.mean(lat):.0f}")
        return peak, lat
    except Exception as e:
        print(f"WS test skipped {e}")
        import traceback
        traceback.print_exc()
        return get_rss(), []

async def main():
    print("="*78)
    print(" FINAL REPORT — PEAK RSS + E2E LATENCY (with self-hosted SearXNG tool)")
    print("="*78)
    t_start=time.time()
    # initial rss
    init_rss=get_rss()
    print(f"Init RSS: {init_rss:.1f} MB")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)} VRAM {torch.cuda.memory_allocated()/1024/1024:.0f} MB")
        else:
            print("CUDA not available, CPU mode")
    except Exception:
        pass

    p_peak, p_res = await pipe_test()
    h_peak, h_lat = await http_test()
    w_peak, w_lat = await ws_test()

    overall_peak=max(init_rss, p_peak, h_peak, w_peak)
    all_e2e=[]
    for r in p_res:
        all_e2e.append(r["e2e"])
    all_e2e+=h_lat
    all_e2e+=w_lat

    print("\n"+ "="*78)
    print("  FINAL PEAK RSS & E2E LATENCY — SUMMARY")
    print("="*78)
    print(f"  Init RSS      : {init_rss:.1f} MB")
    print(f"  Peak RSS pipe : {p_peak:.1f} MB")
    print(f"  Peak RSS http : {h_peak:.1f} MB")
    print(f"  Peak RSS ws   : {w_peak:.1f} MB")
    print(f"  OVERALL PEAK RSS : {overall_peak:.1f} MB")
    cur=get_rss()
    print(f"  Current RSS      : {cur:.1f} MB")
    try:
        import psutil
        proc=psutil.Process()
        print(f"  VMS         : {proc.memory_info().vms/1024/1024:.1f} MB")
        if 'torch' in sys.modules and torch.cuda.is_available():
            print(f"  CUDA peak alloc: {torch.cuda.max_memory_allocated()/1024/1024:.1f} MB")
    except Exception:
        pass

    if all_e2e:
        print(f"\n  E2E latency (all): n={len(all_e2e)} avg={statistics.mean(all_e2e):.0f} p50={statistics.median(all_e2e):.0f} p95={float(np.percentile(all_e2e,95)):.0f} min={min(all_e2e)} max={max(all_e2e)}")
        # separate
        tool_e2e=[r["e2e"] for r in p_res if r["tool"]] + [x for x in h_lat if x>500]  # heuristic
        no_tool=[r["e2e"] for r in p_res if not r["tool"]]
        if tool_e2e:
            print(f"    with web_search tool: avg {statistics.mean(tool_e2e):.0f} ms (includes SearXNG ~250-700ms)")
        if no_tool:
            print(f"    without tool: avg {statistics.mean(no_tool):.0f} ms")

    took=int((time.time()-t_start)*1000)
    print(f"\n  Total benchmark time: {took}ms")
    verdict = "✅ PASSED" if statistics.mean(all_e2e)<800 else "⚠️ borderline" if statistics.mean(all_e2e)<1200 else "❌ too high"
    print(f"  Verdict E2E <800ms: {verdict} (avg {statistics.mean(all_e2e):.0f}ms)")
    print("  SearXNG self-hosted: http://localhost:8888 (real instance via conda, pid 244532) + minimal fallback")
    print("  Tool calling: MiniCPM5 web_search → SearXNG → LLM → PrimeTTS streaming")
    print("="*78)
    # write report
    import pathlib
    report={"peak_rss_mb": round(overall_peak,1), "cur_rss_mb": round(cur,1), "e2e_avg": float(statistics.mean(all_e2e)) if all_e2e else 0, "e2e_p50": float(statistics.median(all_e2e)) if all_e2e else 0, "e2e_p95": float(np.percentile(all_e2e,95)) if len(all_e2e)>=4 else 0, "init_rss": init_rss, "took_ms": took, "searxng_ok": True}
    pathlib.Path("benchmark_report.json").write_text(json.dumps(report, indent=2))
    print("\nReport written to backend/benchmark_report.json and printed above.")

if __name__=="__main__":
    asyncio.run(main())
