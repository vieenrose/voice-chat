#!/usr/bin/env python3
"""
WebSocket E2E latency test — connects to /ws/chat and streams text_input -> measures tts_chunk latency
Peak RSS polled via /health
"""
import asyncio
import time
import json
import statistics
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

async def test_ws(server="ws://localhost:8000/ws/chat", iters=5):
    import websockets
    import aiohttp
    # poll RSS via http
    peak = 0
    async def poll_rss():
        nonlocal peak
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get("http://localhost:8000/health") as r:
                        j=await r.json()
                        rss=j.get("rss_mb",0)
                        if rss>peak:
                            peak=rss
            except Exception:
                pass
            await asyncio.sleep(0.05)
    poll_t = asyncio.create_task(poll_rss())
    await asyncio.sleep(0.2)

    e2e_list=[]
    for i in range(iters):
        uri = f"{server}?session_id=test{i}"
        async with websockets.connect(uri, max_size=16*1024*1024) as ws:
            txt = "Hello how are you today"
            if i%3==1:
                txt="Tell me a joke about AI"
            if i%3==2:
                txt="What is low latency voice chat"
            t0=time.time()
            await ws.send(json.dumps({"type":"text_input","text":txt}))
            first_tts=None
            llm_ttft=None
            # wait for events
            try:
                async for msg in ws:
                    data=json.loads(msg)
                    if data.get("type")=="llm_token" and llm_ttft is None:
                        llm_ttft=int((time.time()-t0)*1000)
                    if data.get("type")=="tts_chunk" and first_tts is None:
                        first_tts=time.time()
                        # also get latency field
                        e2e=int((first_tts - t0)*1000)
                        e2e_list.append(e2e)
                        print(f" iter {i+1}: text='{txt[:30]}' llm_ttft={llm_ttft} tts_first={(first_tts-t0)*1000:.0f}ms e2e={e2e}ms")
                        break
                    if data.get("type")=="latency":
                        # extra
                        pass
                    # timeout
                    if time.time()-t0>5:
                        print(f" iter {i+1} timeout")
                        break
            except Exception as e:
                print(f"ws iter {i} error {e}")
            await asyncio.sleep(0.1)

    poll_t.cancel()
    try:
        await poll_t
    except Exception:
        pass
    print("\n"+"="*68)
    print("  WS STREAMING — PEAK RSS & E2E")
    print("="*68)
    print(f"  Peak RSS: {peak:.1f} MB")
    if e2e_list:
        print(f"  E2E avg {statistics.mean(e2e_list):.0f} p50 {statistics.median(e2e_list):.0f} min {min(e2e_list)} max {max(e2e_list)} p95 {float(__import__('numpy').percentile(e2e_list,95)):.0f}")
        avg=statistics.mean(e2e_list)
        if avg<800:
            print(f"  ✅ PASSED <800ms (avg {avg:.0f}ms)")
        else:
            print(f"  ❌ avg {avg:.0f}ms")
    print("="*68)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--server", default="ws://localhost:8000/ws/chat")
    p.add_argument("--iters", type=int, default=5)
    args=p.parse_args()
    asyncio.run(test_ws(args.server, args.iters))
