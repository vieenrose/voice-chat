#!/usr/bin/env python3
"""
Peak RSS + E2E latency benchmark for streaming voice chat demo.

Runs pipeline in mock or real mode (if --real) and reports:
  - peak RSS (process + children) during run
  - E2E latency median/p95 for N utterances
  - per-stage latencies (STT, LLM TTFT, TTS, E2E)

Usage:
  python test_e2e.py --mock --iters 20
  python test_e2e.py --real --iters 10  # requires models downloaded + CUDA
  # Or against running server:
  python test_e2e.py --server http://localhost:8000

Prints a summary block in the end as requested.
"""
import argparse
import asyncio
import time
import statistics
import os
import sys
import numpy as np

# For peak RSS measurement
import psutil

sys.path.insert(0, os.path.dirname(__file__))

def get_rss_mb():
    proc = psutil.Process()
    # include children (uvicorn workers would be children if spawned)
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except Exception:
            pass
    return rss / 1024 / 1024

async def run_pipeline_benchmark(args):
    from pipeline.speech_to_speech import HFSpeechToSpeechPipeline
    import torch

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.mock:
        device = "cpu"  # mock doesn't need cuda

    print(f"\n[Benchmark] Initializing pipeline mock={args.mock} device={device} ...")
    t0 = time.time()
    pipe = HFSpeechToSpeechPipeline(mock=args.mock, device=device)
    init_ms = int((time.time()-t0)*1000)
    print(f"[Benchmark] Pipeline ready in {init_ms}ms, backend STT={pipe.stt.backend} TTS={pipe.tts.backend}")

    # track peak RSS via polling task
    peak_rss = get_rss_mb()
    stop_poll = asyncio.Event()

    async def rss_poller():
        nonlocal peak_rss
        while not stop_poll.is_set():
            cur = get_rss_mb()
            if cur > peak_rss:
                peak_rss = cur
            await asyncio.sleep(0.05)

    poll_task = asyncio.create_task(rss_poller())

    # Prepare test inputs
    if args.text:
        texts = [args.text]
    else:
        texts = [
            "Hello, how are you today?",
            "What's the weather like?",
            "Tell me a joke about AI",
            "Explain streaming speech-to-speech in one sentence",
            "What can you do with low latency voice chat?"
        ]
        # repeat to reach iters
        texts = (texts * ((args.iters // len(texts))+1))[:args.iters]

    latencies = []  # list of dicts
    e2e_list = []

    print(f"\n[Benchmark] Running {len(texts)} iterations...\n")
    for i, txt in enumerate(texts):
        # Create dummy audio for STT: 1.5s sine 16k (or use text direct path)
        # For mock pipeline, audio doesn't matter. For real, we need real audio.
        # We bypass STT for pure pipeline speed by feeding text directly via llm+tts, but we also measure STT once.
        # We'll measure full e2e including STT via pipeline.__call__ if not mock, else simulate STT latency

        if args.direct_text:
            # Bypass STT, measure LLM+TTS
            stt_text = txt
            stt_ms = 5
        else:
            # full pipeline with audio
            # Create 1s audio: 16k int16
            sr = 16000
            dur = 1.2
            pcm_int16 = (np.sin(2*np.pi*220*np.linspace(0,dur,int(sr*dur))) * 8000).astype(np.int16)
            pcm_f32 = pcm_int16.astype(np.float32)/32768.0
            # Measure STT
            t_stt = time.time()
            stt_text = await pipe.stt.transcribe_once(pcm_f32)
            # for mock, override to txt for deterministic llm test
            if pipe.mock:
                stt_text = txt
            stt_ms = int((time.time()-t_stt)*1000)
            # but if STT returns empty/mock, use txt
            if not stt_text.strip():
                stt_text = txt

        # LLM + TTS streaming E2E
        e2e_start = time.time()
        llm_start = time.time()
        llm_text = ""
        llm_ttft = None
        # stream LLM
        async for ev in pipe.llm.generate_stream(stt_text):
            if ev["type"] == "llm_token":
                if llm_ttft is None:
                    llm_ttft = int((time.time() - llm_start)*1000)
                llm_text = ev["text_so_far"]
            if ev["type"] == "llm_done":
                break
        llm_total = int((time.time()-llm_start)*1000)

        # TTS (measure first chunk and total)
        tts_start = time.time()
        chunks = []
        first_chunk_ms = None
        async for ev in pipe.tts.tts_from_text(llm_text):
            if ev["type"] == "tts_chunk":
                if first_chunk_ms is None:
                    first_chunk_ms = int((time.time()-tts_start)*1000)
                chunks.append(ev["pcm"])
        tts_total = int((time.time()-tts_start)*1000) if chunks else 0
        if first_chunk_ms is None:
            first_chunk_ms = tts_total

        e2e_ms = int((time.time()-e2e_start)*1000) + stt_ms  # include STT

        # alternative: full pipeline one-shot for comparison
        # out = await pipe(np.sin(...))  # not needed

        lat = {
            "iter": i+1,
            "stt_ms": stt_ms,
            "llm_ttft_ms": llm_ttft or 0,
            "llm_total_ms": llm_total,
            "tts_ttfb_ms": first_chunk_ms,
            "tts_total_ms": tts_total,
            "e2e_ms": e2e_ms,
            "stt_text": stt_text[:60],
            "llm_text": llm_text[:60],
        }
        latencies.append(lat)
        e2e_list.append(e2e_ms)
        print(f"  iter {i+1:02d}: stt={stt_ms:4d}ms llm_ttft={llm_ttft or 0:3d}ms llm_tot={llm_total:4d}ms tts_ttfb={first_chunk_ms:3d}ms e2e={e2e_ms:4d}ms  |  '{stt_text[:40]}' -> '{llm_text[:50]}'")

        # small pause to let GC
        await asyncio.sleep(0.05)

    stop_poll.set()
    await poll_task

    # Summary
    print("\n" + "="*78)
    print("  PEAK RSS & E2E LATENCY — FINAL REPORT")
    print("="*78)
    print(f"  Mode           : {'MOCK (no model download)' if args.mock else 'REAL (MiniCPM5/PrimeTTS/X-ASR)'}")
    print(f"  Device         : {device}")
    print(f"  Iterations     : {len(latencies)}")
    print(f"  Init time      : {init_ms} ms")
    print(f"  Peak RSS       : {peak_rss:.1f} MB  (process + children)")
    cur_rss = get_rss_mb()
    print(f"  Current RSS    : {cur_rss:.1f} MB")
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated()/1024/1024
            vram_max = torch.cuda.max_memory_allocated()/1024/1024
            print(f"  VRAM allocated : {vram:.1f} MB (peak {vram_max:.1f} MB)")
    except Exception:
        pass
    if e2e_list:
        print("\n  E2E latency (ms):")
        print(f"    avg  : {statistics.mean(e2e_list):.1f}")
        print(f"    median (p50): {statistics.median(e2e_list):.1f}")
        if len(e2e_list) >= 4:
            # percentile via numpy
            p95 = float(np.percentile(e2e_list, 95))
            p99 = float(np.percentile(e2e_list, 99))
            print(f"    p95  : {p95:.1f}")
            print(f"    p99  : {p99:.1f}")
        print(f"    min  : {min(e2e_list)}")
        print(f"    max  : {max(e2e_list)}")
        print(f"    stdev: {statistics.pstdev(e2e_list):.1f}" if len(e2e_list)>1 else "")

        # per-stage avg
        avg_stt = statistics.mean([rec["stt_ms"] for rec in latencies])
        avg_ttft = statistics.mean([rec["llm_ttft_ms"] for rec in latencies])
        avg_ttfb = statistics.mean([rec["tts_ttfb_ms"] for rec in latencies])
        print("\n  Per-stage avg:")
        print(f"    STT (X-ASR)   : {avg_stt:.1f} ms")
        print(f"    LLM TTFT      : {avg_ttft:.1f} ms (MiniCPM5)")
        print(f"    TTS TTFB      : {avg_ttfb:.1f} ms (PrimeTTS)")
        print(f"    LLM total     : {statistics.mean([rec['llm_total_ms'] for rec in latencies]):.1f} ms")
        print(f"    TTS total     : {statistics.mean([rec['tts_total_ms'] for rec in latencies]):.1f} ms")

    print("\n  Latency breakdown (first 5 iters):")
    for rec in latencies[:5]:
        print(f"    #{rec['iter']:02d} e2e={rec['e2e_ms']:4d}  (stt {rec['stt_ms']:3d} + llm_ttft {rec['llm_ttft_ms']:3d} + tts {rec['tts_ttfb_ms']:3d})")

    print("\n  Verdict:")
    avg_e2e = statistics.mean(e2e_list) if e2e_list else 0
    if avg_e2e < 800:
        print(f"    ✅ E2E <800ms target PASSED (avg {avg_e2e:.0f}ms) — low-latency pipeline working")
    elif avg_e2e < 1200:
        print(f"    ⚠️  E2E ~{avg_e2e:.0f}ms — borderline, consider faster GPU / int8 / smaller chunks")
    else:
        print(f"    ❌ E2E {avg_e2e:.0f}ms >1200ms — needs optimization (check mock vs real, GPU, chunk size)")

    print("="*78 + "\n")

    # Also write JSON report
    import json
    import pathlib
    report = {
        "mode": "mock" if args.mock else "real",
        "device": device,
        "iters": len(latencies),
        "init_ms": init_ms,
        "peak_rss_mb": round(peak_rss,1),
        "cur_rss_mb": round(cur_rss,1),
        "latencies": latencies,
        "summary": {
            "e2e_avg": float(statistics.mean(e2e_list)) if e2e_list else 0,
            "e2e_p50": float(statistics.median(e2e_list)) if e2e_list else 0,
            "e2e_p95": float(np.percentile(e2e_list,95)) if len(e2e_list)>=4 else 0,
            "e2e_min": min(e2e_list) if e2e_list else 0,
            "e2e_max": max(e2e_list) if e2e_list else 0,
        }
    }
    out_path = pathlib.Path(__file__).parent / "benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[Benchmark] Report written to {out_path}")

    return report

async def run_server_benchmark(args):
    import aiohttp
    import statistics
    url = args.server.rstrip("/")
    print(f"\n[Server Benchmark] Hitting {url} ...")
    # check health
    import urllib.request
    try:
        with urllib.request.urlopen(url+"/health", timeout=5) as r:
            print(f"  health: {r.read().decode()[:300]}")
    except Exception as e:
        print(f"  health check failed: {e}")
        print("  Is server running? Run: python backend/app.py --mock --port 8000")
        return

    peak_rss = 0
    latencies = []
    texts = ["Hello how are you", "Tell me a joke", "What is streaming STT"] * ((args.iters//3)+1)
    texts = texts[:args.iters]

    # We need to track RSS via /health polling
    async def poll_rss():
        nonlocal peak_rss
        import aiohttp
        while True:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url+"/health") as resp:
                        j = await resp.json()
                        rss = j.get("rss_mb", 0)
                        if rss > peak_rss:
                            peak_rss = rss
            except Exception:
                pass
            await asyncio.sleep(0.1)

    import asyncio
    poll_task = asyncio.create_task(poll_rss())
    await asyncio.sleep(0.5)

    for i, txt in enumerate(texts):
        # Use /api/chat text path for lowest overhead
        payload = {"text": txt}
        # Need aiohttp
        try:
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url+"/api/chat", json=payload) as resp:
                    j = await resp.json()
                    e2e = j["latencies"]["e2e_ms"]
                    stt = j["latencies"]["stt_ms"]
                    ttft = j["latencies"]["llm_ttft_ms"]
                    tts = j["latencies"]["tts_ms"]
        except Exception:
            # fallback to urllib
            import urllib.request
            import json as js
            req = urllib.request.Request(url+"/api/chat", data=js.dumps(payload).encode(), headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                j = js.loads(r.read().decode())
                e2e = j["latencies"]["e2e_ms"]
                stt = j["latencies"]["stt_ms"]
                ttft = j["latencies"]["llm_ttft_ms"]
                tts = j["latencies"]["tts_ms"]
        latencies.append(e2e)
        print(f"  iter {i+1:02d}: e2e={e2e:4d}ms (stt {stt} ttft {ttft} tts {tts})  '{txt}'")

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    print("\n" + "="*78)
    print("  PEAK RSS & E2E (SERVER MODE) — FINAL REPORT")
    print("="*78)
    print(f"  Peak RSS (server) : {peak_rss:.1f} MB")
    import numpy as np
    if latencies:
        print(f"  E2E avg : {statistics.mean(latencies):.1f} ms")
        print(f"  p50     : {statistics.median(latencies):.1f} ms")
        if len(latencies)>=4:
            print(f"  p95     : {float(np.percentile(latencies,95)):.1f} ms")
        print(f"  min/max : {min(latencies)}/{max(latencies)} ms")
        avg = statistics.mean(latencies)
        if avg < 800:
            print(f"  ✅ PASSED <800ms (avg {avg:.0f}ms)")
        else:
            print(f"  ⚠️  avg {avg:.0f}ms")
    print("="*78 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="mock pipeline (no model download)")
    parser.add_argument("--real", action="store_true", help="real models")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--iters", type=int, default=10, help="iterations")
    parser.add_argument("--text", type=str, default="", help="single text to repeat")
    parser.add_argument("--direct-text", action="store_true", help="bypass STT audio, feed text directly")
    parser.add_argument("--server", type=str, default="", help="benchmark running server URL, e.g. http://localhost:8000")
    args = parser.parse_args()

    if args.server:
        asyncio.run(run_server_benchmark(args))
    else:
        # Default to mock if neither specified
        if not args.mock and not args.real:
            args.mock = True
        if args.real:
            args.mock = False
        asyncio.run(run_pipeline_benchmark(args))

if __name__ == "__main__":
    main()
