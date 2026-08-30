#!/usr/bin/env python3
"""
End-to-end test suite against the LIVE running backend (real models, no mocks).
Exercises the actual deployed stack over HTTP/WS exactly as a client would,
and reports:
  - Accuracy: STT transcription (vs. known ground truth for asr_example.wav)
              and tool-call selection (does the agent call the right tool,
              or correctly call none, for each labeled query)
  - Performance: STT/LLM-TTFT/TTS-TTFB/E2E latency percentiles
  - Peak RSS: polled continuously via /health for the whole run
  - Barge-in correctness: text, voice, and cross-channel supersession

Requires the backend already running (see README Quick Start) at --server.
Writes a JSON summary to backend/benchmark_report.json.

Usage:
  python test_e2e_report.py --server http://127.0.0.1:8000
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import websockets

WAV_PATH = Path(__file__).resolve().parent.parent / "asr_example.wav"
# Full, correct transcript of asr_example.wav, including a quietly-spoken leading
# word ("欢迎" / "welcome") — RMS ~14-94 in the first 0.9s vs ~5000-10000 for the
# main phrase starting at 0.9s. Confirmed by direct waveform inspection: this lead-in
# IS real speech, not silence/noise. It's occasionally trimmed by endpoint/VAD
# sensitivity (most consistently on the first call after a fresh backend start,
# though not deterministically so) — see the "leading_word_trimmed" accuracy metric
# below, which measures this specific, real characteristic separately from actual
# transcription errors elsewhere in the phrase.
STT_GROUND_TRUTH = "欢迎大家来体验达摩院推出的语音识别模型"
STT_GROUND_TRUTH_NO_LEADIN = "大家来体验达摩院推出的语音识别模型"

# Per-invocation unique suffix for session_ids — the backend's turn_id counter is
# scoped per session_id and persists for the lifetime of the (long-running) backend
# process, not per test run. A fixed session_id reused across repeated invocations of
# this suite would accumulate turn_ids from every previous run, breaking any check
# that assumes a fresh turn_id sequence (e.g. "the second turn this run has turn_id 2").
RUN_ID = str(int(time.time()))

TOOL_QUERIES = [
    # (query, expected_tool or None for "no tool expected")
    ("Hello, how are you today?", None),
    ("Hi there!", None),
    # The agent's system prompt (backend/agent/qwen_harness.py) explicitly instructs
    # get_weather for weather queries — not web_search — so that's the correct
    # expectation, not "any search-shaped tool".
    ("What is the weather in Tokyo today?", "get_weather"),
    ("Search for the latest news about artificial intelligence", "web_search"),
    ("Who is the president of France?", "web_search"),
    ("今天是星期幾？", "get_current_datetime"),
    ("What time is it right now?", "get_current_datetime"),
]


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate via Levenshtein edit distance / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(hyp), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev if hyp[i - 1] == ref[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n] / n


def load_pcm() -> np.ndarray:
    w = wave.open(str(WAV_PATH), "rb")
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def frame(pcm_i16: np.ndarray) -> bytes:
    return b"\x01" + pcm_i16.tobytes()


async def stream_pcm(ws, pcm: np.ndarray, chunk=320, realtime=True):
    for i in range(0, len(pcm), chunk):
        c = pcm[i:i + chunk]
        if len(c) < chunk:
            c = np.pad(c, (0, chunk - len(c)))
        await ws.send(frame(c))
        if realtime:
            await asyncio.sleep(chunk / 16000)


async def stream_silence(ws, seconds, chunk=320, sr=16000, realtime=True):
    n = int(seconds * sr)
    zeros = np.zeros(chunk, dtype=np.int16)
    sent = 0
    while sent < n:
        await ws.send(frame(zeros))
        sent += chunk
        if realtime:
            await asyncio.sleep(chunk / sr)


class RSSPoller:
    """Polls /health continuously in the background and tracks peak RSS for
    the whole test run — a single point-in-time reading would miss transient
    spikes during concurrent GPU work (LLM+TTS overlapping across requests)."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.peak_mb = 0.0
        self._task = None
        self._stop = asyncio.Event()

    async def _poll(self):
        async with httpx.AsyncClient(timeout=2.0) as client:
            while not self._stop.is_set():
                try:
                    r = await client.get(f"{self.base_url}/health")
                    rss = r.json().get("rss_mb", 0)
                    if rss > self.peak_mb:
                        self.peak_mb = rss
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass

    def start(self):
        self._task = asyncio.create_task(self._poll())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task


async def test_stt_accuracy(ws_base: str, repeats: int = 3) -> dict:
    print("\n=== STT ACCURACY (real audio, X-ASR sherpa-onnx int8) ===")
    pcm = load_pcm()
    results = []
    for i in range(repeats):
        url = f"{ws_base}/ws/chat?session_id=e2e_stt_{RUN_ID}_{i}"
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps({"type": "start"}))
            transcript = None
            latency_ms = None
            t0 = time.time()

            async def recv():
                nonlocal transcript, latency_ms
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "stt_final" and transcript is None:
                            transcript = msg.get("text", "")
                            latency_ms = msg.get("latency_ms", 0)
                            return
                except Exception as e:
                    # Without this, a single bad message silently kills this task and
                    # every subsequent event is missed with no trace — the outer
                    # asyncio.wait_for(recv_task, ...) only ever sees a CancelledError
                    # from its own timeout, never a hint that recv() died early.
                    print(f"  [recv() error, event capture stopped: {e!r}]")

            recv_task = asyncio.create_task(recv())
            await stream_pcm(ws, pcm)
            # 2.0s of trailing silence — comfortably past the recognizer's own
            # rule1_min_trailing_silence=1.2s endpoint threshold (backend/stt/
            # xasr_streaming.py) with margin for scheduling jitter under GPU load;
            # 1.6s was found to be marginal (intermittently missed the endpoint).
            await stream_silence(ws, 2.0)
            try:
                await asyncio.wait_for(recv_task, timeout=15)
            except asyncio.TimeoutError:
                pass
            elapsed = int((time.time() - t0) * 1000)
            t = (transcript or "").strip()
            # Accept either the full phrase or the phrase with its quiet leading word
            # trimmed as "correct content" — the trim is a real, separately-reported
            # characteristic (see leading_word_trimmed below), not a transcription error.
            content_correct = t in (STT_GROUND_TRUTH, STT_GROUND_TRUTH_NO_LEADIN)
            error = min(cer(t, STT_GROUND_TRUTH), cer(t, STT_GROUND_TRUTH_NO_LEADIN))
            trimmed = t == STT_GROUND_TRUTH_NO_LEADIN
            results.append({"run": i, "transcript": transcript, "content_correct": content_correct, "leading_word_trimmed": trimmed, "cer": error, "wall_ms": elapsed, "stt_latency_ms": latency_ms})
            print(f"  run {i+1}/{repeats}: content_correct={content_correct} trimmed_leadin={trimmed} cer={error:.3f} transcript={transcript!r}")

    content_rate = sum(1 for r in results if r["content_correct"]) / len(results)
    trim_rate = sum(1 for r in results if r["leading_word_trimmed"]) / len(results)
    avg_cer = statistics.mean(r["cer"] for r in results)
    print(f"  -> content_accuracy={content_rate:.0%} avg_cer={avg_cer:.3f} leading_word_trimmed_rate={trim_rate:.0%}")
    return {"ground_truth": STT_GROUND_TRUTH, "runs": results, "content_accuracy": content_rate, "avg_cer": avg_cer, "leading_word_trimmed_rate": trim_rate}


async def test_tool_accuracy(ws_base: str) -> dict:
    print("\n=== TOOL-CALL ACCURACY (text input) ===")
    results = []
    for i, (query, expected_tool) in enumerate(TOOL_QUERIES):
        url = f"{ws_base}/ws/chat?session_id=e2e_tool_{RUN_ID}_{i}"
        async with websockets.connect(url, max_size=None) as ws:
            called_tool = None
            latency = {}
            t0 = time.time()

            async def recv():
                nonlocal called_tool, latency
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "tool_call" and called_tool is None:
                            called_tool = msg.get("name")
                        if msg.get("type") == "latency":
                            latency = msg
                        if msg.get("type") == "tts_end":
                            return
                except Exception as e:
                    print(f"  [recv() error, event capture stopped: {e!r}]")

            recv_task = asyncio.create_task(recv())
            await ws.send(json.dumps({"type": "text_input", "text": query}))
            try:
                await asyncio.wait_for(recv_task, timeout=25)
            except asyncio.TimeoutError:
                pass
            elapsed = int((time.time() - t0) * 1000)
            correct = called_tool == expected_tool
            results.append({
                "query": query, "expected_tool": expected_tool, "called_tool": called_tool,
                "correct": correct, "wall_ms": elapsed,
                "llm_ttft_ms": latency.get("llm_ttft_ms"), "tts_ttfb_ms": latency.get("tts_ttfb_ms"),
                "e2e_ms": latency.get("e2e_ms"),
            })
            print(f"  '{query[:45]:45}' expected={str(expected_tool):20} got={str(called_tool):20} {'OK' if correct else 'MISS'} e2e={latency.get('e2e_ms')}")

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"  -> tool_selection_accuracy={accuracy:.0%} ({sum(1 for r in results if r['correct'])}/{len(results)})")
    return {"queries": results, "accuracy": accuracy}


async def test_barge_in_text(ws_base: str) -> dict:
    print("\n=== BARGE-IN CORRECTNESS: text_input interrupts text_input ===")
    url = f"{ws_base}/ws/chat?session_id=e2e_barge_text_{RUN_ID}"
    events = []
    async with websockets.connect(url, max_size=None) as ws:
        async def recv():
            try:
                async for raw in ws:
                    events.append((time.time(), json.loads(raw)))
            except Exception as e:
                print(f"  [recv() error, event capture stopped: {e!r}]")
        recv_task = asyncio.create_task(recv())
        await ws.send(json.dumps({"type": "text_input", "text": "Tell me a long story about a dragon, at least eight sentences, in English."}))
        deadline = time.time() + 20
        while time.time() < deadline and not any(e[1].get("type") == "tts_chunk" for e in events):
            await asyncio.sleep(0.1)
        turn1_id = next((e[1].get("turn_id") for e in events if e[1].get("type") == "tts_chunk"), None)
        t_barge = time.time()
        await ws.send(json.dumps({"type": "text_input", "text": "Just say OK."}))
        # Poll with a generous deadline rather than a fixed sleep — LLM/TTS latency
        # varies with concurrent GPU load (seen up to ~14s for a trivial query
        # elsewhere in this same suite), so a short fixed wait can time out on a
        # legitimately-still-arriving reply and register a false failure.
        deadline2 = time.time() + 25
        # turn1_id + 1, not a hardcoded 2 — defense in depth alongside the RUN_ID
        # suffix above, in case this session_id is ever reused across invocations
        # after all (e.g. copy-pasted without the suffix).
        while time.time() < deadline2 and not any(e[1].get("type") == "tts_chunk" and e[1].get("turn_id") == (turn1_id or 0) + 1 for e in events):
            await asyncio.sleep(0.2)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
    late_turn1 = [e for e in events if e[1].get("type") == "tts_chunk" and e[1].get("turn_id") == turn1_id and e[0] > t_barge + 0.05] if turn1_id else []
    got_new_turn_audio = any(e[1].get("type") == "tts_chunk" and e[1].get("turn_id") == (turn1_id or 0) + 1 for e in events)
    passed = len(late_turn1) == 0 and got_new_turn_audio
    print(f"  stray chunks from old turn after barge-in: {len(late_turn1)} | new turn produced audio: {got_new_turn_audio} -> {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "stray_chunks": len(late_turn1), "new_turn_audio": got_new_turn_audio}


async def test_barge_in_voice(ws_base: str) -> dict:
    print("\n=== BARGE-IN CORRECTNESS: voice interrupts voice ===")
    pcm = load_pcm()
    url = f"{ws_base}/ws/chat?session_id=e2e_barge_voice_{RUN_ID}"
    events = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))
        async def recv():
            try:
                async for raw in ws:
                    events.append((time.time(), json.loads(raw)))
            except Exception as e:
                print(f"  [recv() error, event capture stopped: {e!r}]")
        recv_task = asyncio.create_task(recv())
        await stream_pcm(ws, pcm)
        await stream_silence(ws, 1.6)
        deadline = time.time() + 20
        while time.time() < deadline and not any(e[1].get("type") == "tts_chunk" for e in events):
            await asyncio.sleep(0.1)
        # Start streaming the interrupting utterance now, but the barge-in can only take
        # effect once STT actually recognizes it (~7s of real-time-paced audio+silence
        # later) — the first turn legitimately keeps playing until then, so staleness
        # must be measured from the real barge_in event, not from when we start sending.
        await stream_pcm(ws, pcm)
        await stream_silence(ws, 1.6)
        await asyncio.sleep(12)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
    barge_ins = [e for e in events if e[1].get("type") == "barge_in" and e[1].get("reason") == "voice"]
    first_turn_id = next((e[1].get("turn_id") for e in events if e[1].get("type") == "tts_chunk"), None)
    t_barge = barge_ins[0][0] if barge_ins else None
    late_first_turn = [e for e in events if e[1].get("type") == "tts_chunk" and e[1].get("turn_id") == first_turn_id and e[0] > t_barge + 0.05] if (first_turn_id and t_barge) else []
    passed = len(barge_ins) >= 1 and len(late_first_turn) == 0
    print(f"  barge_in events: {len(barge_ins)} | stray chunks from interrupted turn: {len(late_first_turn)} -> {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "barge_in_events": len(barge_ins), "stray_chunks": len(late_first_turn)}


async def test_cross_channel(ws_base: str) -> dict:
    print("\n=== BARGE-IN CORRECTNESS: voice supersedes an in-flight text reply ===")
    pcm = load_pcm()
    url = f"{ws_base}/ws/chat?session_id=e2e_barge_cross_{RUN_ID}"
    events = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))
        async def recv():
            try:
                async for raw in ws:
                    events.append((time.time(), json.loads(raw)))
            except Exception as e:
                print(f"  [recv() error, event capture stopped: {e!r}]")
        recv_task = asyncio.create_task(recv())
        await ws.send(json.dumps({"type": "text_input", "text": "Tell me a long story about a robot, at least eight sentences, in English."}))
        await asyncio.sleep(0.15)
        t_voice = time.time()
        await stream_pcm(ws, pcm)
        await stream_silence(ws, 1.6)
        await asyncio.sleep(15)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
    text_turn_id = next((e[1].get("turn_id") for e in events if e[1].get("type") == "tts_chunk"), None)
    stt_finals = [e for e in events if e[1].get("type") == "stt_final"]
    late_text_chunks = []
    if text_turn_id is not None and stt_finals:
        t_stt = stt_finals[0][0]
        late_text_chunks = [e for e in events if e[1].get("type") == "tts_chunk" and e[1].get("turn_id") == text_turn_id and e[0] > t_stt + 0.1]
    passed = len(late_text_chunks) == 0
    print(f"  stray text-turn chunks after voice superseded it: {len(late_text_chunks)} -> {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "stray_chunks": len(late_text_chunks)}


def summarize_latencies(tool_results: list) -> dict:
    def pct(vals, p):
        return float(np.percentile(vals, p)) if vals else 0.0
    e2e = [r["e2e_ms"] for r in tool_results if r.get("e2e_ms")]
    ttft = [r["llm_ttft_ms"] for r in tool_results if r.get("llm_ttft_ms")]
    ttfb = [r["tts_ttfb_ms"] for r in tool_results if r.get("tts_ttfb_ms")]
    return {
        "e2e_ms": {"avg": statistics.mean(e2e) if e2e else 0, "p50": pct(e2e, 50), "p95": pct(e2e, 95), "min": min(e2e) if e2e else 0, "max": max(e2e) if e2e else 0},
        "llm_ttft_ms": {"avg": statistics.mean(ttft) if ttft else 0, "p50": pct(ttft, 50)},
        "tts_ttfb_ms": {"avg": statistics.mean(ttfb) if ttfb else 0, "p50": pct(ttfb, 50)},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--stt-repeats", type=int, default=5)
    args = ap.parse_args()

    base = args.server.rstrip("/")
    ws_base = "ws://" + base.split("://", 1)[1]

    print("=" * 78)
    print(" E2E TEST SUITE — real models, live backend, no mocks")
    print("=" * 78)

    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{base}/health")
        health = r.json()
    print(f"Backend: {health['models_loaded']} | LLM={health.get('llm_manager', {}).get('current_model_id')} | rss={health['rss_mb']}MB")
    if health.get("mock"):
        print("!!! WARNING: backend is running in --mock mode, results will not reflect real model accuracy/performance")

    poller = RSSPoller(base)
    poller.start()
    init_rss = health["rss_mb"]
    t_start = time.time()

    stt_report = await test_stt_accuracy(ws_base, repeats=args.stt_repeats)
    tool_report = await test_tool_accuracy(ws_base)
    barge_text = await test_barge_in_text(ws_base)
    barge_voice = await test_barge_in_voice(ws_base)
    barge_cross = await test_cross_channel(ws_base)

    await poller.stop()
    took_s = time.time() - t_start

    perf = summarize_latencies(tool_report["queries"])
    barge_pass_rate = sum(1 for b in (barge_text, barge_voice, barge_cross) if b["passed"]) / 3

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  ACCURACY")
    print(f"    STT content accuracy : {stt_report['content_accuracy']:.0%}  (avg CER {stt_report['avg_cer']:.3f}, leading-word trimmed {stt_report['leading_word_trimmed_rate']:.0%}, n={args.stt_repeats})")
    print(f"    Tool-call accuracy   : {tool_report['accuracy']:.0%}  ({sum(1 for r in tool_report['queries'] if r['correct'])}/{len(tool_report['queries'])})")
    print(f"    Barge-in pass rate   : {barge_pass_rate:.0%}  (text={barge_text['passed']}, voice={barge_voice['passed']}, cross={barge_cross['passed']})")
    print(f"  PERFORMANCE (tool-accuracy queries, n={len(tool_report['queries'])})")
    print(f"    E2E       avg={perf['e2e_ms']['avg']:.0f}ms  p50={perf['e2e_ms']['p50']:.0f}ms  p95={perf['e2e_ms']['p95']:.0f}ms  min={perf['e2e_ms']['min']:.0f}  max={perf['e2e_ms']['max']:.0f}")
    print(f"    LLM TTFT  avg={perf['llm_ttft_ms']['avg']:.0f}ms  p50={perf['llm_ttft_ms']['p50']:.0f}ms")
    print(f"    TTS TTFB  avg={perf['tts_ttfb_ms']['avg']:.0f}ms  p50={perf['tts_ttfb_ms']['p50']:.0f}ms")
    print(f"  MEMORY")
    print(f"    Init RSS  : {init_rss:.1f} MB")
    print(f"    Peak RSS  : {poller.peak_mb:.1f} MB")
    print(f"  Total test time: {took_s:.1f}s")
    print("=" * 78)

    report = {
        "backend": health,
        "accuracy": {
            "stt_content_accuracy": stt_report["content_accuracy"],
            "stt_avg_cer": stt_report["avg_cer"],
            "stt_leading_word_trimmed_rate": stt_report["leading_word_trimmed_rate"],
            "tool_call_accuracy": tool_report["accuracy"],
            "barge_in_pass_rate": barge_pass_rate,
        },
        "performance": perf,
        "memory": {"init_rss_mb": init_rss, "peak_rss_mb": poller.peak_mb},
        "detail": {
            "stt": stt_report, "tool": tool_report,
            "barge_in_text": barge_text, "barge_in_voice": barge_voice, "barge_in_cross": barge_cross,
        },
        "took_s": took_s,
    }
    out_path = Path(__file__).resolve().parent / "benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
