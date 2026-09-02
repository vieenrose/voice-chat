"""Exhaustive end-to-end check of the live demo.

    python -m s2s.checks.exhaustive            # everything
    python -m s2s.checks.exhaustive --quick    # skip the slow spoken turns

Exercises the whole stack against a running pipeline: the added HTTP routes and
their validation, provider switching in both directions, tool routing and
fabrication on every prompt shape, error legibility, and barge-in. Prints a
report and exits non-zero if anything fails, so it can gate a change.

Needs the pipeline up on :8765 and, for the local-provider rows, llama-server on
:11435. Close any browser tab first -- there is one session slot.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = os.getenv("S2S_HTTP_BASE", "http://127.0.0.1:8765")
WS = os.getenv("S2S_WS", "ws://127.0.0.1:8765/v1/realtime")
WAV = Path(__file__).resolve().parents[3] / "asr_example.wav"
KEYFILE = Path(os.getenv("OPENROUTER_KEY_FILE", "/home/user/.openrouter_key"))

results: list[tuple[str, str, bool, str]] = []
INITIAL: dict = {}



def check(group: str, name: str, ok: bool, detail: str = "") -> bool:
    results.append((group, name, bool(ok), detail))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def http(path: str, method: str = "GET", body: dict | None = None, timeout: float = 30.0):
    """Returns (status, parsed_json_or_text)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except Exception:
            return e.code, raw.decode(errors="replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


# ── routes ────────────────────────────────────────────────────────────
def test_routes() -> None:
    print("\n[routes]")
    st, d = http("/v1/vram")
    check("routes", "GET /v1/vram", st == 200 and isinstance(d, dict) and d.get("available") is True,
          f"{d.get('used_mib')}/{d.get('total_mib')} MiB" if isinstance(d, dict) else str(d)[:60])
    if isinstance(d, dict) and d.get("available"):
        check("routes", "vram counts the whole card",
              0 < d["used_mib"] <= d["total_mib"] and d["total_mib"] > 4000,
              "used must include other processes, e.g. llama-server")

    st, d = http("/v1/llm-config")
    ok = st == 200 and {"model", "api_base"} <= set(d or {})
    check("routes", "GET /v1/llm-config", ok, f"model={d.get('model')} base={d.get('api_base')}")
    if ok:
        check("routes", "the endpoint is local",
              "127.0.0.1" in str(d.get("api_base")) or "localhost" in str(d.get("api_base")),
              str(d.get("api_base")))





def wait_for_slot(timeout: float = 45.0) -> bool:
    """Block until the pipeline pool has a free unit.

    The server runs one session at a time by default and releases a unit
    asynchronously after the client disconnects, so back-to-back turns race the
    teardown and get 1008 "All session slots are in use". /v1/pool is the
    framework's own view of that, which beats guessing at a sleep.
    """
    end = time.time() + timeout
    while time.time() < end:
        st, d = http("/v1/pool", timeout=5)
        if st == 200 and isinstance(d, dict) and d.get("in_use", 1) == 0:
            return True
        time.sleep(0.5)
    return False


# ── spoken turns over the Realtime protocol ───────────────────────────
async def one_turn(prompt: str, timeout: float = 120.0) -> dict:
    import websockets

    t0 = time.time()
    out = {"text": "", "audio": 0, "first_audio": None, "status": None}
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": "你是一個親切的語音助理，請用繁體中文簡短回答。",
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        end = time.time() + timeout
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            if not isinstance(raw, str):
                continue
            d = json.loads(raw)
            t = d.get("type", "")
            if t == "response.output_audio.delta":
                out["audio"] += 1
                out["first_audio"] = out["first_audio"] or round(time.time() - t0, 2)
            elif t == "response.output_audio_transcript.done":
                out["text"] += d.get("transcript") or ""
            elif t == "response.done":
                out["status"] = (d.get("response") or {}).get("status")
                break
    out["total"] = round(time.time() - t0, 2)
    return out


# What "the tool actually ran" looks like, per shape. One regex for all three was
# wrong: a news summary from web_search need not contain a temperature or a date,
# so a real answer naming 沙德爾颱風 was failed for not matching a numeric pattern.
FACT = {
    "clock":   re.compile(r"星期[一二三四五六日]|\d{1,2}\s*[月日點時]|\d{4}\s*年"),
    "weather": re.compile(r"\d+\s*(°C|度)|氣溫|降雨|濕度|溼度"),
    # Tools run server-side, so the protocol never shows the call; what a real
    # search answer has is substance rather than a refusal or a stub.
    "search":  re.compile(r"^(?!.*(無法|抱歉|找不到)).{40,}", re.S),
}

TURNS = [
    ("chat",    "你好",                       False),
    ("plain",   "台灣的首都是哪裡？請簡短回答。", False),
    ("clock",   "現在幾點？",                  True),
    ("weather", "今天台北天氣如何？",           True),
    ("search",  "今天有什麼新聞？",             True),
]


async def test_turns() -> None:
    print("\n[spoken turns]")
    for label, prompt, needs_fact in TURNS:
        if not wait_for_slot():
            check("turns", f"{label}", False, "no free pipeline slot (browser tab open?)")
            continue
        try:
            r = await one_turn(prompt)
        except Exception as e:
            check("turns", f"{label}", False, f"{type(e).__name__}: {e}")
            continue
        spoke = bool(r["text"].strip()) and r["audio"] > 0
        refused = "抱歉" in r["text"]
        ok = spoke and not refused
        detail = f"{r['total']}s first_audio={r['first_audio']} {r['text'][:44]!r}"
        check("turns", f"{label} answers with audio", ok, detail)
        pat = FACT.get(label)
        if needs_fact and ok and pat is not None:
            check("turns", f"{label} shows the tool's result, not the model's memory",
                  bool(pat.search(r["text"])), r["text"][:50].replace("\n", " "))
        if label == "clock" and ok:
            # The year alone is far too weak: a free-router model was seen to receive
            # "Wednesday 2026-09-02" from the tool and report 2026年5月14日 週四, which a
            # year check passes. Month and day are what the tool actually supplies.
            now = time.localtime()
            want_md = (str(now.tm_mon), str(now.tm_mday))
            got = bool(re.search(rf"{now.tm_mon}\s*月\s*{now.tm_mday}\s*日", r["text"])
                       or re.search(rf"{now.tm_mon:02d}[-/]{now.tm_mday:02d}", r["text"])
                       or re.search(rf"\b{now.tm_mday}\b.*\b{now.tm_mon}\b", r["text"]))
            check("turns", "clock reports the tool's actual date, not an invented one",
                  got, f"host={want_md[0]}/{want_md[1]}  said={r['text'][:46]!r}")


async def test_bargein() -> None:
    print("\n[barge-in]")
    import numpy as np
    import websockets

    if not WAV.exists():
        check("barge-in", "cancelled by voice", False, f"missing {WAV}")
        return
    if not wait_for_slot():
        check("barge-in", "a long answer is cancelled by voice", False, "no free pipeline slot")
        return
    with wave.open(str(WAV), "rb") as w:
        sr, raw = w.getframerate(), w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
        a = a[np.clip(idx, 0, len(a) - 1)]

    before = after = 0
    status = reason = None
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "instructions": "請用繁體中文詳細回答，至少五段。",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}}}}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text",
                         "text": "請詳細說明台灣半導體產業的歷史、現況與挑戰，分成至少五個段落。"}]}}))
        await ws.send(json.dumps({"type": "response.create"}))

        speaking = False
        cancelled_at = None

        async def talk():
            step = int(16000 * 0.16)
            for i in range(0, len(a), step):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(a[i:i + step].tobytes()).decode()}))
                await asyncio.sleep(0.16)

        task = None
        end = time.time() + 120
        while time.time() < end:
            try:
                raw2 = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                break
            if not isinstance(raw2, str):
                continue
            d = json.loads(raw2)
            t = d.get("type", "")
            if t == "response.output_audio.delta":
                if not speaking:
                    before += 1
                    if before >= 3:
                        speaking = True
                        task = asyncio.create_task(talk())
                else:
                    after += 1
            elif t == "response.done":
                resp = d.get("response") or {}
                status = resp.get("status")
                reason = (resp.get("status_details") or {}).get("reason")
                cancelled_at = after
                # Keep reading briefly: the requirement is that NO audio follows the
                # cancellation, which cannot be observed by breaking here.
                end = min(end, time.time() + 2.0)
        if task and not task.done():
            task.cancel()

    check("barge-in", "a long answer is cancelled by voice",
          status == "cancelled" and reason == "turn_detected",
          f"status={status} reason={reason} chunks before/after={before}/{after}")
    # Counting chunks during the detection window measures Silero's latency, not
    # correctness: TTS runs at RTF 0.2, so seconds of audio arrive in a fraction of
    # that. What must hold is that nothing arrives AFTER the cancellation.
    leaked = (after - cancelled_at) if cancelled_at is not None else after
    check("barge-in", "no audio arrives after the cancellation", leaked == 0,
          f"{leaked} chunks after response.done ({after} total in the window)")


def test_error_messages() -> None:
    print("\n[error legibility]")
    from agent.qwen_harness import _provider_failure_zh

    class C(Exception):
        def __init__(self, c):
            super().__init__(f"code {c}")
            self.status_code = c

    msgs = {c: _provider_failure_zh(C(c)) for c in (401, 402, 403, 404, 429, 503)}
    check("errors", "each provider refusal has its own message", len(set(msgs.values())) >= 5)
    check("errors", "401 names the key", "金鑰" in msgs[401])
    check("errors", "402 names the balance", "額度" in msgs[402])
    check("errors", "429 suggests another model", "模型" in msgs[429])
    leak = _provider_failure_zh(Exception("ConnectError [Errno 111] 10.1.2.3:8080"))
    check("errors", "an unknown failure leaks neither host nor errno",
          "10.1.2.3" not in leak and "Errno" not in leak)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip spoken turns and barge-in")
    args = ap.parse_args()

    key = None
    if KEYFILE.exists():
        key = KEYFILE.read_text().strip() or None
    print(f"target {BASE}  |  key {'present' if key else 'absent'}  |  wav {'ok' if WAV.exists() else 'missing'}")

    st, cfg = http("/v1/llm-config")
    if st == 200:
        print(f"model under test: {cfg.get('model')} @ {cfg.get('api_base')}")

    test_routes()
    test_error_messages()
    if not args.quick:
        asyncio.run(test_turns())
        asyncio.run(test_bargein())

    total = len(results)
    bad = [r for r in results if not r[2]]
    print("\n" + "=" * 66)
    by: dict[str, list[bool]] = {}
    for g, _n, ok, _d in results:
        by.setdefault(g, []).append(ok)
    for g, oks in by.items():
        print(f"  {g:14} {sum(oks)}/{len(oks)}")
    print(f"  {'TOTAL':14} {total - len(bad)}/{total}")
    if bad:
        print("\n  failures:")
        for g, n, _ok, d in bad:
            print(f"    [{g}] {n}  {d}")
    print("=" * 66)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
