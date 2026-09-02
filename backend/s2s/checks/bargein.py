"""Barge-in over a LONG answer, through the Realtime protocol.

    python -m s2s.checks.bargein [audio_deltas_before_speaking]

Starts a long spoken reply, then streams real mic audio the way a browser does.
A working barge-in cancels the in-flight response and stops its audio; the pass
condition is a response.done carrying status=cancelled reason=turn_detected,
with the audio deltas stopping rather than continuing.

Measured on Qwen3.5 4B: a 525-character answer (~40 s of speech) was cut 1.29 s
after the user started talking, which is Silero's detection window.
"""
import asyncio
import base64
import json
import sys
import time
import wave
import numpy as np
import websockets

URL = "ws://127.0.0.1:8765/v1/realtime"
WAV = "/home/user/voice-chat/asr_example.wav"
PROMPT = "請詳細說明台灣半導體產業的發展歷史、現況與未來挑戰，分成至少五個段落詳細回答。"
FIRE_AFTER = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # audio deltas before speaking


def pcm16_16k():
    with wave.open(WAV, "rb") as w:
        sr, raw = w.getframerate(), w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
        a = a[np.clip(idx, 0, len(a) - 1)]
    return a


async def main():
    pcm = pcm16_16k()
    t0 = time.time()
    before = after = 0
    fired = cancelled_at = None
    timeline = []

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": "你是一個親切的語音助理，請用繁體中文詳細回答。",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}}},
        }}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": PROMPT}]}}))
        await ws.send(json.dumps({"type": "response.create"}))

        async def speak():
            step = int(16000 * 0.16)          # 160 ms frames, as a browser sends
            for i in range(0, len(pcm), step):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i+step].tobytes()).decode()}))
                await asyncio.sleep(0.16)

        speaker = None
        deadline = time.time() + 75
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
            except asyncio.TimeoutError:
                break
            d = json.loads(raw) if isinstance(raw, str) else {"type": "<bin>"}
            t = d.get("type", "")
            if t == "response.output_audio.delta":
                if fired is None:
                    before += 1
                    if before == FIRE_AFTER:
                        fired = time.time()
                        print(f"  -> user starts speaking after {before} audio deltas "
                              f"({fired-t0:.1f}s in)")
                        speaker = asyncio.create_task(speak())
                else:
                    after += 1
            elif fired is not None:
                resp = d.get("response") or {}
                extra = " ".join(str(x) for x in
                                 (resp.get("status"), (resp.get("status_details") or {}).get("reason"),
                                  d.get("reason")) if x)
                timeline.append((round(time.time()-fired, 2), t, extra[:40]))
                if t == "response.done":
                    cancelled_at = cancelled_at or time.time()
                    # keep listening: speech_started / a new response follow
                    deadline = min(deadline, time.time() + 8)
        if speaker and not speaker.done():
            speaker.cancel()

    print(f"\n  audio deltas before speech : {before}")
    print(f"  audio deltas after speech  : {after}   (detection window; must stop, not grow)")
    print("  events after speech began:")
    for dt, t, extra in timeline[:14]:
        print(f"    +{dt:5.2f}s  {t:<46} {extra}")
    cut = next((dt for dt, t, _ in timeline if t == "input_audio_buffer.speech_started"), None)
    dn = next(((dt, x) for dt, t, x in timeline if t == "response.done"), None)
    print(f"\n  VERDICT: speech_started {'at +%.2fs' % cut if cut is not None else 'NEVER FIRED'}"
          f" | response.done {('at +%.2fs (%s)' % (dn[0], dn[1] or 'no status')) if dn else 'none'}")

asyncio.run(main())
