"""Voice barge-in over a LONG reply, against a running backend.

Run:  python3 backend/test_bargein_voice.py [chunks_before_speaking]

Regression cover for the case where speaking over a *typed* reply did nothing until
the user stopped talking: the cross-channel supersede was wired to stt_final only,
so on a long answer the assistant talked over the user for the whole utterance.
Voice-over-voice reacted to the first partial; voice-over-text now does too.

A pass looks like: barge_in fires ~2s in (streaming-ASR detection latency, the
earliest a partial with text exists), and every chunk after it carries the NEW
turn_id. Any chunk still arriving under the interrupted turn_id is the bug.

Starts a long text_input reply, waits until it is well underway, then streams real
mic audio (asr_example.wav) as 0x01 PCM frames the way the browser does. A working
barge-in should cancel the in-flight reply and stop sending its audio.
"""
import asyncio
import json
import os
import sys
import time
import uuid
import wave
import websockets
import numpy as np

WS = "ws://127.0.0.1:8000/ws/chat"
WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "asr_example.wav")
PROMPT = "請詳細說明台灣半導體產業的發展歷史、現況與未來挑戰，並分成至少五個段落詳細回答。"
FIRE_AFTER = int(sys.argv[1]) if len(sys.argv) > 1 else 25   # chunks before speaking


def load_pcm16():
    with wave.open(WAV, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:                                  # resample to 16k
        idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
        a = a[np.clip(idx, 0, len(a) - 1)]
    return a


async def main():
    pcm = load_pcm16()
    sid = f"bv-{uuid.uuid4().hex[:6]}"
    before = after = post_barge = 0
    post_ids = {}
    turn_ids_before = set()
    barge_at = None
    fired = None
    events_after = []
    t0 = time.time()

    async with websockets.connect(f"{WS}?session_id={sid}", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))
        await ws.send(json.dumps({"type": "text_input", "text": PROMPT}))

        async def speak():
            """Stream the wav as 160 ms mic frames, as the browser does."""
            step = int(16000 * 0.16)
            for i in range(0, len(pcm), step):
                await ws.send(b"\x01" + pcm[i:i + step].tobytes())
                await asyncio.sleep(0.16)
            await ws.send(b"\x02")                    # flush

        speaker = None
        while True:
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=45)
            except asyncio.TimeoutError:
                break
            if isinstance(m, bytes):
                continue
            d = json.loads(m)
            t = d.get("type")
            if t == "tts_chunk":
                if fired is None:
                    before += 1
                    turn_ids_before.add(d.get('turn_id'))
                    if before == FIRE_AFTER:
                        fired = time.time()
                        print(f"  -> start speaking after {before} chunks ({fired-t0:.1f}s in)")
                        speaker = asyncio.create_task(speak())
                else:
                    after += 1
                    if barge_at is not None:
                        post_barge += 1
                        ti = d.get('turn_id')
                        post_ids[ti] = post_ids.get(ti,0)+1
            elif fired is not None:
                if t == "barge_in" and barge_at is None:
                    barge_at = time.time()
                events_after.append((round(time.time() - fired, 2), t, str(d.get("text", ""))[:40]))
                if t == "tts_end" and after >= 0 and len(events_after) > 3:
                    break
        if speaker and not speaker.done():
            speaker.cancel()

    print(f"\n  chunks before speaking : {before}")
    print(f"  chunks AFTER speaking  : {after}  (includes STT detection window)")
    print(f"  chunks AFTER barge_in  : {post_barge}  by turn_id -> {post_ids}")
    print("  events after speech began:")
    for dt, t, tx in events_after[:14]:
        print(f"    +{dt:5.2f}s  {t:<12} {tx}")
    cut = next((dt for dt, t, _ in events_after if t == "barge_in"), None)
    leaked = {k: v for k, v in post_ids.items() if k in turn_ids_before}
    ok = cut is not None and not leaked
    print(f"\n  barge_in: {'+%.2fs' % cut if cut is not None else 'NEVER FIRED'}")
    print(f"  leaked chunks from the interrupted turn: {leaked or 'none'}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


asyncio.run(main())
