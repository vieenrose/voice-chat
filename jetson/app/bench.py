"""End-to-end latency bench: streams WAV clips to the running server at real-time pace, as the browser does.

Per clip it keeps the server's own latency message (ms from the last voiced window): LLM first token,
first audio sent, first audio played (the client acks seq 1 like index.html). Prints each clip and the
median / p90 over clips. --pause-ms inserts silence in the middle of each clip, to exercise Smart Turn's
incomplete-turn path.

Run on the Jetson while the server is up (one session at a time):
  venv/bin/python app/bench.py [--pause-ms 700] clip.wav ...
"""
import argparse
import asyncio
import json
import math
import ssl
import statistics
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
TOKEN = (ROOT / "server" / "lan_token").read_text().strip()  # read, never printed
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
KEYS = ("llm_first_token_ms", "first_audio_sent_ms", "first_audio_played_ms")


def load(path, pause_ms):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    if pause_ms:
        mid = len(pcm) // 2
        pcm = np.concatenate([pcm[:mid], np.zeros(16 * pause_ms, "<i2"), pcm[mid:]])
    return np.concatenate([pcm, np.zeros(3 * 16000, "<i2")])  # trailing silence closes the turn


async def run_clip(pcm, timeout=30.0):
    """Returns (last latency message or None, Smart Turn probabilities seen in the session)."""
    lat = None
    checks = []
    head = None  # (turn, seq) announced by the last audio_start
    loop = asyncio.get_running_loop()
    async with websockets.connect("wss://127.0.0.1:8443/ws?token=" + TOKEN, ssl=CTX, max_size=None) as ws:
        async def send():
            for i in range(0, len(pcm), 2048):
                await ws.send(pcm[i:i + 2048].tobytes())
                await asyncio.sleep(2048 / 16000)

        sender = asyncio.ensure_future(send())
        deadline = loop.time() + timeout
        try:
            while loop.time() < deadline:
                msg = await asyncio.wait_for(ws.recv(), deadline - loop.time())
                if isinstance(msg, bytes):
                    if head and head[1] == 1:  # first chunk of a turn: it is audible now
                        await ws.send('{"type": "played", "turn": %d, "seq": 1}' % head[0])
                    continue
                m = json.loads(msg)
                if m["type"] == "audio_start":
                    head = (m["turn"], m["seq"])
                elif m["type"] == "latency":
                    lat = m
                elif m["type"] == "turn_check":
                    checks.append(m["p"])
                elif m["type"] in ("error", "busy"):
                    print("  server:", m)
                    break
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            pass
        finally:
            sender.cancel()
    return lat, checks


def pct(values, q):
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


async def main(clips, pause_ms):
    rows = []
    for clip in clips:
        lat, checks = await run_clip(load(clip, pause_ms))
        name = Path(clip).name
        held = sum(1 for p in checks if p < 0.5)  # Smart Turn said "unfinished" at a VAD speech_end
        if lat is None:
            print("%-16s no latency message  checks=%s" % (name, checks))
            continue
        print("%-16s " % name + "  ".join("%s=%s" % (k, lat.get(k)) for k in KEYS)
              + "  checks=%s held=%d" % (checks, held))
        rows.append(lat)
    for k in KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            print("%-24s median %.0f ms  p90 %.0f ms  (n=%d)" % (k, statistics.median(vals), pct(vals, 0.9), len(vals)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause-ms", type=int, default=0)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    asyncio.run(main(a.clips, a.pause_ms))
