"""Truncation check: streams clips through the running server at real-time pace and compares the
concatenated `user` texts with an offline transcription of the whole clip (same ASR, flushed).

A clip passes when the live text covers the reference: similarity >= MIN_SIM and the last
TAIL characters of the reference appear in the live text (the end of the sentence is not lost).
Run on the Jetson with the server up (one session at a time), from the deployed folder:
  venv/bin/python -u app/text_check.py clip.wav ...
"""
import asyncio
import json
import os
import re
import ssl
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import websockets

try:
    import zhconv
except ImportError:
    zhconv = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr import Asr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN = (ROOT / "server" / "lan_token").read_text().strip()  # read, never printed
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
MIN_SIM = 0.85
TAIL = 4
SKIP = re.compile(r"[\s，。！？、,.!?；;：:]")


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def norm(s):
    s = SKIP.sub("", s)
    return zhconv.convert(s, "zh-tw") if zhconv else s


def reference(asr, pcm):
    asr.reset()
    y = np.concatenate([pcm.astype(np.float32) / 32768, np.zeros(16000, np.float32)])
    for i in range(0, len(y), 2048):
        asr.feed(y[i:i + 2048])
    return asr.finish()


async def live(pcm, settle=8.0):
    texts = []
    pcm = np.concatenate([pcm, np.zeros(int(2.5 * 16000), "<i2")])
    async with websockets.connect("wss://127.0.0.1:8443/ws?token=" + TOKEN, ssl=CTX, max_size=None) as ws:
        async def send():
            for i in range(0, len(pcm), 2048):
                await ws.send(pcm[i:i + 2048].tobytes())
                await asyncio.sleep(2048 / 16000)

        sender = asyncio.ensure_future(send())
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), settle)
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m["type"] == "user":
                    if m.get("replace") and texts:  # reopened turn: the whole utterance again
                        texts[-1] = m["text"]
                    else:
                        texts.append(m["text"])
                elif m["type"] == "busy":
                    raise SystemExit("server busy: close the browser tab")
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            pass
        finally:
            sender.cancel()
    return texts


async def main(clips):
    asr = Asr(os.path.expanduser(os.environ.get("VC_ASR_DIR", "~/xasr-eval/base")), threads=1)
    bad = 0
    for clip in clips:
        pcm = load(clip)
        ref = norm(reference(asr, pcm))
        texts = await live(pcm)
        got = norm("".join(texts))
        # coverage: share of the reference found in the live text, in order (live may say more than the reference)
        blocks = SequenceMatcher(None, ref, got, autojunk=False).get_matching_blocks()
        sim = sum(b.size for b in blocks) / len(ref) if ref else 0.0
        tail_ok = ref[-TAIL:-1] in got if ref else False  # last character may be a borderline word
        ok = sim >= MIN_SIM and tail_ok
        bad += not ok
        print("%s %s sim=%.2f tail=%s turns=%d" % ("PASS" if ok else "FAIL", Path(clip).name, sim, tail_ok, len(texts)))
        if not ok:
            print("  ref:", ref)
            print("  got:", got)
            print("  turns:", texts)
    return bad


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main(sys.argv[1:])) else 0)
