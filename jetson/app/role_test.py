"""End-to-end check of role switching against the running server: the roles list, the role message,
the spoken opening line (voice differs per role) and the stop when switching again mid-opening.

  venv/bin/python app/role_test.py
"""
import asyncio
import json
import ssl
import sys
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
TOKEN = (ROOT / "server" / "lan_token").read_text().strip()  # read, never printed
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


async def collect(ws, seconds):
    """JSON messages and total audio samples received in `seconds`."""
    msgs, samples = [], 0
    end = asyncio.get_running_loop().time() + seconds
    while (left := end - asyncio.get_running_loop().time()) > 0:
        try:
            m = await asyncio.wait_for(ws.recv(), left)
        except asyncio.TimeoutError:
            break
        if isinstance(m, bytes):
            samples += len(m) // 2
        else:
            msgs.append(json.loads(m))
    return msgs, samples


async def main():
    ok = True

    def check(cond, text):
        nonlocal ok
        ok &= bool(cond)
        print("%s %s" % ("PASS" if cond else "FAIL", text))

    async with websockets.connect("wss://127.0.0.1:8443/ws?token=" + TOKEN, ssl=CTX, max_size=None) as ws:
        msgs, _ = await collect(ws, 1.0)
        roles = next((m for m in msgs if m["type"] == "roles"), None)
        check(roles and len(roles["items"]) >= 5 and roles["current"] == "assistant", "roles list on connect")
        await ws.send(json.dumps({"type": "role", "id": "cafe"}))
        msgs, samples = await collect(ws, 8.0)
        types = [m["type"] for m in msgs]
        check("role" in types and any(m.get("name") == "咖啡店店員" for m in msgs if m["type"] == "role"), "role acknowledged")
        spoken = "".join(m["text"] for m in msgs if m["type"] == "audio_start")
        check(spoken == "歡迎光臨！請問今天想喝點什麼？" and samples > 16000, "opening spoken: %r, %.1fs audio" % (spoken, samples / 16000))
        check(any(m["type"] == "assistant" for m in msgs), "opening shown as an assistant line")
        await ws.send(json.dumps({"type": "role", "id": "interview"}))
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "role", "id": "guide"}))  # switch again while the opening is still going
        msgs, samples = await collect(ws, 8.0)
        spoken = "".join(m["text"] for m in msgs if m["type"] == "audio_start")
        check("stop_audio" in [m["type"] for m in msgs], "switching mid-opening stops the audio")
        check(spoken.endswith("嗨，歡迎來台南！今天想先去逛哪裡呢？") and "你好，請坐" not in spoken[-20:], "last role spoke: %r" % spoken[-24:])
        await ws.send(json.dumps({"type": "role", "id": "nope"}))
        msgs, _ = await collect(ws, 1.0)
        check(not any(m["type"] == "role" for m in msgs), "unknown role ignored")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
