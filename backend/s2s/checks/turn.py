"""One text turn against the Realtime server; prints transcript and latencies.

    python -m s2s.checks.turn "台灣的首都是哪裡？"

Needs the pipeline up (see s2s/serve.py) and its llama-server behind it.
"""
import asyncio
import json
import sys
import time
import websockets

URL = "ws://127.0.0.1:8765/v1/realtime"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "台灣的首都是哪裡？請簡短回答。"

async def main():
    t0 = time.time()
    audio_bytes = 0
    transcript = ""
    kinds = {}
    first_audio = None
    done = False
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type":"session.update","session":{
            "type":"realtime","instructions":"你是一個親切的語音助理，請用繁體中文簡短回答。"}}))
        await ws.send(json.dumps({"type":"conversation.item.create","item":{
            "type":"message","role":"user","content":[{"type":"input_text","text":PROMPT}]}}))
        await ws.send(json.dumps({"type":"response.create"}))
        while not done:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
            except asyncio.TimeoutError:
                print("!! timeout")
                break
            d = json.loads(raw) if isinstance(raw, str) else {"type":"<binary>"}
            t = d.get("type", "")
            kinds[t] = kinds.get(t, 0) + 1
            if t.endswith("output_audio.delta") or t == "response.output_audio.delta":
                audio_bytes += len(d.get("delta") or "")
                if first_audio is None:
                    first_audio = time.time()-t0
            elif "transcript.delta" in t:
                transcript += d.get("delta") or ""
            elif t == "response.output_audio_transcript.done":
                # With audio responses s2s carries the text on .done per chunk,
                # not as deltas.
                transcript += d.get("transcript") or ""
            elif t == "response.done":
                done = True
            elif t == "error" or "error" in t:
                print("ERROR EVENT:", json.dumps(d)[:400])
    print(f"\nprompt      : {PROMPT}")
    print(f"transcript  : {transcript.strip()!r}")
    print(f"first audio : {first_audio if first_audio is None else round(first_audio,2)} s")
    print(f"audio b64   : {audio_bytes} chars")
    print(f"total       : {round(time.time()-t0,2)} s")
    print("events      :", {k:v for k,v in sorted(kinds.items()) if v})

asyncio.run(main())
