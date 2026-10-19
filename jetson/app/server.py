"""LAN voice-chat server for the Jetson: HTTPS + token-gated WebSocket, one session at a time.

Client -> server
  binary  int16 PCM, 16 kHz mono (mic)
  json    {"type": "played", "turn": n, "seq": k}   first sample of chunk k of turn n is audible
Server -> client
  json    {"type": "partial" | "user", "text"}     ASR hypotheses / finished user utterance
  json    {"type": "audio_start", "turn", "seq", "text", "samples"} followed by one binary int16 frame
  json    {"type": "stop_audio"}                   barge-in: drop everything queued for playback
  json    {"type": "assistant", "turn", "text"}    full reply (not sent when the turn was cut)
  json    {"type": "latency", ...}                 ms from last voiced chunk, server clock
  json    {"type": "busy"}                         another session holds the slot (close 4409)

Barge-in: a partial ASR hypothesis of at least BARGE_CHARS characters while a turn is running
cancels that turn: STOP to the engine, the TTS queue is dropped, the client stops playback.
Cut turns are never written to the history, as on iOS.
"""
import asyncio
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr import Asr  # noqa: E402
from llm import Llm  # noqa: E402
from smart_turn import SmartTurn  # noqa: E402
import timeline as tr  # noqa: E402
from tts import Tts  # noqa: E402
from vad import SileroVad  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "index.html"
TRACE_PAGE = ROOT / "web" / "trace.html"
TOKEN_FILE = ROOT / "server" / "lan_token"
CERT = ROOT / "server" / "cert.pem"
KEY = ROOT / "server" / "key.pem"
END_MARKS = "。！？!?；;\n"
SOFT_MARKS = "，,"
SOFT_MIN = 6  # first comma after this many characters ends a TTS chunk
SMART_TURN = os.environ.get("VC_SMART_TURN", "0") == "1"  # off by default until measured on the Nano
THRESHOLD_TURN = 0.5  # Smart Turn: probability that the turn is complete
INCOMPLETE_DELAY_S = 0.6  # upstream smart_turn_incomplete_delay_ms: wait, then ask again
MAX_WAIT_S = 2.0  # upstream smart_turn_max_wait_ms: the turn ends at the latest here

try:
    import zhconv
except ImportError:  # display-only conversion; the models still run without it
    zhconv = None


def to_tw(s):
    return zhconv.convert(s, "zh-tw") if zhconv else s


CJK_GAP = re.compile(r"(?<=[㐀-鿿])\s+(?=[㐀-鿿])")
PUNCT = re.compile(r"[\s，。！？、,.!?；;：:]")
PRE_ROLL = int(1.6 * 16000)  # buffer kept while the gate is closed
PRE_MARGIN = int(0.25 * 16000)  # audio before the confirmed onset sent to the ASR
REOPEN_S = 1.0  # speech resuming this soon after a turn ended, while its reply runs, continues that turn
ECHO_RATIO = 0.6 # share of the heard text found verbatim in what the assistant just said
ECHO_TAIL_S = 1.5  # speech that starts while the assistant plays, or this long after, may be echo


def clean_cjk(s):
    """X-ASR emits a space between characters; drop it so the LLM and the UI see normal text."""
    return CJK_GAP.sub("", s)


def is_echo(text, said):
    """True when `text` is mostly a stretch of the assistant's own recent speech (mic picked up the speaker)."""
    t = PUNCT.sub("", to_tw(text))
    if len(t) < 4:
        return False
    for ref in list(said) + ["".join(said)]:
        r = PUNCT.sub("", to_tw(ref))
        if r and SequenceMatcher(None, t, r, autojunk=False).find_longest_match(0, len(t), 0, len(r)).size \
                >= ECHO_RATIO * len(t):
            return True
    return False


def find_cut(buf):
    """Index just after the first sentence end (or soft break after SOFT_MIN chars), else -1."""
    for i, ch in enumerate(buf):
        if ch in END_MARKS:
            return i + 1
        if ch in SOFT_MARKS and i + 1 >= SOFT_MIN:
            return i + 1
    return -1


def token_ok(given):
    try:
        want = TOKEN_FILE.read_text().strip()
    except OSError:
        return False
    return hmac.compare_digest(given.encode(), want.encode())


def ensure_token():
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TOKEN_FILE.exists():
        fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(24) + "\n")
    os.chmod(TOKEN_FILE, 0o600)


def ensure_cert(hosts):
    """Self-signed certificate for the LAN; regenerated only if missing."""
    if CERT.exists() and KEY.exists():
        return
    san = ",".join(hosts)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                    "-keyout", str(KEY), "-out", str(CERT), "-subj", "/CN=voice-chat",
                    "-addext", "subjectAltName=" + san], check=True, capture_output=True)
    os.chmod(KEY, 0o600)


class Engines:
    llm = tts = asr = smart = None
    busy = False


ENG = Engines()


def load_engines():
    model_dir = os.environ.get("VC_ASR_DIR", os.path.expanduser("~/xasr-eval/base"))
    ENG.asr = Asr(model_dir, threads=int(os.environ.get("VC_ASR_THREADS", "2")))
    ENG.tts = Tts(os.environ.get("PRIME_ONNX", str(ROOT / "primetts_v21_3voice.onnx")),
                  str(ROOT / "models" / "g2pw" / "g2pw_int8.onnx"), str(ROOT / "tts_res"),
                  threads=int(os.environ.get("TTS_THREADS", "4")))
    ENG.vad = SileroVad(str(ROOT / "silero_vad.onnx"), threshold=0.6, min_speech_ms=384, min_silence_ms=450)
    if SMART_TURN:
        ENG.smart = SmartTurn(str(ROOT / "models" / "smart-turn" / "smart-turn-v3.2-int8.onnx"),
                              threads=int(os.environ.get("VC_SMART_TURN_THREADS", "2")))
    ENG.llm = Llm(str(ROOT / "fwd35_serve"), str(ROOT / "models" / "Ternary-Bonsai-2-2B-PTQ1_0.gguf"),
                  str(ROOT / "models" / "tok" / "tokenizer.json"), max_new=200)


class Turn:
    def __init__(self, n, text, t_last_voice):
        self.n = n
        self.text = text
        self.t_last_voice = t_last_voice
        self.cancelled = threading.Event()
        self.reply = ""
        self.t_first_token = None
        self.t_first_sent = None


def run_turn(turn, emit, history):
    """Worker thread: LLM deltas -> sentences -> TTS -> events for the sender coroutine."""
    buf = ""
    t_req = time.monotonic()
    try:
        for delta in ENG.llm.stream(history, turn.text):
            if turn.cancelled.is_set():
                continue  # keep draining so the engine is free for the next request
            if turn.t_first_token is None:
                turn.t_first_token = time.monotonic()
                tr.add_span("LLM", t_req, turn.t_first_token, "prefill")
                tr.mark("first_token")
            turn.reply += delta
            buf += delta
            while True:
                cut = find_cut(buf)
                if cut < 0:
                    break
                sentence, buf = buf[:cut], buf[cut:]
                _speak(turn, emit, sentence)
        if buf.strip():
            _speak(turn, emit, buf)
    except Exception as e:  # reported to the client, the turn ends
        emit(("error", turn, str(e)))
    if turn.t_first_token is not None:
        tr.add_span("LLM", turn.t_first_token, time.monotonic(), "decode" + (" (cut)" if turn.cancelled.is_set() else ""))
    emit(("done", turn))


def _speak(turn, emit, sentence):
    if turn.cancelled.is_set() or not sentence.strip():
        return
    with tr.span("TTS", sentence):
        wav, _ = ENG.tts.synth_sentence(sentence)
    if turn.cancelled.is_set() or len(wav) == 0:
        return
    emit(("audio", turn, sentence, wav))


async def session(ws):
    loop = asyncio.get_running_loop()
    out_q = asyncio.Queue()
    st = {"cur": None, "last": None, "n": 0, "play_end": 0.0,  # play_end: estimated end of queued audio
          "tail": np.zeros(0, dtype=np.float32),  # last 8 s of mic audio, for Smart Turn
          "held": None,  # (t_last_voice, monotonic time) of a turn Smart Turn judged unfinished
          "said": deque(maxlen=4), "said_t": 0.0,  # what was sent to the speaker, for the echo filter
          "open": False, "pre": np.zeros(0, dtype=np.float32),  # VAD-gated STT: open flag, pre-roll audio
          "prev": None,  # (text, end time) of the last user utterance, for turn reopen
          "reopen": False,  # the current utterance continues the previous one
          "echo_zone": False}  # the current utterance started while (or right after) the assistant spoke
    history = []

    def emit(item):
        loop.call_soon_threadsafe(out_q.put_nowait, item)

    async def start_turn(text, t_last_voice):
        st["n"] += 1
        turn = Turn(st["n"], text, t_last_voice)
        st["cur"], st["last"] = turn, turn
        threading.Thread(target=run_turn, args=(turn, emit, list(history)), daemon=True).start()

    async def cut_current(notify=True):
        """Cancels generation and playback. Playback counts as active until its estimated end."""
        cur = st["cur"]
        speaking = time.monotonic() < st["play_end"]
        if cur is not None or speaking:
            tr.mark("barge_in")
        if cur is not None:
            cur.cancelled.set()
            ENG.llm.cancel()
            st["cur"] = None
        st["play_end"] = 0.0
        if (cur is not None or speaking) and notify:
            try:
                await ws.send_json({"type": "stop_audio"})
            except RuntimeError:  # socket already closed
                pass

    async def sender():
        seq = 0
        while True:  # ends when the task is cancelled on disconnect
            item = await out_q.get()
            kind, turn = item[0], item[1]
            if turn.cancelled.is_set() and kind != "done":
                continue
            if kind == "audio":
                _, _, text, wav = item
                seq += 1
                pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                await ws.send_json({"type": "audio_start", "turn": turn.n, "seq": seq,
                                    "text": to_tw(text), "samples": len(wav)})
                await ws.send_bytes(pcm)
                tr.mark("audio_sent", "seq %d" % seq)
                tr.add_span("Speaker", max(time.monotonic(), st["play_end"]),
                            max(time.monotonic(), st["play_end"]) + len(wav) / 16000, text)
                st["play_end"] = max(time.monotonic(), st["play_end"]) + len(wav) / 16000
                st["said"].append(text)
                st["said_t"] = st["play_end"]
                if turn.t_first_sent is None:
                    turn.t_first_sent = time.monotonic()
            elif kind == "error":
                await ws.send_json({"type": "error", "message": item[2]})
            elif kind == "done":
                if st["cur"] is turn:
                    st["cur"] = None
                if not turn.cancelled.is_set() and turn.reply.strip():
                    history.append((turn.text, turn.reply))
                    await ws.send_json({"type": "assistant", "turn": turn.n, "text": to_tw(turn.reply)})
                seq = 0

    async def smart_says_unfinished():
        p = await asyncio.to_thread(ENG.smart.predict, st["tail"])
        await ws.send_json({"type": "turn_check", "p": round(p, 3)})  # every VAD speech_end, for the UI and tuning
        return p < THRESHOLD_TURN

    async def end_utterance(t_voice):
        st["open"] = False  # gate closed: the STT hears nothing until the next speech_start
        with tr.span("ASR", "finish"):
            text = clean_cjk(to_tw(await asyncio.to_thread(ENG.asr.finish)))
        if text and st["echo_zone"] and is_echo(text, st["said"]):
            print("echo dropped: %s" % text, flush=True)
            await ws.send_json({"type": "echo_dropped"})  # the UI clears its partial line
            return
        if text and st["reopen"] and st["prev"]:
            text = st["prev"][0] + text  # as upstream's turn reopen: one user turn, not two
        if text:
            st["prev"] = (text, time.monotonic())
            tr.mark("user", text)
            await ws.send_json({"type": "user", "text": text, "replace": st["reopen"]})
            await cut_current()
            await start_turn(text, t_voice)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                samples = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768
                st["tail"] = np.concatenate([st["tail"], samples])[-128000:]
                playing = time.monotonic() < st["play_end"]
                with tr.span("VAD"):
                    events = ENG.vad.feed(samples, strict=playing)
                # As upstream: the STT only sees audio the VAD delivered. Gate closed -> keep a pre-roll;
                # speech_start opens it (fresh stream, pre-roll first); end_utterance closes it.
                if not st["open"] and any(ev == "speech_start" for ev, _ in events):
                    await asyncio.to_thread(ENG.asr.reset)
                    # only the confirmed onset plus a margin: long leading silence makes the ASR drop words
                    keep = ENG.vad.start_samples + PRE_MARGIN + len(samples)  # ~0.76 s: measured best on the test clips
                    with tr.span("ASR", "pre-roll"):
                        await asyncio.to_thread(ENG.asr.feed, st["pre"][-keep:])
                    st["open"], st["pre"] = True, np.zeros(0, dtype=np.float32)
                if st["open"]:
                    with tr.span("ASR"):
                        asr_events = await asyncio.to_thread(ENG.asr.feed, samples)
                    for kind, text in asr_events:
                        if not playing or ENG.vad.triggered:  # while the assistant talks, partials are mostly echo
                            await ws.send_json({"type": "partial", "text": clean_cjk(to_tw(text))})
                else:
                    st["pre"] = np.concatenate([st["pre"], samples])[-PRE_ROLL:]
                for ev, t_voice in events:
                    tr.mark(ev)
                    if ev == "speech_start":
                        # barge-in: the VAD decides, not the ASR text. Cut generation and playback now.
                        st["held"] = None
                        now = time.monotonic()
                        st["echo_zone"] = now < st["said_t"] + ECHO_TAIL_S
                        onset = now - ENG.vad.start_samples / 16000
                        # reply still pending or playing: the user had not finished, reopen their turn
                        replying = st["cur"] is not None or now < st["play_end"]
                        st["reopen"] = bool(st["prev"] and replying and onset - st["prev"][1] < REOPEN_S)
                        await cut_current()
                        await ws.send_json({"type": "speech_started"})
                    elif SMART_TURN and await smart_says_unfinished():
                        now = time.monotonic()  # incomplete: hold, the ASR utterance stays open
                        st["held"] = (t_voice, now, now + INCOMPLETE_DELAY_S)
                    else:
                        await end_utterance(t_voice)
                held = st["held"]
                if held is not None and time.monotonic() >= held[2]:
                    t_voice, t_end, _ = held
                    now = time.monotonic()
                    if now - t_end >= MAX_WAIT_S or not await smart_says_unfinished():
                        st["held"] = None
                        await end_utterance(t_voice)
                    else:
                        st["held"] = (t_voice, t_end, now + INCOMPLETE_DELAY_S)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "played":
                    turn = st["last"]
                    if turn is not None and turn.n == data.get("turn") and turn.t_first_sent is not None \
                            and data.get("seq") == 1:
                        tr.mark("played")
                        now = time.monotonic()
                        lv = turn.t_last_voice
                        await ws.send_json({
                            "type": "latency", "turn": turn.n,
                            "llm_first_token_ms": round(1000 * (turn.t_first_token - lv), 1)
                            if turn.t_first_token else None,
                            "first_audio_sent_ms": round(1000 * (turn.t_first_sent - lv), 1),
                            "first_audio_played_ms": round(1000 * (now - lv), 1),
                        })
    finally:
        await cut_current(notify=False)
        send_task.cancel()


app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(WEB)


@app.get("/trace")
async def trace_page():
    return FileResponse(TRACE_PAGE)


@app.get("/trace.json")
async def trace_json(token: str = Query(""), s: float = Query(30.0)):
    if not token_ok(token):
        return JSONResponse({"error": "token"}, status_code=401)
    return JSONResponse(tr.snapshot(min(max(s, 5.0), 180.0)))


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "busy": ENG.busy})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str = Query("")):
    if not token_ok(token):
        await ws.close(code=4401)
        return
    await ws.accept()
    if ENG.busy:
        await ws.send_json({"type": "busy"})
        await ws.close(code=4409)
        return
    ENG.busy = True
    try:
        ENG.asr.reset()
        ENG.vad.reset()
        try:
            await session(ws)
        except WebSocketDisconnect:  # client left mid-send: normal end of session
            pass
    finally:
        ENG.busy = False


def lan_address():
    """Address of the interface that routes to the LAN (no packet is actually sent)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    ensure_token()
    lan_ip = lan_address()
    ensure_cert(["DNS:localhost", "IP:127.0.0.1", "IP:" + lan_ip])
    load_engines()
    tr.start()
    port = int(os.environ.get("VC_PORT", "8443"))
    print("voice-chat: https://%s:%d  (token in %s)" % (lan_ip, port, TOKEN_FILE), flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_keyfile=str(KEY), ssl_certfile=str(CERT),
                ws_max_size=1 << 20, log_level="warning")


if __name__ == "__main__":
    main()
