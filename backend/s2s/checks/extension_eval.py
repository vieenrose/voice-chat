"""Measured accuracy and latency for extension-lookup-by-voice.

Every case is a real spoken turn over the Realtime protocol: the question is
synthesised by this demo's own TTS, played into the pipeline, and scored against
the directory. Nothing is mocked, so the numbers include VAD endpointing, Gemma's
audio encoder, the tool call, and TTS.

Three case shapes, because they fail differently:

  unique     a name only one person has -> the extension must be spoken exactly
  ambiguous  a name several people share -> the bot must ASK which department,
             and must NOT read an extension out
  misheard   the name with one character swapped for a homophone -> the phonetic
             fallback should still land on the right person

Scored on what the user hears, not on what the tool returned: an answer is right
only if the correct extension appears in the spoken text and no wrong 4-digit
number does. That is the failure this exists to catch -- the tool returned 1102
every time while the assistant said 3567.

    python3 -m s2s.checks.extension_eval --n 30
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import statistics as st
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.contact_db import CONTACTS  # noqa: E402

WS = "ws://127.0.0.1:8765/v1/realtime"
CACHE = Path("/tmp/claude-1000/ext_eval_clips")
_EXT = re.compile(r"\b(\d{4})\b")
def _homophones() -> dict[str, list[str]]:
    """Characters that sound alike, learned from the directory's own names.

    A hand-written swap table was the first version of this, and it was both
    arbitrary and small. Grouping every character in the directory by its toneless
    pinyin gives real confusions -- 陳/程, 怡/宜, 建/健 -- without anyone choosing
    them, and it grows with the data.
    """
    from pypinyin import Style, lazy_pinyin

    from tools.make_directory import GIVEN_1, GIVEN_2, SURNAMES

    # The directory's own characters are realistic but sparse (25 groups over 500
    # names). The generator's name-character pools widen it while staying inside
    # characters that actually occur in Taiwanese names.
    pool = {ch for c in CONTACTS for ch in c.name}
    pool |= set(GIVEN_1) | set(GIVEN_2) | {s for s, _ in SURNAMES}

    by_sound: dict[str, set[str]] = {}
    for ch in sorted(pool):
        syl = lazy_pinyin(ch, style=Style.NORMAL)
        if syl:
            by_sound.setdefault(syl[0], set()).add(ch)
    return {ch: sorted(g - {ch}) for g in by_sound.values() for ch in g if len(g) > 1}


_SWAPS = _homophones()


def _talk(pcm: bytes, rate: int) -> bytes:
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _synth(sentence: str, path: Path) -> None:
    """Speak a sentence with the demo's own TTS, so the input is real audio."""
    pcm = bytearray()
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime",
            "instructions": "使用者要求你照唸時，只輸出那句話本身，不要回答它。",
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text",
                         "text": f"請完全照著唸出以下這一句，不要回答它：{sentence}"}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            if ev.get("type") == "response.output_audio.delta":
                pcm += base64.b64decode(ev["delta"])
            if ev.get("type") == "response.done":
                break
    path.write_bytes(_talk(bytes(pcm), 24000))


async def _ask(clips: list[Path], attempts: int = 5) -> tuple[str, float, float]:
    """Play clips as one session. Returns (spoken text, first-audio s, total s).

    Retries on "all session slots are in use": /v1/pool reports the slot free a
    moment before the server has actually released it, and a bare run lost every
    case after the first collision.
    """
    for k in range(attempts):
        try:
            return await _ask_once(clips)
        except websockets.exceptions.ConnectionClosedError as exc:
            if "slots are in use" not in str(exc) or k == attempts - 1:
                raise
            await asyncio.sleep(3 * (k + 1))
    raise RuntimeError("unreachable")


async def _ask_once(clips: list[Path]) -> tuple[str, float, float]:
    said = ""
    first = None
    t_end = None
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime",
            "instructions": "你是一個親切的語音助理。一律使用繁體中文（台灣用語）回答。",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}},
                      "output": {"format": {"type": "audio/pcm", "rate": 24000}}}}}))
        for clip in clips:
            w = wave.open(str(clip))
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if sr != 16000:
                from scipy.signal import resample_poly
                g = np.gcd(16000, sr)
                a = resample_poly(a.astype(np.float32), 16000 // g, sr // g).astype(np.int16)
            for i in range(0, len(a), 512):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                    "audio": base64.b64encode(a[i:i + 512].tobytes()).decode()}))
                await asyncio.sleep(0.032)
            sil = np.zeros(512, dtype=np.int16)
            for _ in range(40):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                    "audio": base64.b64encode(sil.tobytes()).decode()}))
                await asyncio.sleep(0.032)
            t_end = time.monotonic()
            said = ""
            first = None
            while True:
                try:
                    ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                except asyncio.TimeoutError:
                    break
                t = ev.get("type")
                if t == "response.output_audio.delta" and first is None:
                    first = time.monotonic() - t_end
                if t == "response.output_audio_transcript.done":
                    said += ev.get("transcript", "")
                if t == "response.done":
                    break
    return said, (first or float("nan")), (time.monotonic() - t_end if t_end else float("nan"))


def _wait_slot() -> None:
    import urllib.request
    for _ in range(90):
        try:
            if json.load(urllib.request.urlopen(
                    "http://127.0.0.1:8765/v1/pool", timeout=5)).get("in_use") == 0:
                return
        except Exception:
            pass
        time.sleep(2)


def _cases(n: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    by_name: dict[str, list] = {}
    for c in CONTACTS:
        by_name.setdefault(c.name, []).append(c)
    uniq = [n_ for n_, v in by_name.items() if len(v) == 1]
    ambi = [n_ for n_, v in by_name.items() if len(v) > 1]

    # English names are only usable as a query when exactly one person has one.
    by_en: dict[str, list] = {}
    for c in CONTACTS:
        if c.english:
            by_en.setdefault(c.english.casefold(), []).append(c)
    en_uniq = [v[0] for v in by_en.values() if len(v) == 1]
    # A department+title pair that exactly one person holds.
    by_role: dict[tuple, list] = {}
    for c in CONTACTS:
        by_role.setdefault((c.dept, c.title), []).append(c)
    role_uniq = [v[0] for k, v in by_role.items() if len(v) == 1]
    role_many = [k for k, v in by_role.items() if len(v) > 3]

    per = max(1, n // 6)
    cases: list[dict] = []
    for name in rng.sample(uniq, min(per * 2, len(uniq))):
        cases.append({"shape": "unique", "name": name, "spoken": f"請問{name}的分機是多少",
                      "want": by_name[name][0].ext})
    for name in rng.sample(ambi, min(per, len(ambi))):
        cases.append({"shape": "ambiguous", "name": name,
                      "spoken": f"請問{name}的分機是多少", "want": None})
    for c in rng.sample(en_uniq, min(per, len(en_uniq))):
        cases.append({"shape": "english", "name": c.english,
                      "spoken": f"請問 {c.english} 的分機是多少", "want": c.ext})
    for c in rng.sample(role_uniq, min(per, len(role_uniq))):
        cases.append({"shape": "by_title", "name": f"{c.dept}/{c.title}",
                      "spoken": f"請問{c.dept}的{c.title}分機是多少", "want": c.ext})
    for dept, title in rng.sample(role_many, min(per, len(role_many))):
        cases.append({"shape": "too_broad", "name": f"{dept}/{title}",
                      "spoken": f"請問{dept}的{title}分機是多少", "want": None})
    for name in rng.sample(uniq, min(per, len(uniq))):
        swapped = None
        for i, ch in enumerate(name):
            alts = _SWAPS.get(ch)
            if alts:
                swapped = name[:i] + rng.choice(alts) + name[i + 1:]
                break
        if not swapped or swapped == name:
            continue
        cases.append({"shape": "misheard", "name": name,
                      "spoken": f"請問{swapped}的分機是多少", "want": by_name[name][0].ext})
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    cases = _cases(args.n, args.seed)
    print(f"directory: {len(CONTACTS)} people · cases: {len(cases)}\n", flush=True)

    rows, lat_first, lat_total = [], [], []
    for i, c in enumerate(cases, 1):
        clip = CACHE / f"{c['shape']}_{abs(hash(c['spoken'])) % 10**10}.wav"
        if not clip.exists():
            _wait_slot()
            asyncio.run(_synth(c["spoken"], clip))
        _wait_slot()
        said, first, total = asyncio.run(_ask([clip]))
        nums = set(_EXT.findall(said))
        if c["want"] is None:
            # Right behaviour: ask, and do not read an extension out.
            ok = not nums
        else:
            ok = c["want"] in nums and nums <= {c["want"]}
        rows.append({**c, "said": said, "nums": sorted(nums), "ok": ok,
                     "first": first, "total": total})
        lat_first.append(first)
        lat_total.append(total)
        print(f"  [{i:>2}/{len(cases)}] {c['shape']:9s} {c['name'][:14]:14s} "
              f"want={c['want'] or '(ask)':>6}  got={sorted(nums) or '[]'}  "
              f"{'OK ' if ok else 'BAD'}  {first:4.1f}s  {said.strip()[:46]}", flush=True)

    print()
    for shape in ("unique", "misheard", "english", "by_title", "ambiguous", "too_broad"):
        sub = [r for r in rows if r["shape"] == shape]
        if sub:
            print(f"  {shape:9s} {sum(r['ok'] for r in sub)}/{len(sub)}")
    ok = sum(r["ok"] for r in rows)
    print(f"  {'overall':9s} {ok}/{len(rows)}  ({100*ok/len(rows):.0f}%)")
    good = [x for x in lat_first if x == x]
    if good:
        # Two different things, and conflating them flatters the demo: the first
        # sound is the acknowledgement ("好的，我查一下。"), spoken before the tool
        # runs. What the caller waits for is the extension.
        print(f"\n  first sound (acknowledgement)  median {st.median(good):.2f}s  "
              f"p90 {sorted(good)[int(0.9*len(good))-1]:.2f}s")
        tot = [x for x in lat_total if x == x]
        print(f"  answer complete                median {st.median(tot):.2f}s  "
              f"p90 {sorted(tot)[int(0.9*len(tot))-1]:.2f}s")
    out = CACHE / "results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n  detail: {out}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
