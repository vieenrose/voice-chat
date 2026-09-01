#!/usr/bin/env python3
"""
TTS ↔ ASR round-trip evaluator: is the speech we synthesize the speech we meant?

Why this exists
---------------
"Qwen3-TTS sometimes pronounces part of a sentence wrong" has at least five possible
causes, and they need different fixes:

  1. the TTS model itself            (wrong/extra/garbled audio for correct input)
  2. the streaming pipeline          (a chunk boundary cuts or repeats audio)
  3. text shaping before synthesis   (markdown, digits, mixed script, sentence splitting)
  4. the sample-rate / dtype path    (24k->16k resample, int16 clipping)
  5. the *evaluator*                 (the ASR mishearing, not the TTS misspeaking)

So this harness does not just print a CER. It varies one factor at a time and
cross-checks the transcription with two independent recognizers and two resamplers,
then localizes each mismatch against the chunk boundaries of the audio that was
actually produced:

  * same text, engine `synthesize()` (one call) vs `synthesize_streaming()` at three
    chunk sizes            -> separates (1) from (2)
  * same text repeated N times
                            -> separates (1) (stable) from sampling/nondeterminism
  * transcript from the production streaming X-ASR path vs the same audio one-shot,
    and 24k->16k by scipy polyphase vs naive decimation
                            -> separates (5) from (1..4)
  * clipping / silence / duration stats per utterance
                            -> catches (4), which sounds like "partially wrong"
  * mismatch positions vs chunk-boundary times
                            -> the actual smoking gun for (2)

Usage
-----
    python3 backend/test_tts_asr_roundtrip.py                       # whole corpus
    python3 backend/test_tts_asr_roundtrip.py --category mixed      # one category
    python3 backend/test_tts_asr_roundtrip.py --repeats 3 --modes full,stream
    python3 backend/test_tts_asr_roundtrip.py --tts qwen3 --asr xasr --out /tmp/tts_roundtrip.json

Needs the real models (GPU): it loads the TTS + STT adapters directly, and nothing
else — no LLM, no web, so VRAM is ~2 GB and a run is a couple of minutes.
Exit code is non-zero when any category's mean CER exceeds --max-cer, so this can
gate a TTS change instead of merely describing one.
"""
import argparse
import asyncio
import difflib
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET_SR = 16000          # what X-ASR consumes
EVAL_SR = TARGET_SR

# --------------------------------------------------------------------------
# Corpus. Categories matter more than count: each one targets a different failure
# mechanism. Nothing here is a benchmark question from test_e2e_report.py — this is
# the kind of text this assistant actually speaks (search summaries, forecasts, dates).
# --------------------------------------------------------------------------
CORPUS = [
    # (category, text, note)
    ("plain_zh", "今天天氣不錯，我們下午去河濱公園騎腳踏車。", "simple zh-TW sentence"),
    ("plain_zh", "沒問題，我馬上幫您處理这件事情。", "note the simplified 该 — script mix inside"),
    ("numbers", "最高攝氏三十四度，降雨機率百分之六十。", "numbers spelled out"),
    ("numbers", "最高溫度 34°C，最低溫 26°C，降雨機率 68%。", "digits + units — TTS reading style"),
    ("numbers", "會議安排在 2026 年 8 月 31 日下午 2 點 20 分。", "date + time"),
    ("proper_zh_en", "法國總統是艾曼紐·馬克宏（Emmanuel Macron）。", "CJK + latin in one sentence"),
    ("proper_zh_en", "台積電在纳斯达克上市，CEO 是趙淑敏。", "acronym + CJK + title"),
    ("pure_en", "The forecast for Tokyo calls for scattered showers this afternoon.", "English only"),
    ("pure_en", "Samsung's Q3 revenue rose 12 percent, beating analyst estimates.", "english + digits"),
    ("mixed", "IBM 的 quantum 團隊發表了新 paper，_interesting_ 吧？", "3-script soup + markdown"),
    ("markdown", "重點：**颱風路徑北移**、`停班停課` 名單已公布。", "markdown emphasis reaching TTS"),
    ("long", "根據中央氣象署的預報，明天上午開始降雨，午後有局部大雨的機率，"
             "恆春半島與東北部累積雨量可能超過一百毫米，請留意排水與土石流警示。", "one long sentence"),
    ("short", "好的。", "two characters — truncation-sensitive"),
    ("short", "抱歉，我找不到相關的資料。", "apology shape"),
    ("names", "台積電、鴻海、聯發科与智邦的股價今天都不漲。", "list of proper nouns + simplified 与"),
    # --- more coverage per category: with 1-2 items each, the category means moved by
    # +/-5pp from pure engine sampling noise, which is enough to "prove" any conclusion.
    ("plain_zh", "不好意思，我沒有聽清楚，可以請你再說一次嗎？", "clarification request"),
    ("plain_zh", "我已經帮您把行程安排好了，請確認一下時間。", "assistant confirmation, script mix 帮"),
    ("plain_zh", "這個建議不錯，不過我們還是得先看看預算。", "hedged opinion"),
    ("numbers", "气温攝氏 22 到 28 度，紫外線指數約 7。", "range + degree + index"),
    ("numbers", "營收 1,234.5 億元，年增 12%。", "thousands separator + percent"),
    ("numbers", "電話 02-2345-6789，分機 123。", "phone number reading"),
    ("numbers", "下午 3 點 15 分有 45 分鐘的會議。", "time + duration"),
    ("markdown", "三個重點：1. **降價**；2. `限時`；3. *送運費*。", "numbered markdown list"),
    ("markdown", "詳細說明請見 [氣象署](https://www.cwb.gov.tw/V8/) 官網。", "link with URL"),
    ("markdown", "| 項目 | 價格 |\n| --- | --- |\n| 咖啡 | 35 元 |", "markdown table"),
    ("markdown", "## 摘要\n- 第一項說明\n- 第二項說明", "heading + bullets"),
    ("mixed", "我用 Python 寫了一個 script 來 parse JSON，還挺好用。", "latin tech vocabulary"),
    ("mixed", "Google 的 Gemini 和 OpenAI 的 GPT 都在做多模態。", "many latin proper nouns"),
    ("mixed", "請將 PDF 裡的 3 個 table 匯出成 Excel。", "mixed with numbers"),
    ("proper_zh_en", "新任 CEO 魏哲家在法說會提到 AI 伺服器需求。", "title + latin acronym"),
    ("proper_zh_en", "东京（Tokyo）下周举办 Generative AI 展。", "CJK + latin + parens"),
    ("pure_en", "Turn off the AC and close the blinds before you leave.", "english phrasal verbs"),
    ("pure_en", "The meeting was rescheduled to Thursday at 3:45 p.m.", "english time format"),
    ("long", "首先，您需要準備申請表和身份證明文件；其次，在线上填妥資料後上傳；最後，等待審核結果通知，大約五個工作天。", "multi-clause enumeration"),
    ("short", "沒問題！", "two chars + full-width bang"),
    ("short", "這是第 2 個選項。", "short with a digit"),
    ("names", "新北市新店區思源路 100 號 3 樓。", "address with proper nouns"),
    ("names", "請轉接給張經理和李工程師。", "titles + surnames"),
]

_PUNCT = re.compile(r"[\s，。、；：？！「」『』（）《》〈〉…—·,.;:!?\-\"'()\[\]{}<>~/\\|@#$%^&*_+=`]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u00b7]")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_NUM_WORDS = re.compile(r"[零一二兩三四五六七八九十百千萬億点]{1,8}")

# X-ASR (like most zh ASR) emits SIMPLIFIED characters. Comparing it against a
# traditional-Chinese reference char-by-char scores 今天天气不错 vs 今天天氣不錯 as ~100%
# error — measured: a correctly spoken sentence scored 42% CER until this was normalized.
# So both sides are folded to simplified before diffing. Eval-only optional dependency
# (installed with `pip install --target $EVAL_LIB_DIR zhconv` to respect PEP 668); when it
# is missing the tool still runs but says so, because the CERs would be fiction.
_EVAL_LIB = os.getenv("EVAL_LIB_DIR", "/home/user/.eval-libs")
if _EVAL_LIB not in sys.path and os.path.isdir(_EVAL_LIB):
    sys.path.insert(0, _EVAL_LIB)
try:
    from zhconv import convert as _zh_convert          # noqa: E402

    def fold_script(text: str) -> str:
        return _zh_convert(text or "", "zh-cn")

    SCRIPT_FOLDED = True
except Exception:                                        # pragma: no cover - optional dep
    def fold_script(text: str) -> str:
        return text or ""

    SCRIPT_FOLDED = False


def spoken_units(text: str) -> list[str]:
    """Compare *what was said*, not how it was written."""
    t = fold_script((text or "").lower())
    t = _PUNCT.sub(" ", t)
    units: list[str] = []
    buf = ""
    for ch in t:
        if _CJK.match(ch):
            if buf.strip():
                units.extend(buf.split())
            buf = ""
            units.append(ch)
        else:
            buf += ch
    if buf.strip():
        units.extend(buf.split())
    return units


def strip_numeric(units: list[str]) -> list[str]:
    return [u for u in units if not _DIGITS.fullmatch(u) and not _NUM_WORDS.fullmatch(u)]


def cer(ref: str, hyp: str) -> tuple[float, list]:
    """Character/word-level edit distance ratio + the raw opcodes (positions included)."""
    a, b = units_and_nums(ref), units_and_nums(hyp)
    if not a:
        return (0.0, []) if not b else (1.0, [])
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    wrong = sum(max(o[2] - o[1], o[4] - o[3]) for o in ops)
    return wrong / len(a), ops


def units_and_nums(text: str) -> list[str]:
    return strip_numeric(spoken_units(text))


def numeric_style(text: str) -> str:
    has_digits = bool(_DIGITS.search(text or ""))
    has_words = bool(_NUM_WORDS.search(text or ""))
    return {"digits": has_digits, "zh_words": has_words}


def to_float(pcm: np.ndarray) -> np.ndarray:
    """Adapters yield int16 (the wire format); stats and ASR work in float [-1, 1]."""
    a = np.asarray(pcm)
    if a.dtype == np.int16 or (a.size and float(np.max(np.abs(a.astype(np.float64)))) > 4.0):
        return a.astype(np.float32) / 32768.0
    return a.astype(np.float32)


def resample(src: np.ndarray, sr_in: int, sr_out: int, method: str) -> np.ndarray:
    x = to_float(src)
    if sr_in == sr_out:
        return x
    if method == "polyphase":
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    if method == "naive":
        n = int(round(len(x) * sr_out / sr_in))
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        return x[idx].astype(np.float32)
    raise ValueError(method)


def audio_stats(pcm: np.ndarray, sr: int) -> dict:
    """Clipping and silence are the mechanical causes of 'partially wrong speech'."""
    if pcm is None or len(pcm) == 0:
        return {"seconds": 0.0, "clipped_ratio": 0.0, "leading_silence_s": 0.0,
                "trailing_silence_s": 0.0, "peak": 0.0, "rms": 0.0, "empty": True}
    peak = float(np.max(np.abs(pcm)))
    clipped = float(np.mean(np.abs(pcm) >= 0.985))
    frame = int(0.02 * sr)
    if len(pcm) < frame:
        return {"seconds": len(pcm) / sr, "clipped_ratio": clipped, "leading_silence_s": 0.0,
                "trailing_silence_s": 0.0, "peak": peak, "rms": float(np.sqrt(np.mean(pcm ** 2))),
                "empty": False}
    frames = pcm[: len(pcm) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    loud = np.where(rms > 0.01)[0]
    lead = (loud[0] * frame / sr) if len(loud) else len(pcm) / sr
    trail = ((len(frames) - 1 - loud[-1]) * frame / sr) if len(loud) else 0.0
    return {"seconds": len(pcm) / sr, "clipped_ratio": round(clipped, 5),
            "leading_silence_s": round(lead, 3), "trailing_silence_s": round(trail, 3),
            "peak": round(peak, 4), "rms": round(float(np.sqrt(np.mean(pcm ** 2))), 4), "empty": False}


class Engine:
    """The TTS adapter under test, loaded alone (no LLM, no STT-side GPU waste)."""

    def __init__(self, which: str, device: str, raw_text: bool = False):
        self.which = which
        # Measure the text the service would actually send. --raw-text reproduces the
        # pre-front-end behaviour, which is how the before/after comparison stays honest.
        from tts.spoken_text import normalize as _norm
        self.spoken = (lambda t: (t, [])) if raw_text else _norm
        if which in ("current", "qwen3"):
            from tts.qwen3_streaming import StreamingPrimeTTS, SAMPLE_RATE
            self.tts = StreamingPrimeTTS(device=device, mock=False)
        elif which == "audio8":
            from tts.audio8_onnx_streaming import StreamingPrimeTTS
            SAMPLE_RATE = 44100
            self.tts = StreamingPrimeTTS(device="cpu", mock=False)
        elif which == "moss":
            from tts.moss_streaming import StreamingPrimeTTS, SAMPLE_RATE
            self.tts = StreamingPrimeTTS(model_id=None, device=device, mock=False)
        elif which == "voxcpm":
            from tts.voxcpm_streaming import StreamingPrimeTTS, SAMPLE_RATE
            self.tts = StreamingPrimeTTS(model_id=None, device=device, mock=False)
        else:
            raise SystemExit(f"unknown --tts {which!r}")
        self.sr = int(getattr(self.tts, "sample_rate", SAMPLE_RATE))
        self.backend = getattr(self.tts, "backend", which)
        if getattr(self.tts, "mock", False):
            raise SystemExit(f"{which} adapter fell back to mock — the round-trip would measure "
                             "a tone generator. Fix the model install first.")

    async def full(self, text: str, chunk_frames: int | None = None) -> np.ndarray:
        import inspect
        text = self.spoken(text)[0]
        try:
            if chunk_frames and "chunk_frames" in inspect.signature(self.tts.synthesize).parameters:
                return await self.tts.synthesize(text, chunk_frames=chunk_frames)
        except (ValueError, TypeError):
            pass
        return await self.tts.synthesize(text)

    async def stream(self, text: str, chunk_frames: int) -> tuple[np.ndarray, list[float]]:
        chunks = []
        text = self.spoken(text)[0]
        async for c in self.tts.synthesize_streaming(text, chunk_frames=chunk_frames):
            if c is not None and len(c):
                chunks.append(np.asarray(c, dtype=np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32), []
        audio = np.concatenate(chunks)
        boundaries = []
        pos = 0
        for c in chunks[:-1]:
            pos += len(c)
            boundaries.append(pos / self.sr)
        return audio, boundaries


async def measure_asr_floor(rec: "Recognizer") -> dict | None:
    """Transcribe a HUMAN recording with a known transcript -> the ASR's own error floor.

    Every CER in this report is TTS->audio->ASR, so it can never be lower than whatever the
    ASR gets wrong on real speech. The repo ships exactly that: asr_example.wav plus its
    verified transcript (backend/test_e2e_report.py). Read the category numbers as
    `cer - floor`, not as `cer`.
    """
    wav = Path(__file__).resolve().parent.parent / "asr_example.wav"
    if not wav.is_file():
        return None
    from test_e2e_report import STT_GROUND_TRUTH, STT_GROUND_TRUTH_NO_LEADIN
    import wave
    with wave.open(str(wav)) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        sr = w.getframerate()
    f32 = resample(raw.astype(np.float32) / 32768.0, sr, TARGET_SR, "polyphase")
    paced = await rec.paced_stream(f32)
    one = await rec.one_shot(f32)
    return {
        "seconds": round(len(f32) / TARGET_SR, 2),
        "cer_paced": round(cer(STT_GROUND_TRUTH, paced)[0], 4),
        "cer_paced_no_leadin": round(cer(STT_GROUND_TRUTH_NO_LEADIN, paced)[0], 4),
        "cer_one_shot": round(cer(STT_GROUND_TRUTH, one)[0], 4),
        "hyp_paced": paced, "hyp_one_shot": one,
    }


class Recognizer:
    """X-ASR, used two ways: the production paced-streaming path, and a one-shot pass.

    A single recognizer is not ground truth — that is the point of the second method and
    the second resampler: an error that only appears under one of them is an evaluation
    artifact, not a mispronunciation.
    """

    def __init__(self, device: str):
        from stt.xasr_streaming import StreamingXASR
        self.asr = StreamingXASR(device=device, mock=False, chunk_ms=160)
        if getattr(self.asr, "mock", False):
            raise SystemExit("X-ASR came up mock — transcripts would be fiction. Fix the STT install.")
        self.backend = self.asr.backend

    async def one_shot(self, pcm_16k: np.ndarray) -> str:
        return await self.asr.transcribe_once(pcm_16k)

    async def paced_stream(self, pcm_16k: np.ndarray) -> str:
        """Feed real 100 ms frames through the same streaming API the voice path uses.

        The adapter's queue contract is int16 (it divides by 32768 itself) plus a
        {"type":"flush"} sentinel — and the final transcript is produced *while handling
        flush*, so setting stop_event at the same moment makes the generator exit before it
        decodes the tail (both bugs cost a debugging cycle each: float32 items and an
        immediately-set stop flag both yielded "" and therefore a suspicious 100% CER).
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        stop = asyncio.Event()
        frame = int(0.1 * TARGET_SR)
        pcm16 = np.clip(pcm_16k, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        text = ""

        async def pump():
            for i in range(0, len(pcm16), frame):
                await q.put(pcm16[i:i + frame])
                await asyncio.sleep(0.004)             # mild pacing: the decoder must keep up
            for _ in range(10):
                await q.put(np.zeros(frame, dtype=np.int16))
                await asyncio.sleep(0.004)
            await q.put({"type": "flush"})

        pump_task = asyncio.create_task(pump())
        try:
            async with asyncio.timeout(180):
                async for ev in self.asr.transcribe_stream(q, stop):
                    if ev.get("type") == "stt_final" and ev.get("text"):
                        text = ev["text"]
                        break                          # flush answered -> nothing more wanted
                    elif ev.get("type") == "stt_partial" and ev.get("text"):
                        text = ev["text"]
        except TimeoutError:
            print("  ! paced_stream timed out; using the last partial", file=sys.stderr)
        finally:
            stop.set()
            pump_task.cancel()
        return text


async def evaluate(args):
    engine = Engine(args.tts, args.device, raw_text=args.raw_text)
    print(f"TTS   : {engine.backend} (output {engine.sr} Hz)")
    if args.raw_text:
        print("TEXT  : RAW (markdown/units go to the TTS verbatim)")
    else:
        print("TEXT  : through tts/spoken_text.py, same as the service")
    rec = Recognizer(args.device)
    print(f"ASR   : {rec.backend} (16 kHz)")
    floor = await measure_asr_floor(rec)
    if floor:
        print(f"FLOOR : ASR error on human speech with known text = {floor['cer_paced']:.1%} "
              f"(paced) / {floor['cer_one_shot']:.1%} (one-shot)  <- read CER below as cer-floor")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    chunk_sizes = [int(x) for x in args.chunk_frames.split(",")]
    corpus = [c for c in CORPUS if args.category in (None, c[0])]

    rows = []
    t_start = time.time()
    for idx, (cat, text, note) in enumerate(corpus, 1):
        per_mode = {}
        for mode in modes:
            if mode == "full":
                variants = [("full", 0)]
            else:
                variants = [(f"stream{cs}", cs) for cs in chunk_sizes]
            for label, cs in variants:
                reps = []
                for rep in range(args.repeats):
                    try:
                        if mode == "full":
                            audio, bounds = await engine.full(text, args.full_chunk_frames), []
                        else:
                            audio, bounds = await engine.stream(text, cs)
                    except Exception as e:
                        rows.append({"category": cat, "text": text, "variant": label, "rep": rep,
                                     "error": f"{type(e).__name__}: {e}"})
                        continue
                    st = audio_stats(to_float(audio), engine.sr)
                    if st["empty"]:
                        rows.append({"category": cat, "text": text, "variant": label, "rep": rep,
                                     "error": "empty audio"})
                        continue
                    # two independent decode paths on the SAME audio
                    t_rec = time.time()
                    p16 = resample(audio, engine.sr, EVAL_SR, "polyphase")
                    p16_naive = resample(audio, engine.sr, EVAL_SR, "naive")
                    txt_stream = await rec.paced_stream(p16)
                    txt_oneshot = await rec.one_shot(p16)
                    txt_naive = await rec.one_shot(p16_naive)
                    c_stream, ops = cer(text, txt_stream)
                    c_oneshot, _ = cer(text, txt_oneshot)
                    c_naive, _ = cer(text, txt_naive)
                    # localized mismatch positions vs the audio's own chunk boundaries
                    near_boundary = 0
                    for op in ops:
                        pos_units = op[1]
                        frac = pos_units / max(1, len(units_and_nums(text)))
                        pos_s = frac * st["seconds"]
                        if any(abs(pos_s - b) < 0.45 for b in bounds):
                            near_boundary += 1
                    reps.append({
                        "variant": label, "rep": rep,
                        "cer_stream_asr": round(c_stream, 4),
                        "cer_oneshot_asr": round(c_oneshot, 4),
                        "cer_naive_resample": round(c_naive, 4),
                        "audio": st, "asr_s": round(time.time() - t_rec, 2),
                        "chunk_boundaries_s": [round(b, 2) for b in bounds],
                        "mismatches_near_boundary": near_boundary, "n_mismatch_ops": len(ops),
                        "hyp_stream": txt_stream, "hyp_oneshot": txt_oneshot,
                        "numeric_in_ref": numeric_style(text), "numeric_in_hyp": numeric_style(txt_stream),
                    })
                per_mode[label] = reps
        rows.append({"category": cat, "text": text, "note": note, "variants": per_mode})
        best = min((r["cer_stream_asr"] for lbl in per_mode for r in per_mode[lbl]), default=1.0)
        print(f"[{idx:2}/{len(corpus)}] {cat:11} cer(min)={best:5.1%}  {text[:38]}")

    # ---- aggregate ----
    by_cat, by_variant = defaultdict(list), defaultdict(list)
    clip_n = 0
    boundary_hits = boundary_total = 0
    disagreement = 0
    for row in rows:
        for lbl, reps in (row.get("variants") or {}).items():
            for r in reps:
                if "error" in r:
                    continue
                by_cat[row["category"]].append(r["cer_stream_asr"])
                by_variant[lbl].append(r["cer_stream_asr"])
                if r["audio"]["clipped_ratio"] > 0.001:
                    clip_n += 1
                boundary_hits += r["mismatches_near_boundary"]
                boundary_total += r["n_mismatch_ops"]
                if abs(r["cer_stream_asr"] - r["cer_oneshot_asr"]) > 0.08:
                    disagreement += 1

    def agg(d):
        return {k: {"mean_cer": round(statistics.mean(v), 4), "max_cer": round(max(v), 4),
                    "worst": round(sorted(v)[-1], 4) if v else None, "n": len(v),
                    "clean_ratio": round(sum(1 for x in v if x <= args.max_cer) / max(1, len(v)), 3)}
                for k, v in sorted(d.items())}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tts_backend": engine.backend, "asr_backend": rec.backend,
        "tts_sr": engine.sr, "repeats": args.repeats, "chunk_frames": chunk_sizes,
        "corpus_n": len(corpus), "wall_s": round(time.time() - t_start, 1),
        "max_cer_threshold": args.max_cer,
        "by_category": agg(by_cat), "by_variant": agg(by_variant),
        "signals": {
            "asr_error_floor": floor,
            "script_folding": ("zh-cn character fold (zhconv) applied to reference and hypothesis"
                               if SCRIPT_FOLDED else
                               "UNAVAILABLE — CER includes traditional/simplified mismatches"),
            "clipped_utterances": clip_n,
            "mismatch_ops_total": boundary_total,
            "mismatch_ops_within_0.45s_of_a_chunk_boundary": boundary_hits,
            "boundary_correlation": round(boundary_hits / boundary_total, 3) if boundary_total else None,
            "asr_path_disagreements": disagreement,
            "note": ("boundary_correlation well above chance means the streaming chunker is cutting "
                     "speech (pipeline bug); asr_path_disagreements means the *evaluation* is unstable; "
                     "high CER that is identical across variants/repeats is the model itself."),
        },
    }
    out = args.out or "/tmp/tts_asr_roundtrip.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=1)

    print("\n" + "=" * 74)
    print("  CER by category (production paced-streaming ASR path)")
    for k, v in summary["by_category"].items():
        print(f"    {k:12} mean {v['mean_cer']:6.1%}   max {v['max_cer']:6.1%}   clean {v['clean_ratio']:5.0%}  n={v['n']}")
    print("  CER by synthesis variant")
    for k, v in summary["by_variant"].items():
        print(f"    {k:12} mean {v['mean_cer']:6.1%}   max {v['max_cer']:6.1%}   n={v['n']}")
    s = summary["signals"]
    print("  signals")
    print(f"    utterances with clipping        : {s['clipped_utterances']}")
    print(f"    mismatches near a chunk boundary: {s['mismatch_ops_within_0.45s_of_a_chunk_boundary']}"
          f"/{s['mismatch_ops_total']}  ({s['boundary_correlation']})")
    print(f"    ASR one-shot vs paced disagree  : {s['asr_path_disagreements']}")
    print(f"  report: {out}")
    print("=" * 74)

    worst = max((v["mean_cer"] for v in summary["by_category"].values()), default=0.0)
    return 0 if worst <= args.max_cer else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default="current", choices=["current", "qwen3", "audio8", "moss", "voxcpm"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--category", default=None)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--modes", default="full,stream")
    ap.add_argument("--chunk-frames", default="8,24,64")
    ap.add_argument("--full-chunk-frames", type=int, default=64,
                    help="chunk size for the non-streaming call: some engines re-decode a "
                         "window per chunk, so one big chunk is far cheaper than many small ones")
    ap.add_argument("--max-cer", type=float, default=0.08)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw-text", action="store_true",
                    help="skip the TTS text front-end (measures the engine on written text as-is)")
    args = ap.parse_args()
    sys.exit(asyncio.run(evaluate(args)))


if __name__ == "__main__":
    main()
