"""PrimeTTS chunked vocoder vs one-shot model on the Nano: latency to first audio, RTF, seam quality.

The streaming split (Luigi/PrimeTTS v21_streaming) runs `enc` once (text -> z [1,192,T]) then `dec`
on z windows: CHUNK frames kept, LEFT frames of context before and RIGHT after, 256 samples per
frame (as the PrimeTTS-Streaming Space). Seam quality: the chunked audio against `dec` on the whole
z in one call (same z, so the difference is only the chunking).

  venv/bin/python -u app/tts_stream_check.py [chunk left right]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prime_frontend import blank_interleave  # noqa: E402
from tts import Tts, _session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOP = 256
TEXTS = ["好，收到。", "台南是台灣南部的一個城市。", "三月十五日第二百九十九元折扣已確認，謝謝你的耐心等待。"]


def inputs(tts, text):
    out = tts.frontend.convert(text)
    x = np.array([blank_interleave(out["phone_ids"])], dtype=np.int64)
    return {"x": x, "tone": np.array([blank_interleave(out["tones"])], dtype=np.int64),
            "lang": np.array([blank_interleave(out["langs"])], dtype=np.int64),
            "x_lengths": np.array([x.shape[1]], dtype=np.int64), "sid": tts.voice,
            "noise_scale": np.array([0.667], dtype=np.float32), "length_scale": np.array([1.0], dtype=np.float32)}


def main(chunk, left, right):
    threads = int(os.environ.get("TTS_THREADS", "4"))
    tts = Tts(os.environ.get("PRIME_ONNX", os.path.expanduser("~/primetts_v2/primetts_v21_3voice.onnx")),
              str(ROOT / "models" / "g2pw" / "g2pw_int8.onnx"), str(ROOT / "tts_res"), threads=threads)
    enc = _session(str(ROOT / "models" / "primetts_stream" / "v21_enc.onnx"), threads)
    dec = _session(str(ROOT / "models" / "primetts_stream" / "v21_dec.onnx"), threads)
    enc_in = {i.name for i in enc.get_inputs()}
    print("enc inputs", sorted(enc_in), "-> dec", [i.name for i in dec.get_inputs()], "| chunk", chunk, left, right)
    for text in TEXTS * 2:  # first pass warms up the sessions
        t0 = time.monotonic()
        feed = inputs(tts, text)
        t_front = time.monotonic() - t0
        tts.vits.run(["wav"], feed)
        t_full = time.monotonic() - t0

        t0 = time.monotonic()
        feed = inputs(tts, text)
        z = enc.run(None, {k: v for k, v in feed.items() if k in enc_in})[0]
        t_enc = time.monotonic() - t0
        T = z.shape[2]
        parts, first = [], None
        for b in range(0, T, chunk):
            e = min(b + chunk, T)
            v = max(0, b - left)
            w = dec.run(None, {"z": np.ascontiguousarray(z[:, :, v:min(T, e + right)])})[0].reshape(-1)
            parts.append(w[(b - v) * HOP:(e - v) * HOP])
            if first is None:
                first = time.monotonic() - t0
        t_stream = time.monotonic() - t0
        wav = np.concatenate(parts)
        ref = dec.run(None, {"z": z})[0].reshape(-1)[:len(wav)]
        snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - wav) ** 2), 1e-12))
        dur = len(wav) / 16000
        print("%-28s audio %.2fs | one-shot %.2fs (front %.2f) | stream first %.2fs (enc %.2f) total %.2fs RTF %.2f | seam SNR %.1f dB"
              % (text[:14], dur, t_full, t_front, first, t_enc, t_stream, t_stream / dur, snr))


if __name__ == "__main__":
    a = [int(v) for v in sys.argv[1:4]] or [24, 64, 16]
    main(*a)
