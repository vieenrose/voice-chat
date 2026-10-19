"""Smart Turn probe: P(turn complete) for a clip cut at several points, followed by VAD-length silence.

A good turn detector gives a low probability when the clip stops mid-sentence and a high one at the true end.
usage: venv/bin/python app/probe_turn.py model.onnx clip.wav ...
"""
import sys
import wave

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from smart_turn import SmartTurn  # noqa: E402

CUTS = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
SILENCE_S = 0.45  # what the VAD has seen when it calls speech_end


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def speech_span(x, thr=0.01, win=800):
    """(first, last) sample index of the voiced part, by short-window RMS."""
    n = len(x) // win
    rms = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1))
    voiced = np.nonzero(rms > thr)[0]
    return (voiced[0] * win, (voiced[-1] + 1) * win) if len(voiced) else (0, len(x))


if __name__ == "__main__":
    st = SmartTurn(sys.argv[1], threads=2)
    for clip in sys.argv[2:]:
        x = load(clip)
        a, b = speech_span(x)
        row = []
        for c in CUTS:
            cut = a + int((b - a) * c)
            seg = np.concatenate([np.zeros(16000, np.float32), x[:cut], np.zeros(int(SILENCE_S * 16000), np.float32)])
            row.append("%.2f:%.3f" % (c, st.predict(seg)))
        print(clip.rsplit("/", 1)[-1], "speech %.1fs" % ((b - a) / 16000), " ".join(row))
