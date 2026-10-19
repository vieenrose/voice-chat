"""PrimeTTS v2.1 on onnxruntime CPU: Python frontend (prime_frontend) + g2pw ONNX + VITS ONNX.

Mirrors PrimeSynthesizer.swift: g2pw scores polyphones, the frontend turns text into phone ids,
tones and languages, then the VITS model renders 16 kHz float audio. Text is cut into sentences so
the first chunk can play while the next one is synthesised.
"""
import json
import os
import re
import time

import numpy as np
import onnxruntime as ort

from prime_frontend import G2PEn, G2PW, PrimeFrontend, blank_interleave

SAMPLE_RATE = 16000
SENTENCE_END = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]*")


def split_sentences(text):
    return [s.strip() for s in SENTENCE_END.findall(text) if s.strip()]


def _session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class Tts:
    def __init__(self, model_path, g2pw_path, res_dir, threads=4, voice=0):
        self.voice = np.array([voice], dtype=np.int64)  # 0 Xinran (f), 1 Anchen (m), 2 Bowen (m)
        self.vits = _session(model_path, threads)
        self.g2pw_sess = _session(g2pw_path, threads)
        res = json.load(open(os.path.join(res_dir, "g2pw.json"), encoding="utf-8"))
        self.label_count = len(res["labels"])
        vocab = open(os.path.join(res_dir, "vocab.txt"), encoding="utf-8").read()
        symbols = json.load(open(os.path.join(res_dir, "symbol_table.json"), encoding="utf-8"))["symbols"]
        g2pw = G2PW(res, vocab)
        g2pen = G2PEn(open(os.path.join(res_dir, "cmudict.txt"), encoding="utf-8").read(),
                      os.path.join(res_dir, "g2pen.bin"))
        self.frontend = PrimeFrontend(symbols, g2pw, g2pen, self._score)

    def _score(self, ids, positions, char_ids, masks):
        batch = len(ids)
        width = max(len(r) for r in ids)
        input_ids = np.zeros((batch, width), dtype=np.int64)
        attention = np.zeros((batch, width), dtype=np.int64)
        for i, row in enumerate(ids):
            input_ids[i, :len(row)] = row
            attention[i, :len(row)] = 1
        probs = self.g2pw_sess.run(["probs"], {
            "input_ids": input_ids,
            "token_type_ids": np.zeros((batch, width), dtype=np.int64),
            "attention_mask": attention,
            "phoneme_mask": np.asarray(masks, dtype=np.float32).reshape(batch, self.label_count),
            "char_ids": np.asarray(char_ids, dtype=np.int64),
            "position_ids": np.asarray(positions, dtype=np.int64),
        })[0]
        return [list(map(float, probs[i])) for i in range(batch)]

    def synth_sentence(self, text):
        """Returns (float32 wav at 16 kHz, seconds spent) for one sentence."""
        t0 = time.monotonic()
        out = self.frontend.convert(text)
        if not out["phone_ids"]:
            return np.zeros(0, dtype=np.float32), time.monotonic() - t0
        phones = np.array([blank_interleave(out["phone_ids"])], dtype=np.int64)
        tones = np.array([blank_interleave(out["tones"])], dtype=np.int64)
        langs = np.array([blank_interleave(out["langs"])], dtype=np.int64)
        wav = self.vits.run(["wav"], {
            "x": phones, "tone": tones, "lang": langs,
            "x_lengths": np.array([phones.shape[1]], dtype=np.int64),
            "sid": self.voice,
            "noise_scale": np.array([0.667], dtype=np.float32),
            "length_scale": np.array([1.0], dtype=np.float32),
        })[0].reshape(-1).astype(np.float32)
        return wav, time.monotonic() - t0

    def stream(self, text):
        """Yields (sentence, wav, synth_seconds) per sentence, in order."""
        for s in split_sentences(text):
            wav, dt = self.synth_sentence(s)
            if len(wav):
                yield s, wav, dt


if __name__ == "__main__":
    import sys
    import wave

    root = sys.argv[1]
    tts = Tts(os.environ.get("PRIME_ONNX", os.path.join(root, "primetts_v21_3voice.onnx")), os.path.join(root, "models/g2pw/g2pw_int8.onnx"),
              os.path.join(root, "tts_res"), threads=int(os.environ.get("TTS_THREADS", "3")))
    text = sys.argv[2] if len(sys.argv) > 2 else "台灣的首都是台北市，人口大約有兩千五百萬。"
    audio, synth = [], 0.0
    for sent, wav, dt in tts.stream(text):
        audio.append(wav)
        synth += dt
        print("%.2f s audio, synth %.2f s : %s" % (len(wav) / SAMPLE_RATE, dt, sent))
    pcm = np.concatenate(audio)
    secs = len(pcm) / SAMPLE_RATE
    print("total audio %.2f s, synth %.2f s, RTF %.3f" % (secs, synth, synth / max(secs, 1e-6)))
    out = sys.argv[3] if len(sys.argv) > 3 else None
    if out:
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
