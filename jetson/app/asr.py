"""Streaming X-ASR on sherpa-onnx, same models and endpoint rules as the iOS LocalASR.

Feed 16 kHz mono float32 samples. `feed` returns ("partial", text) when the hypothesis changes.
Utterance boundaries come from the VAD (vad.py): call `finish()` on speech_end. Sherpa's own
endpointing is off, so it never discards text before the VAD closes the turn.
"""
import os
import time

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
FLUSH_S = float(os.environ.get("VC_ASR_FLUSH_S", "1.0"))  # 0.6 s cut the end of sentences (CER 27 -> 16.5%)
CHUNK = int(0.16 * SAMPLE_RATE)  # 160 ms, as on iOS


class Asr:
    def __init__(self, model_dir, threads=2):
        m = lambda name: os.path.join(model_dir, name)
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=m("tokens.txt"), encoder=m("encoder.int8.onnx"), decoder=m("decoder.onnx"),
            joiner=m("joiner.int8.onnx"), num_threads=threads, sample_rate=SAMPLE_RATE,
            feature_dim=80, enable_endpoint_detection=False, decoding_method="greedy_search")
        self.stream = self.rec.create_stream()
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_text = ""

    def finish(self):
        """Final text of the utterance (called on VAD speech_end); starts the next utterance."""
        # Flush: the streaming encoder looks ahead, so the last frames are only decoded once more audio
        # (here silence) arrives. Without it the end of the sentence is lost.
        tail = np.concatenate([self.pending, np.zeros(int(FLUSH_S * SAMPLE_RATE), dtype=np.float32)])
        self.stream.accept_waveform(SAMPLE_RATE, tail)
        self.stream.input_finished()
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = self.rec.get_result(self.stream).strip()
        self.reset()
        return text

    def reset(self):
        self.stream = self.rec.create_stream()
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_text = ""

    def feed(self, samples):
        events = []
        self.pending = np.concatenate([self.pending, np.asarray(samples, dtype=np.float32)])
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            self.stream.accept_waveform(SAMPLE_RATE, chunk)
            while self.rec.is_ready(self.stream):
                self.rec.decode_stream(self.stream)
            text = self.rec.get_result(self.stream).strip()
            if text != self.last_text:
                self.last_text = text
                if text:
                    events.append(("partial", text))
        return events


if __name__ == "__main__":
    import sys
    import wave

    asr = Asr(sys.argv[1])
    with wave.open(sys.argv[2], "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    pcm = np.concatenate([pcm, np.zeros(3 * SAMPLE_RATE, dtype=np.float32)])  # trailing silence closes the utterance
    t0 = time.monotonic()
    for i in range(0, len(pcm), CHUNK):
        for ev in asr.feed(pcm[i:i + CHUNK]):
            print(round(time.monotonic() - t0, 2), ev)
    print("last_voice_rel", None if asr.last_voice is None else round(asr.last_voice - t0, 2))
