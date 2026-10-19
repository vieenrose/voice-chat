"""Silero VAD (ONNX, CPU) with the speech start/end state machine of speech-to-speech's VADIterator.

Silero v6 at 16 kHz: 512-sample windows, 64 samples of context carried between windows, state
[2,1,128]. `feed` returns events:
  ("speech_start", None)                 voiced for at least MIN_SPEECH_MS (barge-in trigger)
  ("speech_end", last_voice_monotonic)   silent for at least min_silence_ms after speech
The timestamp of speech_end points at the last voiced window, so "last speech detected" is exact.
"""
import time

import numpy as np
import onnxruntime as ort

WINDOW = 512
CONTEXT = 64
SAMPLE_RATE = 16000
WINDOW_MS = 1000 * WINDOW / SAMPLE_RATE  # 32 ms


class SileroVad:
    def __init__(self, model_path, threshold=0.5, min_speech_ms=192, min_silence_ms=450,
                 strict_threshold=0.85, strict_min_speech_ms=640):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.min_speech_windows = max(1, round(min_speech_ms / WINDOW_MS))
        # while the assistant is talking, echo leaks into the mic: starting a barge-in needs more evidence
        self.strict_threshold = strict_threshold
        self.strict_min_speech_windows = max(1, round(strict_min_speech_ms / WINDOW_MS))
        self.min_silence_windows = max(1, round(min_silence_ms / WINDOW_MS))
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.triggered = False
        self.voiced_run = 0
        self.silent_run = 0
        self.last_voice = None
        self.start_samples = 0
        self.loose_run = 0  # consecutive windows above neg_threshold: the whole voiced stretch

    def _prob(self, window):
        x = np.concatenate([self.context, window])[None, :].astype(np.float32)
        out, self.state = self.sess.run(["output", "stateN"], {
            "input": x, "state": self.state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self.context = window[-CONTEXT:]
        return float(out[0, 0])

    def feed(self, samples, now=None, strict=False):
        """samples: float32 mono 16 kHz. `now` is the monotonic time at which the last sample arrived.

        strict: the assistant is playing, so a speech start needs strict_threshold for
        strict_min_speech_ms (an ongoing utterance keeps the normal threshold).
        """
        now = time.monotonic() if now is None else now
        self.pending = np.concatenate([self.pending, np.asarray(samples, dtype=np.float32)])
        n = len(self.pending) // WINDOW
        events = []
        for k in range(n):
            # window k ends (n - k - 1) windows before the newest sample
            t_win = now - (n - k - 1) * WINDOW_MS / 1000
            window = self.pending[k * WINDOW:(k + 1) * WINDOW]
            p = self._prob(window)
            self.loose_run = self.loose_run + 1 if p >= self.neg_threshold else 0
            guarded = strict and not self.triggered
            if p >= (self.strict_threshold if guarded else self.threshold):
                self.last_voice = t_win
                self.silent_run = 0
                self.voiced_run += 1
                need = self.strict_min_speech_windows if guarded else self.min_speech_windows
                if not self.triggered and self.voiced_run >= need:
                    self.triggered = True
                    # strict start: voice before the strict run counts too; otherwise extra lead-in hurts the ASR
                    self.start_samples = (max(need, self.loose_run) if guarded else need) * WINDOW
                    events.append(("speech_start", None))
            elif p < self.neg_threshold or guarded:  # guarded: the run must be consecutive
                self.voiced_run = 0
                if self.triggered:
                    self.silent_run += 1
                    if self.silent_run >= self.min_silence_windows:
                        self.triggered = False
                        events.append(("speech_end", self.last_voice))
                        self.silent_run = 0
            # in between: keep current state (hysteresis)
        self.pending = self.pending[n * WINDOW:]
        return events
