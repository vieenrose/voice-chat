"""Smart Turn v3.2 (soniqo ONNX, int8) on onnxruntime CPU: does the user's turn end here?

The model takes the last 8 s of 16 kHz mono audio; the log-mel front end and the zero-mean
normalisation are inside the graph, and zeros pad the front of short turns. `predict` returns the
probability that the turn is complete (>= 0.5 means finished, as in upstream Smart Turn).
"""
import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
WINDOW = 8 * SAMPLE_RATE  # 128000 samples
THRESHOLD = 0.5


class SmartTurn:
    def __init__(self, model_path, threads=2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

    def predict(self, tail):
        """tail: float32 mono 16 kHz, most recent sample last. Returns P(turn complete)."""
        window = np.zeros(WINDOW, dtype=np.float32)
        tail = np.asarray(tail, dtype=np.float32)[-WINDOW:]
        if len(tail):
            window[WINDOW - len(tail):] = tail
        out = self.sess.run(["probability"], {"audio": window[None]})[0]
        return float(out[0, 0])
