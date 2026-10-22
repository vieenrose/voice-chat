"""Builds the small text-frontend resources the app bundles for PrimeTTS (run on a dev machine).

Needs: g2pw (package data), the G2PWModel zip contents, bert-base-chinese vocab.txt, nltk cmudict.
Usage: python build_tts_resources.py <G2PWModel dir> <vocab.txt> <out dir>
"""
import json
import sys
from pathlib import Path

import importlib.util
from nltk.corpus import cmudict

model_dir, vocab, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
pkg = Path(importlib.util.find_spec("g2pw").origin).parent  # find_spec: no import, torch not needed

poly = [line.split("\t") for line in (model_dir / "POLYPHONIC_CHARS.txt").read_text().strip().split("\n")]
mono = dict(line.split("\t") for line in (model_dir / "MONOPHONIC_CHARS.txt").read_text().strip().split("\n"))
fallback = json.loads((pkg / "char_bopomofo_dict.json").read_text())

labels = sorted({ph for _, ph in poly})
char2ph: dict[str, list[int]] = {}
for ch, ph in poly:
    char2ph.setdefault(ch, []).append(labels.index(ph))
chars = sorted(char2ph)

# Same precedence as G2PWConverter._prepare_data: polyphonic, then monophonic, then dictionary.
single = dict(mono)
for ch, readings in fallback.items():
    if ch not in char2ph and ch not in single:
        single[ch] = readings[0]

(out / "g2pw.json").write_text(
    json.dumps({"labels": labels, "chars": chars, "char2ph": char2ph, "single": single}, ensure_ascii=False, separators=(",", ":"))
)
(out / "vocab.txt").write_text(vocab.read_text())

cmu = cmudict.dict()
lines = [f"{w} {' '.join(prons[0])}" for w, prons in sorted(cmu.items()) if w.isascii() and w.replace("'", "").isalpha()]
(out / "cmudict.txt").write_text("\n".join(lines))
print(len(labels), "labels,", len(chars), "polyphonic chars,", len(single), "single,", len(lines), "cmu words")

# g2p_en's neural OOV model (GRU seq2seq, hidden 256) as one float32 blob, arrays in this fixed order.
import numpy as np

g2pen = Path(importlib.util.find_spec("g2p_en").origin).parent / "checkpoint20.npz"
w = np.load(g2pen)
order = ["enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
         "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh", "fc_w", "fc_b"]
(out / "g2pen.bin").write_bytes(b"".join(w[k].astype("<f4").tobytes() for k in order))
print("g2pen.bin", {k: w[k].shape for k in order})
