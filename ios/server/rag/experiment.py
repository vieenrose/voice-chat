"""Compare retrieval on the 和合本 corpus: BM25 (jieba), EmbeddingGemma 2, and a hybrid.

Usage: python experiment.py <corpus_dir> [--model google/embeddinggemma-2] [--dim 256]
A query counts as a hit at k when one of the top-k passages contains its expected phrase.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import jieba
import numpy as np

# (question, expected phrase that must appear in a relevant passage)
EVAL = [
    ("神起初創造天地是怎麼回事", "起初，神創造天地"),
    ("挪亞造方舟躲避洪水", "挪亞"),
    ("以色列人過逾越節為什麼要殺羔羊", "逾越節"),
    ("摩西從神那裡領受的十條誡命", "不可殺人"),
    ("大衛如何戰勝巨人歌利亞", "歌利亞"),
    ("耶穌在哪裡出生", "伯利恆"),
    ("誰是好撒瑪利亞人的故事", "撒瑪利亞"),
    ("浪子回家父親的故事", "浪子"),
    ("耶和華是我的牧者我必不致缺乏", "牧者"),
    ("愛是恆久忍耐又有恩慈", "恆久忍耐"),
    ("約拿被大魚吞下去的事", "大魚"),
    ("以西結看見乾骨復活", "乾骨"),
]


def load_passages(corpus: Path, max_chars: int = 400) -> list[dict]:
    """Group paragraphs into passages of roughly max_chars, keeping the book name."""
    passages = []
    for path in sorted(corpus.glob("*.txt")):
        book = path.stem.split("_", 1)[-1]
        buf = ""
        for para in path.read_text(encoding="utf-8").split("\n"):
            if not para.strip():
                continue
            if buf and len(buf) + len(para) > max_chars:
                passages.append({"book": book, "text": buf})
                buf = ""
            buf += para
        if buf:
            passages.append({"book": book, "text": buf})
    return passages


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / len(self.lens)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            for t in query:
                if t in d:
                    f = d[t]
                    out[i] += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
        return out


def top_k(scores: np.ndarray, k: int) -> list[int]:
    return list(np.argsort(-scores)[:k])


def hit_rate(rank_lists: list[list[int]], passages: list[dict], k: int) -> float:
    hits = 0
    for (_, phrase), ranks in zip(EVAL, rank_lists):
        if any(phrase in passages[i]["text"] for i in ranks[:k]):
            hits += 1
    return hits / len(EVAL)


def minmax(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--candidates", type=int, default=50, help="embed only the BM25 top-N per question (CPU is slow)")
    args = ap.parse_args()

    passages = load_passages(Path(args.corpus))
    print(f"{len(passages)} passages from {len({p['book'] for p in passages})} books")

    tok = lambda s: [w for w in jieba.lcut(s) if w.strip()]  # noqa: E731
    bm25 = BM25([tok(p["text"]) for p in passages])
    bm25_scores = [bm25.scores(tok(q)) for q, _ in EVAL]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, truncate_dim=args.dim, config_kwargs={"vision_config": None, "audio_config": None})
    cands = [top_k(b, args.candidates) for b in bm25_scores]
    needed = sorted({i for c in cands for i in c})
    print(f"embedding {len(needed)} candidate passages", flush=True)
    embs = model.encode([passages[i]["text"] for i in needed], prompt_name="Document", normalize_embeddings=True, batch_size=16)
    row = {i: n for n, i in enumerate(needed)}
    queries = model.encode([q for q, _ in EVAL], prompt_name="SearchQuery", normalize_embeddings=True)
    # Passages outside a question's BM25 candidates keep the lowest score, so dense/hybrid are rerankers of BM25 top-N.
    dense_scores = []
    for q, c in zip(queries, cands):
        d = np.full(len(passages), -1.0)
        for i in c:
            d[i] = float(embs[row[i]] @ q)
        dense_scores.append(d)

    hybrid_scores = [0.5 * minmax(b) + 0.5 * minmax(d) for b, d in zip(bm25_scores, dense_scores)]
    hybrid_scores = [np.where(d > -1.0, h, -1.0) for h, d in zip(hybrid_scores, dense_scores)]

    for name, scores in [("BM25", bm25_scores), ("dense", dense_scores), ("hybrid", hybrid_scores)]:
        ranks = [top_k(s, 10) for s in scores]
        print(f"{name:7s} hit@1={hit_rate(ranks, passages, 1):.2f} hit@3={hit_rate(ranks, passages, 3):.2f} hit@5={hit_rate(ranks, passages, 5):.2f}")

    for (q, phrase), s in zip(EVAL, hybrid_scores):
        best = top_k(s, 1)[0]
        ok = phrase in passages[best]["text"]
        print(f"{'OK ' if ok else 'MISS'} {q} -> {passages[best]['book']}")


if __name__ == "__main__":
    sys.exit(main())
