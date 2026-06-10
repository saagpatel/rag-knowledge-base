"""BM25 sparse vector generator for lexical search in Qdrant."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from qdrant_client.models import SparseVector

STOP_WORDS = frozenset({
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "can", "no", "just", "him", "how", "its", "may", "into", "than",
    "then", "now", "only", "come", "could", "also", "more", "some",
    "very", "was", "were", "been", "is", "are", "am", "had", "has",
    "did", "does", "a",
})

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter short tokens and stop words."""
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in STOP_WORDS
    ]


@dataclass
class BM25Vectorizer:
    """BM25-based sparse vector generator.

    Build from a corpus with ``from_texts``, then call ``vectorize`` to produce
    Qdrant ``SparseVector`` objects suitable for the ``"sparse"`` named vector.
    """

    vocab: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    avg_doc_len: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        max_vocab_size: int = 30_000,
    ) -> BM25Vectorizer:
        """Build vocabulary and IDF from a corpus of texts."""
        n = len(texts)
        if n == 0:
            return cls()

        doc_freqs: Counter[str] = Counter()
        total_tokens = 0

        for text in texts:
            tokens = tokenize(text)
            total_tokens += len(tokens)
            unique = set(tokens)
            for t in unique:
                doc_freqs[t] += 1

        # Take top terms by document frequency
        top_terms = [t for t, _ in doc_freqs.most_common(max_vocab_size)]
        vocab = {term: idx for idx, term in enumerate(top_terms)}

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1), floored at 0
        idf: dict[str, float] = {}
        for term, idx in vocab.items():
            df = doc_freqs[term]
            score = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            idf[term] = max(score, 0.0)

        avg_doc_len = total_tokens / n

        return cls(vocab=vocab, idf=idf, avg_doc_len=avg_doc_len)

    def vectorize(self, text: str) -> SparseVector:
        """Convert a single text to a BM25 sparse vector."""
        if not self.vocab:
            return SparseVector(indices=[], values=[])

        tokens = tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf = Counter(tokens)
        dl = len(tokens)
        avgdl = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        indices: list[int] = []
        values: list[float] = []

        for term, count in tf.items():
            if term not in self.vocab:
                continue
            idf_score = self.idf.get(term, 0.0)
            numerator = count * (self.k1 + 1)
            denominator = count + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score = idf_score * numerator / denominator
            if score > 0:
                indices.append(self.vocab[term])
                values.append(score)

        return SparseVector(indices=indices, values=values)

    def vectorize_batch(self, texts: list[str]) -> list[SparseVector]:
        """Vectorize multiple texts."""
        return [self.vectorize(text) for text in texts]
