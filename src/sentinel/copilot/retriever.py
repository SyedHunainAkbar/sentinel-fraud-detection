"""Retrieval for the fraud copilot.

A TF-IDF retriever runs offline (tests/CI) with no API keys. The interface is designed so
an embedding-based retriever (Bedrock / Anthropic / OpenAI) can be swapped in for
production without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Chunk, load_chunks


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class TfidfRetriever:
    """Offline retriever over the policy corpus."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or load_chunks()
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(
            [f"{c.heading}. {c.text}" for c in self.chunks]
        )

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        """Return the top-k most relevant chunks for a query."""
        q = self._vec.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [Retrieved(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]


def get_retriever():
    """Factory — returns the production retriever if configured, else TF-IDF.

    To use embeddings in production, implement an EmbeddingRetriever with the same
    ``retrieve(query, k) -> list[Retrieved]`` signature and return it here based on an env
    flag. Kept as a single seam so callers never change.
    """
    return TfidfRetriever()
