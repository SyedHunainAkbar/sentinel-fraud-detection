"""Retrieval for the fraud copilot.

A TF-IDF retriever runs offline (tests/CI) with no API keys. The interface is designed so
an embedding-based retriever (Bedrock / Anthropic / OpenAI) can be swapped in for
production without changing callers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Chunk, load_chunks

logger = logging.getLogger(__name__)


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


class BedrockEmbeddingRetriever:
    """Production retriever using AWS Bedrock Titan embeddings.

    Uses the Amazon Titan Embeddings V2 model to encode chunks and queries,
    then ranks by cosine similarity. Falls back to TF-IDF if the Bedrock call
    fails at runtime (network issues, credential expiry, etc.).

    Parameters
    ----------
    chunks : list[Chunk] | None
        Pre-loaded corpus chunks; loads from disk if not provided.
    model_id : str
        Bedrock model identifier for the embedding model.
    region : str | None
        AWS region; defaults to ``AWS_DEFAULT_REGION`` env var or us-east-1.
    """

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str | None = None,
    ) -> None:
        import boto3  # noqa: F811 — lazy import so offline tests don't need boto3

        self.chunks = chunks or load_chunks()
        self.model_id = model_id
        region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._embeddings: np.ndarray | None = None
        self._fallback = TfidfRetriever(self.chunks)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Call Bedrock to embed a batch of texts."""
        import json

        vectors = []
        for text in texts:
            body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            vectors.append(result["embedding"])
        return np.array(vectors)

    def _ensure_index(self) -> None:
        """Lazily build the embedding index on first retrieve call."""
        if self._embeddings is not None:
            return
        texts = [f"{c.heading}. {c.text}" for c in self.chunks]
        self._embeddings = self._embed_texts(texts)

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        """Return the top-k most relevant chunks using Bedrock embeddings.

        Falls back to the TF-IDF retriever if Bedrock is unreachable.
        """
        try:
            self._ensure_index()
            q_vec = self._embed_texts([query])
            sims = cosine_similarity(q_vec, self._embeddings)[0]
            order = sims.argsort()[::-1][:k]
            return [
                Retrieved(self.chunks[i], float(sims[i]))
                for i in order
                if sims[i] > 0
            ]
        except Exception:  # noqa: BLE001
            logger.warning(
                "Bedrock embedding call failed; falling back to TF-IDF retriever."
            )
            return self._fallback.retrieve(query, k)


def get_retriever(force_tfidf: bool = False):
    """Factory — returns the Bedrock embedding retriever if configured, else TF-IDF.

    Selection logic:
    1. If ``force_tfidf=True``, always return TF-IDF (useful for tests).
    2. If env var ``SENTINEL_RETRIEVER=bedrock``, attempt Bedrock.
    3. Otherwise return TF-IDF (safe offline default for CI).

    The Bedrock retriever itself has an internal TF-IDF fallback so callers
    never experience a hard failure from network issues.
    """
    if force_tfidf:
        return TfidfRetriever()

    if os.environ.get("SENTINEL_RETRIEVER", "").lower() == "bedrock":
        try:
            return BedrockEmbeddingRetriever()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not initialize Bedrock retriever; falling back to TF-IDF."
            )
            return TfidfRetriever()

    return TfidfRetriever()
