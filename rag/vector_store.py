"""Hybrid vector store: FAISS dense retrieval plus BM25 sparse retrieval.

Embeddings are pluggable. HashEmbeddings is a deterministic, download-free
embedding used for tests and demos. SentenceTransformerEmbeddings wraps a
real embedding model when one is available.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HashEmbeddings:
    """Deterministic bag-of-words hash embeddings (no model download).

    Each token hashes into one of `dim` buckets; the vector is L2
    normalized, so cosine similarity reflects token overlap.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class SentenceTransformerEmbeddings:
    """Real dense embeddings via sentence-transformers (lazy import)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text])[0], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts), dtype=np.float32)


class HybridVectorStore:
    """FAISS index for dense search plus BM25 for keyword search.

    search() returns documents ranked by a weighted blend of normalized
    dense similarity and BM25 score, so exact keyword matches and semantic
    matches both surface.
    """

    def __init__(self, embeddings=None, dense_weight: float = 0.7):
        import faiss
        from rank_bm25 import BM25Okapi

        self.embeddings = embeddings or HashEmbeddings()
        self.dense_weight = dense_weight
        self._faiss = faiss
        self._BM25Okapi = BM25Okapi
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self._index: Any = None
        self._bm25: Any = None

    @property
    def dim(self) -> int:
        return self.embeddings.dim

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None):
        vectors = self.embeddings.embed_documents(texts)
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(self.dim)
        self._index.add(vectors.astype(np.float32))
        self.documents.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in texts])
        self._bm25 = self._BM25Okapi([tokenize(t) for t in self.documents])

    def _dense_scores(self, query: str) -> np.ndarray:
        query_vec = self.embeddings.embed(query).astype(np.float32)
        scores, indices = self._index.search(
            query_vec.reshape(1, -1), len(self.documents)
        )
        # FAISS returns scores sorted by rank, not by document position,
        # so map them back through the returned indices.
        out = np.zeros(len(self.documents), dtype=np.float32)
        out[indices[0]] = scores[0]
        return out

    def _sparse_scores(self, query: str) -> np.ndarray:
        return np.asarray(
            self._bm25.get_scores(tokenize(query)), dtype=np.float32
        )

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.documents:
            return []
        dense = self._normalize(self._dense_scores(query))
        sparse = self._normalize(self._sparse_scores(query))
        combined = self.dense_weight * dense + (1 - self.dense_weight) * sparse
        ranked = np.argsort(-combined)[:k]
        return [
            {
                "text": self.documents[i],
                "metadata": self.metadatas[i],
                "score": float(combined[i]),
                "dense_score": float(dense[i]),
                "sparse_score": float(sparse[i]),
            }
            for i in ranked
        ]
