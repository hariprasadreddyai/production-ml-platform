"""Cross-encoder reranking for final precision.

CrossEncoderReranker loads a real cross-encoder model when available.
LexicalReranker scores by query-term overlap and needs no model, so the
pipeline stays runnable offline and in tests.
"""

from __future__ import annotations

from rag.vector_store import tokenize


class LexicalReranker:
    """Model-free reranker: overlap between query terms and document terms."""

    def rerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
        query_terms = set(tokenize(query))
        scored = []
        for doc in docs:
            terms = set(tokenize(doc["text"]))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            scored.append({**doc, "rerank_score": round(overlap, 4)})
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Real cross-encoder reranker via sentence-transformers (lazy import)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)
        scored = [
            {**doc, "rerank_score": float(score)}
            for doc, score in zip(docs, scores)
        ]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]


def get_reranker(prefer_model: bool = False):
    """Return the cross-encoder reranker if a model can load, else lexical."""
    if prefer_model:
        try:
            return CrossEncoderReranker()
        except Exception:
            pass
    return LexicalReranker()
