"""RAG pipeline: hybrid retrieval, reranking, and multi-agent answering.

Ships with a small built-in demo corpus so the pipeline runs end to end
without any external data. Swap in real documents with load_documents()
and a real embedding model to use it on your own data.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.agents import build_agent_graph
from rag.reranker import get_reranker
from rag.vector_store import HybridVectorStore

DEMO_DOCS = [
    {
        "text": "MLServe is an end-to-end ML platform. It covers distributed "
        "training with PyTorch, model optimization with ONNX, real-time "
        "serving with FastAPI, automated evaluation with CI quality gates, "
        "and production monitoring with Prometheus and Grafana.",
        "source": "mlserve-overview",
    },
    {
        "text": "The serving layer exposes a POST /predict endpoint for single "
        "predictions and POST /predict/batch for batch inference. A/B testing "
        "splits traffic 90 percent to the stable model and 10 percent to the "
        "canary model, and the canary rolls back automatically if its error "
        "rate rises above the configured threshold.",
        "source": "serving-guide",
    },
    {
        "text": "The evaluation engine runs a golden test dataset of labeled "
        "examples and computes accuracy, precision, recall, and F1. A CI "
        "quality gate blocks deployment when accuracy drops below 0.85 or "
        "when regression against the baseline exceeds 0.02.",
        "source": "evaluation-guide",
    },
    {
        "text": "Monitoring tracks request counts, latency histograms, and "
        "error rates in Prometheus. Drift detection uses the population "
        "stability index and the Kolmogorov-Smirnov test to compare live "
        "input distributions against a stored reference, and alerts fire to "
        "Slack when thresholds are crossed.",
        "source": "monitoring-guide",
    },
    {
        "text": "Model optimization exports PyTorch models to ONNX and applies "
        "INT8 or FP16 quantization. Benchmarking measures p50, p95, and p99 "
        "latency plus throughput in requests per second, and produces a "
        "before and after comparison report.",
        "source": "optimization-guide",
    },
]


class RAGPipeline:
    def __init__(self, embeddings=None, dense_weight: float = 0.7,
                 prefer_model_reranker: bool = False):
        self.store = HybridVectorStore(embeddings, dense_weight)
        self.reranker = get_reranker(prefer_model=prefer_model_reranker)
        self.graph = build_agent_graph(self.store, self.reranker)
        self.load_documents([d["text"] for d in DEMO_DOCS], DEMO_DOCS)

    def load_documents(self, texts: list[str], metadatas: list[dict] | None = None,
                       chunk_size: int = 500, chunk_overlap: int = 50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks: list[str] = []
        chunk_metas: list[dict] = []
        for i, text in enumerate(texts):
            meta = (metadatas or [{}] * len(texts))[i]
            for chunk in splitter.split_text(text):
                chunks.append(chunk)
                chunk_metas.append(meta)
        self.store.add_documents(chunks, chunk_metas)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        candidates = self.store.search(query, k=k)
        return self.reranker.rerank(query, candidates, top_k=min(3, k))

    def answer(self, question: str) -> dict:
        result = self.graph.invoke(
            {
                "question": question,
                "plan": [],
                "evidence": [],
                "draft_answer": "",
                "verdict": "",
                "answer": "",
            }
        )
        return {
            "answer": result["answer"],
            "verdict": result["verdict"],
            "sources": [
                doc.get("metadata", {}).get("source", "unknown")
                for doc in result["evidence"][:3]
            ],
        }
