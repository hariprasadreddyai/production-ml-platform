"""Tests for the RAG pipeline: retrieval, reranking, agents."""

from rag.pipeline import DEMO_DOCS, RAGPipeline
from rag.reranker import LexicalReranker
from rag.vector_store import HashEmbeddings, HybridVectorStore


def test_hash_embeddings_are_deterministic():
    embedder = HashEmbeddings()
    first = embedder.embed("hello world")
    second = embedder.embed("hello world")
    assert (first == second).all()
    assert abs(float((first**2).sum()) - 1.0) < 1e-5


def test_hybrid_store_returns_relevant_doc():
    store = HybridVectorStore(HashEmbeddings())
    store.add_documents(
        ["The cat sits on the mat", "Quantum computing uses qubits"],
        [{"source": "a"}, {"source": "b"}],
    )
    results = store.search("quantum computing qubits", k=2)
    assert results[0]["metadata"]["source"] == "b"
    assert results[0]["score"] >= results[1]["score"]


def test_hybrid_store_empty_returns_empty():
    store = HybridVectorStore(HashEmbeddings())
    assert store.search("anything") == []


def test_lexical_reranker_orders_by_overlap():
    reranker = LexicalReranker()
    docs = [
        {"text": "dogs and cats are pets"},
        {"text": "quantum entanglement and superposition"},
    ]
    ranked = reranker.rerank("quantum superposition", docs, top_k=2)
    assert "quantum" in ranked[0]["text"]
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_rag_pipeline_retrieves_serving_docs():
    pipeline = RAGPipeline()
    results = pipeline.retrieve("How does A/B testing split traffic?", k=5)
    assert results
    combined = " ".join(d["text"] for d in results)
    assert "90" in combined or "canary" in combined.lower()


def test_rag_pipeline_answer_is_grounded():
    pipeline = RAGPipeline()
    result = pipeline.answer("What does the evaluation engine check?")
    assert result["verdict"] == "PASS"
    assert result["answer"]
    assert result["sources"]
    assert any(s.endswith("guide") for s in result["sources"])


def test_rag_pipeline_demo_corpus_loaded():
    assert len(DEMO_DOCS) == 5
    pipeline = RAGPipeline()
    assert len(pipeline.store.documents) >= 5
