# tests/test_rerank_vector.py

from rag.rerank_vector import VectorReranker


def test_vector_reranker():
    reranker = VectorReranker()

    docs = [
        {"page_content": "a"},
        {"page_content": "b"},
    ]

    ranked = reranker.rerank("x", docs, top_k=1)
    assert len(ranked) == 1