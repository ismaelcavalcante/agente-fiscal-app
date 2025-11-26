from rag.pipeline import HybridRAGPipeline


class MockRetriever:
    def query(self, q, p, limit=12):
        return [
            {"page_content": "A", "metadata": {}},
            {"page_content": "B", "metadata": {}},
        ]


class MockVector:
    def rerank(self, q, docs, top_k):
        return docs


class MockJudge:
    def rerank(self, q, docs, top_k):
        return docs[:1]


def test_rag_pipeline(monkeypatch):
    pipe = HybridRAGPipeline(
        qdrant_retriever=MockRetriever(),
        llm=None
    )
    pipe.vector_reranker = MockVector()
    pipe.llm_reranker = MockJudge()

    meta, ctx = pipe.run("pergunta", "perfil")

    assert ctx.strip() == "A"