import pytest
from rag.qdrant import QdrantRetriever


class MockClient:
    def search(self, **kwargs):
        class P:
            def __init__(self, text):
                self.payload = {"page_content": text}
        return [P("doc1"), P("doc2")]


class MockEmbed:
    def embed_query(self, x): return [0.1, 0.2]


def test_qdrant_retriever_basic(monkeypatch):
    retriever = QdrantRetriever("url", "key", "collection", "model", "openai")

    retriever.client = MockClient()
    retriever.embeddings = MockEmbed()

    docs = retriever.query("pergunta", "perfil")
    assert len(docs) == 2
    assert docs[0]["page_content"] == "doc1"