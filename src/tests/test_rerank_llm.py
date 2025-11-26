# tests/test_rerank_llm.py

from rag.rerank_llm import LLMJudgeReranker
from langchain_core.messages import AIMessage


class MockLLM:
    def invoke(self, msgs):
        return AIMessage(content='[{"index": 0, "score": 10}, {"index": 1, "score": 1}]')


def test_llm_judge_reranker():
    reranker = LLMJudgeReranker(MockLLM())

    docs = [
        {"page_content": "doc1"},
        {"page_content": "doc2"},
    ]

    ranked = reranker.rerank("pergunta", docs, top_k=1)
    assert ranked[0]["page_content"] == "doc1"