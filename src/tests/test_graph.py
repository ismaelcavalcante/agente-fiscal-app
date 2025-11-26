# tests/test_graph.py

from graph.builder import build_graph
from langchain_core.messages import HumanMessage, AIMessage


class MockRAGPipe:
    def run(self, q, p):
        return [{"source": "RAG"}], "ctx"


class MockWeb:
    def execute(self, q):
        return {"answer": "web ctx", "sources": [{"source": "WEB"}]}


class MockLLM:
    def invoke(self, m):
        return AIMessage(content="final")


def test_graph_flow():
    graph = build_graph(
        llm=MockLLM(),
        rag_pipeline=MockRAGPipe(),
        web_tool=MockWeb()
    )

    state = {
        "messages": [HumanMessage(content="Oi")],
        "ultima_pergunta": "Qual a alíquota?",
        "perfil_cliente": "x"
    }

    result = graph.invoke(state)
    assert isinstance(result["messages"][-1], AIMessage)