from graph.nodes import node_rag_qdrant, node_web_search, node_generate_final
from langchain_core.messages import HumanMessage, AIMessage


class MockRAG:
    def run(self, q, p):
        return [{"source": "RAG"}], "conteudo RAG"


class MockWeb:
    def execute(self, q):
        return {"answer": "web ctx", "sources": [{"source": "WEB"}]}


class MockLLM:
    def invoke(self, m):
        return AIMessage(content="resposta LLM")


def test_node_rag():
    state = {"ultima_pergunta": "x", "perfil_cliente": "y"}
    res = node_rag_qdrant(state, MockRAG())
    assert res["rag_ok"] is True
    assert "conteudo RAG" in res["contexto_juridico_bruto"]


def test_node_web():
    state = {"rag_ok": False, "ultima_pergunta": "x"}
    res = node_web_search(state, MockWeb())
    assert res["rag_ok"] is True
    assert "web ctx" in res["contexto_juridico_bruto"]


def test_node_generate_final():
    state = {
        "ultima_pergunta": "Qual a alíquota?",
        "perfil_cliente": "Simples",
        "contexto_juridico_bruto": "ctx",
        "sources_data": [{"source": "RAG"}],
        "messages": [HumanMessage(content="oi")]
    }

    res = node_generate_final(state, MockLLM())
    assert isinstance(res["messages"][-1], AIMessage)