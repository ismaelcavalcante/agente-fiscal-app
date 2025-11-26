from graph.router import node_router


def test_router_rag():
    state = {"ultima_pergunta": "Qual a alíquota do IBS?"}
    result = node_router(state)
    assert result["__route__"] == "RAG"


def test_router_web():
    state = {"ultima_pergunta": "pesquise ICMS no paraná"}
    result = node_router(state)
    assert result["__route__"] == "WEB"


def test_router_default():
    state = {"ultima_pergunta": "preciso de ajuda"}
    result = node_router(state)
    assert result["__route__"] == "RAG"