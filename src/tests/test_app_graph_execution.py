from langchain_core.messages import HumanMessage, AIMessage
from tests.helpers.fake_streamlit import FakeStreamlit
from tests.helpers.fake_graph import FakeGraph
import app_web


def test_graph_integration(monkeypatch):
    st = FakeStreamlit()
    st.session_state.state["messages"] = []
    st.session_state.state["perfis"] = {"A": {"regime": "Simples"}}
    st.session_state.state["perfil_ativo"] = "A"

    # Entrada do usuário simulada
    st.session_state.state["chat_input"] = "Qual a alíquota do IBS?"

    # Mock do grafo
    fake_graph = FakeGraph(answer="Resposta simulada")
    monkeypatch.setattr("app_web.app_graph", fake_graph)

    # Mock do Streamlit
    def fake_input(_):
        return "Pergunta teste"

    monkeypatch.setattr("app_web.st", st)
    monkeypatch.setattr("app_web.st.chat_input", fake_input)

    # Mock do modelo
    monkeypatch.setattr("app_web.llm", None)

    # Rodar ciclo do app
    # Forçamos a execução até o ponto desejado
    try:
        app_web.user_input = fake_input(None)
    except Exception:
        pass

    # Ao final, verificamos histórico
    msgs = st.session_state.state["messages"]
    assert isinstance(msgs[-1], AIMessage)
    assert msgs[-1].content == "Resposta simulada"