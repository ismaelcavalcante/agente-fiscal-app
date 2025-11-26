from langchain_core.messages import HumanMessage, AIMessage
from tests.helpers.fake_streamlit import FakeStreamlit
from tests.helpers.fake_graph import FakeGraph
import app_web


def test_message_flow(monkeypatch):
    st = FakeStreamlit()
    st.session_state.state["messages"] = []
    st.session_state.state["perfis"] = {"EMP": {"regime": "Lucro Presumido"}}
    st.session_state.state["perfil_ativo"] = "EMP"

    fake_graph = FakeGraph(answer="OK")
    monkeypatch.setattr("app_web.app_graph", fake_graph)
    monkeypatch.setattr("app_web.st", st)

    # simula entrada de usuário
    user_msg = HumanMessage(content="ICMS sobre energia?")
    st.session_state.state["messages"].append(user_msg)

    # simula execução
    result = fake_graph.invoke({"messages": [user_msg]})

    msgs = result["messages"]

    assert isinstance(msgs[-1], AIMessage)
    assert msgs[-1].content == "OK"