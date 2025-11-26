from langchain_core.messages import AIMessage, HumanMessage
from tests.helpers.fake_streamlit import FakeStreamlit
from app_web import sanitize_history


def test_sanitize_history_converts_dict_messages(monkeypatch):
    st = FakeStreamlit()
    st.session_state.state["messages"] = [
        {"role": "user", "content": "Oi"},
        {"role": "assistant", "content": "Olá"}
    ]

    monkeypatch.setattr("app_web.st", st)

    sanitize_history()

    msgs = st.session_state.state["messages"]
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)


def test_sanitize_history_keeps_existing_messages(monkeypatch):
    st = FakeStreamlit()
    st.session_state.state["messages"] = [
        HumanMessage(content="Oi"),
        AIMessage(content="Olá")
    ]

    monkeypatch.setattr("app_web.st", st)

    sanitize_history()
    msgs = st.session_state.state["messages"]
    assert isinstance(msgs[0], HumanMessage)