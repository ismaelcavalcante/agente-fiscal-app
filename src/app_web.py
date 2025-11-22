import streamlit as st
from langchain_openai import ChatOpenAI
from langfuse import Langfuse

from components.perfil_select import selecionar_perfil
from components.perfil_form import editar_perfil_form
from components.perfil_upload import upload_perfil_json

from graph.builder import build_graph
from rag.qdrant import build_retriever
from rag.web import build_web_tool
from utils.logs import logger

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json


# ===========================
# Streamlit Config
# ===========================
st.set_page_config(page_title="Consultor Fiscal IA", page_icon="💼")
st.title("💼 Assistente Fiscal Inteligente")


# ===========================
# Perfis do cliente
# ===========================
if "perfis" not in st.session_state:
    st.session_state.perfis = {}

if "perfil_ativo" not in st.session_state:
    st.session_state.perfil_ativo = None


with st.sidebar:
    st.header("🏢 Perfis da Empresa")
    selecionar_perfil()

    st.subheader("➕ Criar / Editar Perfil")
    editar_perfil_form()

    st.subheader("📤 Importar Perfil via JSON")
    upload_perfil_json()


if not st.session_state.perfil_ativo:
    st.warning("Selecione ou crie um perfil para continuar.")
    st.stop()

perfil_cliente = st.session_state.perfis[st.session_state.perfil_ativo]


# ===========================
# Histórico
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"


# ===========================
# Sanitização do Histórico
# ===========================
def sanitize_history():
    """Converte quaisquer mensagens antigas (dict) em mensagens LC."""
    fixed = []
    for msg in st.session_state.messages:
        if isinstance(msg, BaseMessage):
            fixed.append(msg)
            continue
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                fixed.append(AIMessage(content=content))
            else:
                fixed.append(HumanMessage(content=content))
    st.session_state.messages = fixed


sanitize_history()


# ===========================
# LLM
# ===========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.15
)


# ===========================
# RAG + TAVILY
# ===========================
retriever = build_retriever(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"],
    collection="leis_fiscais_v1",
    embedding_model="text-embedding-3-large",
    openai_key=st.secrets["OPENAI_API_KEY"],
)

web_tool = build_web_tool(st.secrets["TAVILY_API_KEY"])


# ===========================
# L angfuse
# ===========================
langfuse = Langfuse(
    public_key=st.secrets["LANGFUSE_PUBLIC_KEY"],
    secret_key=st.secrets["LANGFUSE_SECRET_KEY"]
)


# ===========================
# Grafo
# ===========================
app_graph = build_graph(llm, retriever, web_tool)


# ===========================
# Mostrar histórico
# ===========================
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)


# ===========================
# Entrada
# ===========================
user_input = st.chat_input("Escreva sua dúvida tributária...")


# ===========================
# Execução
# ===========================
if user_input:

    # registrar no histórico
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # construir state de entrada
    state_input = {
        "messages": st.session_state.messages,
        "perfil_cliente": perfil_cliente,
    }

    try:
        result = app_graph.invoke(
            state_input,
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )

        ai_msg = result["messages"][-1]

        st.chat_message("assistant").write(ai_msg.content)
        st.session_state.messages.append(ai_msg)

        langfuse.generation(
            name="resposta_final",
            model="gpt-4o-mini",
            input=user_input,
            output=ai_msg.content
        )

    except Exception as e:
        logger.error(f"Erro no fluxo: {e}")
        st.error("Erro durante o processamento. Veja os logs.")