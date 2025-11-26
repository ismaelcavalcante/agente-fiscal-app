# app_web.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from utils.logs import logger
from rag.pipeline import HybridRAGPipeline
from rag.qdrant import QdrantRetriever
from rag.web import WebSearch
from graph.builder import build_graph

from langfuse import Langfuse

# Components UI
from components.perfil_select import selecionar_perfil
from components.perfil_form import editar_perfil_form
from components.perfil_upload import upload_perfil_json


# ===========================
# Config Streamlit
# ===========================
st.set_page_config(page_title="Consultor Fiscal IA", page_icon="💼")
st.title("💼 Assistente Fiscal Inteligente")


# ===========================
# Sessão: Perfis
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

    st.subheader("📤 Upload JSON do Perfil")
    upload_perfil_json()


# ===========================
# Bloquear fluxo sem perfil
# ===========================
if not st.session_state.perfil_ativo:
    st.warning("Selecione ou crie um perfil na lateral para começar.")
    st.stop()

perfil_cliente = st.session_state.perfis[st.session_state.perfil_ativo]


# ===========================
# Histórico
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"


def sanitize_history():
    """Converte manualmente mensagens soltas em objetos HumanMessage/AIMessage."""
    msgs = []
    for msg in st.session_state.messages:
        if isinstance(msg, BaseMessage):
            msgs.append(msg)
        elif isinstance(msg, dict):  # fallback caso algo tenha vindo em formato sujo
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                msgs.append(AIMessage(content=content))
            else:
                msgs.append(HumanMessage(content=content))
    st.session_state.messages = msgs


sanitize_history()


# ===========================
# Inicializar LLM
# ===========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.1
)


# ===========================
# Inicializar RAG Híbrido
# ===========================
retriever = QdrantRetriever(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"],
    collection="leis_fiscais_v1",
    embedding_model="text-embedding-3-large",
    openai_key=st.secrets["OPENAI_API_KEY"],
)

rag_pipeline = HybridRAGPipeline(
    qdrant_retriever=retriever,
    llm=llm,
    vector_top_k=6,
    final_top_k=4,
)


# ===========================
# Inicializar Web Search
# ===========================
web_tool = WebSearch(api_key=st.secrets["TAVILY_API_KEY"])


# ===========================
# Inicializar Langfuse
# ===========================
langfuse = Langfuse(
    public_key=st.secrets["LANGFUSE_PUBLIC_KEY"],
    secret_key=st.secrets["LANGFUSE_SECRET_KEY"]
)


# ===========================
# Construir Grafo
# ===========================
app_graph = build_graph(llm=llm, retriever=rag_pipeline, web_tool=web_tool)


# ===========================
# Exibir histórico
# ===========================
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)


# ===========================
# Entrada do usuário
# ===========================
user_input = st.chat_input("Digite sua pergunta tributária...")


# ===========================
# Pipeline de execução
# ===========================
if user_input:
    # 1) registrar mensagem
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)
    st.chat_message("user").write(user_input)

    sanitize_history()

    # 2) state inicial para LangGraph
    state = {
        "messages": list(st.session_state.messages),  # imutável
        "perfil_cliente": perfil_cliente,
        "ultima_pergunta": user_input,
    }

    # 3) Execução segura do grafo
    try:
        result = app_graph.invoke(
            state,
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )

        msgs = result.get("messages", [])
        if not msgs:
            st.error("Nenhuma resposta foi gerada pelo grafo.")
            logger.error("Grafo retornou messages vazio!")
            st.stop()

        ai_msg = msgs[-1]

        st.chat_message("assistant").write(ai_msg.content)
        st.session_state.messages.append(ai_msg)

        # 4) Log no Langfuse
        langfuse.generation(
            name="resposta_final",
            model="gpt-4o-mini",
            input=user_input,
            output=ai_msg.content,
        )

    except Exception as e:
        logger.error(f"Erro ao executar grafo: {e}")
        st.error("Erro interno ao processar sua pergunta.")