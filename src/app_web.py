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

from langchain_core.messages import HumanMessage, AIMessage
import json


# ===============================
#  Configuração Streamlit
# ===============================
st.set_page_config(page_title="Consultor Fiscal IA", page_icon="💼")
st.title("💼 Assistente Fiscal Inteligente")


# ===============================
#  Sessão: Perfis
# ===============================
if "perfis" not in st.session_state:
    st.session_state.perfis = {}
if "perfil_ativo" not in st.session_state:
    st.session_state.perfil_ativo = None


# ===============================
#  Sidebar: seleção/edição de perfis
# ===============================
with st.sidebar:
    st.header("🏢 Perfis da Empresa")

    selecionar_perfil()

    st.subheader("➕ Criar / Editar Perfil")
    editar_perfil_form()

    st.subheader("📤 Upload JSON do Perfil")
    upload_perfil_json()


# ===============================
#  Impedir uso sem perfil carregado
# ===============================
if not st.session_state.perfil_ativo:
    st.warning("Selecione ou crie um perfil na lateral para começar.")
    st.stop()

perfil_cliente = st.session_state.perfis[st.session_state.perfil_ativo]  # ← Como dict


# ===============================
#  Sessão de histórico de mensagens
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []  # Sempre HumanMessage / AIMessage
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"


# ===============================
#  Inicializar o modelo LLM
# ===============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.2
)


# ===============================
#  RAG + Web Search
# ===============================
retriever = build_retriever(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"],
    collection="leis_fiscais_v1",
    embedding_model="text-embedding-3-large",
    openai_key=st.secrets["OPENAI_API_KEY"],
)

web_tool = build_web_tool(st.secrets["TAVILY_API_KEY"])


# ===============================
#  Langfuse Tracking
# ===============================
langfuse = Langfuse(
    public_key=st.secrets["LANGFUSE_PUBLIC_KEY"],
    secret_key=st.secrets["LANGFUSE_SECRET_KEY"]
)


# ===============================
#  Construir o Grafo
# ===============================
app_graph = build_graph(llm, retriever, web_tool)


# ===============================
#  Mostrar histórico no chat
# ===============================
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)


# ===============================
#  Entrada do Usuário
# ===============================
user_input = st.chat_input("Faça sua pergunta tributária...")


# ===============================
#  Execução Principal
# ===============================
if user_input:

    # Mostrar mensagem do usuário
    st.chat_message("user").write(user_input)

    # Armazenar como HumanMessage
    st.session_state.messages.append(HumanMessage(content=user_input))

    try:
        # Invocar grafo
        result = app_graph.invoke(
            {
                "messages": st.session_state.messages,
                "perfil_cliente": perfil_cliente,
            },
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

        # Extraindo a última mensagem (AI)
        ai_msg = result["messages"][-1]

        # Mostrar no chat
        st.chat_message("assistant").write(ai_msg.content)

        # Armazenar histórico
        st.session_state.messages.append(ai_msg)

        # Langfuse tracking
        langfuse.generation(
            name="resposta_final",
            model="gpt-4o-mini",
            input=user_input,
            output=ai_msg.content
        )

    except Exception as e:
        logger.error(f"Erro no fluxo: {e}")
        st.error("Ocorreu um erro durante a análise. Verifique os logs.")