import streamlit as st
from langchain_openai import ChatOpenAI
from langfuse import Langfuse

# Módulos internos
from utils.logs import logger
from utils.messages import (
    convert_history_to_lc,
    lc_to_dict
)

from rag.qdrant import build_retriever
from rag.web import build_web_tool
from graph.builder import build_graph


# ==============================
# 🌐 CONFIGURAÇÃO DO STREAMLIT
# ==============================
st.set_page_config(page_title="Consultor Fiscal IA", page_icon="💼")
st.title("💼 Assistente Fiscal Inteligente")


# ==============================
# 🔑 CREDENCIAIS (Streamlit Secrets)
# ==============================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]


# ==============================
# 🧠 ESTADO INICIAL
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "perfil_cliente" not in st.session_state:
    st.session_state.perfil_cliente = """
Empresa do regime geral/presumido, comércio varejista,
atuando em múltiplos estados.
"""

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"

if "mcp" not in st.session_state:
    st.session_state.mcp = None


# ==============================
# 🤖 LLM
# ==============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)


# ==============================
# 🔍 RAG TOOLS
# ==============================
retriever = build_retriever(
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    collection_name="leis_fiscais_v1",
    embedding_model_name="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
)

web_tool = build_web_tool(TAVILY_API_KEY)


# ==============================
# 🧭 LANGFUSE (SDK novo)
# ==============================
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY
)


# ==============================
# 🔗 GRAFO (LangGraph)
# ==============================
app_graph = build_graph(llm, retriever, web_tool)


# ==============================
# 📜 HISTÓRICO NO CHAT
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ==============================
# 🔎 VISUALIZAÇÃO DO MCP
# ==============================
with st.expander("🔍 Ver contexto MCP"):
    if st.session_state.mcp:
        st.json(st.session_state.mcp.model_dump())
    else:
        st.caption("Nenhum MCP disponível ainda.")


# ==============================
# 💬 ENTRADA DO USUÁRIO
# ==============================
user_input = st.chat_input("Digite sua pergunta tributária...")


# ==============================
# 🚀 EXECUÇÃO DO AGENTE
# ==============================
if user_input:

    # Mostra no chat
    st.chat_message("user").write(user_input)

    # Salva histórico
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Converte histórico para LangChain messages
    lc_messages = convert_history_to_lc(st.session_state.messages)

    try:
        # EXECUTA O GRAFO → ATENÇÃO AQUI: CONFIGURABLE!!!
        result = app_graph.invoke(
            {
                "messages": lc_messages,
                "perfil_cliente": st.session_state.perfil_cliente,
            },
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
        )

        # PEGAR A RESPOSTA FINAL
        ai_msg = result["messages"][-1]

        # Mostrar no chat
        st.chat_message("assistant").write(ai_msg.content)

        # Armazenar no histórico
        st.session_state.messages.append(
            lc_to_dict(ai_msg)
        )

        # Se MCP foi gerado → salvar
        if "mcp" in result:
            st.session_state.mcp = result["mcp"]

        # TRACKING LANGFUSE (manual)
        langfuse.generation(
            name="resposta_final",
            model="gpt-4o-mini",
            input=user_input,
            output=ai_msg.content,
            metadata={
                "thread_id": st.session_state.thread_id,
                "perfil_cliente": st.session_state.perfil_cliente
            }
        )

    except Exception as e:
        st.error("Ocorreu um erro durante a análise.")
        logger.error(f"ERRO NO GRAFO: {e}")