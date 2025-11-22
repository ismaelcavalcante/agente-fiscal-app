import streamlit as st
from langchain_openai import ChatOpenAI
from langfuse import Langfuse

# módulos locais
from utils.logs import logger
from utils.messages import (
    convert_history_to_lc,
    dict_to_lc,
    lc_to_dict
)

from rag.qdrant import build_retriever
from rag.web import build_web_tool
from graph.builder import build_graph


# ==============================
# 🟦 CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="Consultor Fiscal IA", page_icon="💼")
st.title("💼 Assistente Fiscal Inteligente")

# ==============================
# 🔑 CREDENCIAIS
# ==============================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]

# ==============================
# 📌 PERFIL DO CLIENTE
# ==============================
DEFAULT_PROFILE = """
Empresa do regime geral/presumido, comércio varejista, atuação em múltiplos estados.
"""


# ==============================
# 🧠 INICIALIZAÇÃO DE ESTADO
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "perfil_cliente" not in st.session_state:
    st.session_state.perfil_cliente = DEFAULT_PROFILE

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"

if "mcp" not in st.session_state:
    st.session_state.mcp = None


# ==============================
# 🧠 LLM
# ==============================
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
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
# 🧠 LANGFUSE (SDK NOVO)
# ==============================
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
)


# ==============================
# 🔗 GRAFO
# ==============================
app_graph = build_graph(llm, retriever, web_tool)


# ==============================
# 💬 EXIBIR HISTÓRICO
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ==============================
# 🔎 VISUALIZAR MCP
# ==============================
with st.expander("🔍 Ver contexto MCP"):
    if st.session_state.mcp:
        st.json(st.session_state.mcp.model_dump())
    else:
        st.caption("Nenhum MCP gerado ainda.")


# ==============================
# ✏️ INPUT
# ==============================
user_input = st.chat_input("Digite sua pergunta tributária...")


# ==============================
# 🚀 PROCESSAMENTO
# ==============================
if user_input:

    # Exibe no chat
    st.chat_message("user").write(user_input)

    # Salva no histórico
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Converte histórico para LangChain
    lc_messages = convert_history_to_lc(st.session_state.messages)

    try:
        # EXECUTA O GRAFO (sem callbacks)
        result = app_graph.invoke(
            {
                "messages": lc_messages,
                "perfil_cliente": st.session_state.perfil_cliente,
                "thread_id": st.session_state.thread_id,
            }
        )

        # Resposta final
        ai_msg = result["messages"][-1]

        # Exibe
        st.chat_message("assistant").write(ai_msg.content)

        # Salva no histórico
        st.session_state.messages.append(
            lc_to_dict(ai_msg)
        )

        # SALVAR MCP (SE EXISTIR)
        if "mcp" in result:
            st.session_state.mcp = result["mcp"]

        # ==============================
        # 📌 Tracking manual do Langfuse (novo SDK)
        # ==============================
        langfuse.generation(
            name="resposta_final",
            model="gpt-4o-mini",
            input=user_input,
            output=ai_msg.content,
            metadata={
                "thread_id": st.session_state.thread_id,
                "perfil_cliente": st.session_state.perfil_cliente,
            }
        )

    except Exception as e:
        st.error("Ocorreu um erro durante a análise.")
        logger.error(f"ERRO NO GRAFO: {e}")