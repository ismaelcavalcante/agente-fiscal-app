import streamlit as st
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

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
# 🔑 CREDENCIAIS (use st.secrets no Streamlit Cloud)
# ==============================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
LANGFUSE_PUBLIC = st.secrets["LANGFUSE_PUBLIC"]
LANGFUSE_SECRET = st.secrets["LANGFUSE_SECRET"]


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


# ==============================
# 🧠 MODELO LLM (OpenAI)
# ==============================
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2,
)


# ==============================
# 🔍 FERRAMENTAS (Qdrant + Web)
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
    public_key=LANGFUSE_PUBLIC,
    secret_key=LANGFUSE_SECRET,
)

callback_handler = CallbackHandler(
    user_id="user",
    session_id=st.session_state.thread_id
)


# ==============================
# 🔗 GRAFO
# ==============================
app_graph = build_graph(llm, retriever, web_tool)


# ==============================
# 💬 EXIBIR HISTÓRICO NO CHAT
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==============================
# 💬 EXIBIR CONTEXTO MCP
# ==============================
with st.expander("🔍 Ver contexto MCP"):
    if "mcp" in st.session_state:
        st.json(st.session_state["mcp"].model_dump())
    else:
        st.caption("Nenhum MCP gerado ainda.")

# ==============================
# ✏️ CAMPO DE ENTRADA
# ==============================
user_input = st.chat_input("Digite sua pergunta tributária...")


# ==============================
# 🚀 PROCESSAMENTO
# ==============================
if user_input:

    # 1. Exibe no chat
    st.chat_message("user").write(user_input)

    # 2. Salva no histórico
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 3. Converte histórico para LangChain
    lc_messages = convert_history_to_lc(st.session_state.messages)

    # 4. EXECUTA GRAFO
    try:
        result = app_graph.invoke(
            {
                "messages": lc_messages,
                "perfil_cliente": st.session_state.perfil_cliente,
                "thread_id": st.session_state.thread_id,
            },
            config={"callbacks": [callback_handler]}
        )

        ai_msg = result["messages"][-1]

        # 5. Exibe resposta
        st.chat_message("assistant").write(ai_msg.content)

        # 6. Salva no histórico como dict
        st.session_state.messages.append(
            lc_to_dict(ai_msg)
        )

    except Exception as e:
        st.error("Ocorreu um erro durante a análise.")
        logger.error(f"ERRO NO GRAFO: {e}")

if "mcp" in result:
    st.session_state["mcp"] = result["mcp"]