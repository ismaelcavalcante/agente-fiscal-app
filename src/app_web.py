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


# ===========================
# Streamlit Config
# ===========================
st.set_page_config(page_title="Consultor Fiscal IA", page_icon="💼")
st.title("💼 Assistente Fiscal Inteligente")


# ===========================
# Perfis no SessionState
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
# Bloquear se não houver perfil selecionado
# ===========================
if not st.session_state.perfil_ativo:
    st.warning("Selecione ou crie um perfil na lateral para começar.")
    st.stop()

perfil_cliente = st.session_state.perfis[st.session_state.perfil_ativo]


# ===========================
# Histórico no SessionState
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"


# ===========================
# Sanitização do histórico
# ===========================
def sanitize_history():
    fixed = []
    for msg in st.session_state.messages:
        if isinstance(msg, BaseMessage):
            fixed.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                fixed.append(AIMessage(content=content))
            else:
                fixed.append(HumanMessage(content=content))
        else:
            # qualquer outra coisa -> descartamos
            continue

    st.session_state.messages = fixed


sanitize_history()


# ===========================
# Inicializar LLM
# ===========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.15
)


# ===========================
# RAG e Web Search
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
# Langfuse
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
# Exibir histórico no chat
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
# Execução do fluxo
# ===========================
if user_input:

    # 1) Sempre adicionar a mensagem AO HISTÓRICO antes de tudo
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    # 2) Exibir imediatamente no chat
    st.chat_message("user").write(user_input)

    # 3) RE-SANITIZAR para garantir nenhum item inválido
    sanitize_history()

    # 4) Proteção: se o histórico estiver vazio por algum bug → Pare
    if len(st.session_state.messages) == 0:
        st.error("Erro interno: histórico vazio antes de chamar o grafo.")
        logger.error("ERROR: histórico vazio antes do grafo.")
        st.stop()

    # 5) Montar o state de entrada
    state_input = {
        "messages": st.session_state.messages,
        "perfil_cliente": perfil_cliente,
    }

    # Proteção: validar state_input
    if not isinstance(state_input, dict):
        st.error("State inválido!")
        logger.error(f"STATE INVÁLIDO: {state_input}")
        st.stop()

    if "messages" not in state_input or not isinstance(state_input["messages"], list):
        st.error("State.messages inválido!")
        logger.error(f"STATE MESSAGES INVALIDO: {state_input}")
        st.stop()

    # 6) Invocar o grafo
    try:
        result = app_graph.invoke(
            state_input,
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )

        ai_msg = result["messages"][-1]

        # 7) Exibir a resposta
        st.chat_message("assistant").write(ai_msg.content)

        # 8) Salvar no histórico
        st.session_state.messages.append(ai_msg)

        # 9) Tracking Langfuse
        langfuse.generation(
            name="resposta_final",
            model="gpt-4o-mini",
            input=user_input,
            output=ai_msg.content,
        )

    except Exception as e:
        logger.error(f"Erro no fluxo: {e}")
        st.error("Erro durante o processamento. Veja os logs.")