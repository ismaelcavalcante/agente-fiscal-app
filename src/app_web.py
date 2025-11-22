import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from typing import Union, TypedDict, Annotated, List, Dict, Any
import os

# LangChain e LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# NOVO SDK LANGFUSE
from langfuse.callback import CallbackHandler

# ----------------------------------------------------
# GARANTE QUE TODAS AS MENSAGENS DO SESSION STATE SEJAM DICT
# ----------------------------------------------------
def normalize_message(msg):
    """Padroniza mensagens para dict(role, content)."""
    if isinstance(msg, dict):
        return msg
    
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    
    if isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}

    return {"role": "assistant", "content": str(msg)}

def message_to_lc(msg):
    """Converte dict → HumanMessage | AIMessage"""
    if msg["role"] == "user":
        return HumanMessage(content=msg["content"])
    else:
        return AIMessage(content=msg["content"])

# ----------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ----------------------------------------------------
st.set_page_config(
    page_title="Agente Fiscal v4.3 (Corrigido)",
    page_icon="⚖️",
    layout="wide"
)

# ----------------------------------------------------
# SESSION STATE (SEGURO)
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# NORMALIZA TODAS AS ANTIGAS MENSAGENS
st.session_state.messages = [normalize_message(m) for m in st.session_state.messages]

if "client_profile" not in st.session_state:
    st.session_state.client_profile = """{
"nome_empresa": "Construtora Alfa Ltda",
"cnae_principal": "4120-4/00 (Construção de Edifícios)",
"regime_tributario": "Simples Nacional",
"faturamento_anual": "R$ 3.000.000,00"
}"""

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

if "db_count" not in st.session_state:
    st.session_state.db_count = 0

# ----------------------------------------------------
# CONSTANTES
# ----------------------------------------------------
NOME_DA_COLECAO = "leis_fiscais_v1"
MODELO_LLM = st.secrets.get("MODELO_LLM", "gpt-4o")
MODELO_EMBEDDING = st.secrets.get("MODELO_EMBEDDING", "text-embedding-3-large")

# ----------------------------------------------------
# ESTADO DO GRAFO
# ----------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], lambda x, y: x + y]  
    perfil_cliente: str
    sources_data: List[Dict[str, Any]]
    thread_id: str
    contexto_juridico_bruto: str
    mcp_data: str

# ----------------------------------------------------
# CARREGAR SERVIÇOS
# ----------------------------------------------------
@st.cache_resource
def carregar_servicos_e_grafo():

    # validação dos secrets
    required_secrets = [
        "OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY"
    ]
    for sname in required_secrets:
        if sname not in st.secrets:
            st.error(f"Secret faltando: {sname}")
            return None, None

    # instâncias
    llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model=MODELO_LLM,
        temperature=0
    )

    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"],
        model=MODELO_EMBEDDING
    )

    qdrant_client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    # retriever SEM filtros problemáticos
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name=NOME_DA_COLECAO,
        embeddings=embeddings
    )
    
    retriever_obj = qdrant_store.as_retriever(search_kwargs={"k": 5})

    biblioteca_tool = create_retriever_tool(
        retriever_obj,
        "biblioteca_fiscal",
        "Use esta ferramenta para buscar legislação, artigos, LC 214, EC 132, IBS, CBS, regras fiscais."
    )

    web_tool = TavilySearchResults(max_results=3)

    tools = [biblioteca_tool, web_tool]

    # -------------------------------------------------
    # DEFINIÇÃO DOS NODES DO GRAFO
    # -------------------------------------------------

    def roteador(state: AgentState) -> str:
        last = state["messages"][-1]
        last_lc = message_to_lc(last)

        tool_router = llm.bind_tools(tools)

        smsg = SystemMessage(content="""
Você é um assistente jurídico tributário especializado.
Se a pergunta envolver leis, alíquotas, artigos, IBS, CBS, EC 132, use SEMPRE 'biblioteca_fiscal'.
Se for sobre notícias ou informações atuais, use 'tavily_search_results'.
""")

        resp = tool_router.invoke([smsg, last_lc])

        # nenhuma ferramenta detectada
        if not resp.tool_calls:

            txt = last["content"].lower()
            keywords = ["lei", "artigo", "imposto", "alíquota", "tribut", "ibs", "cbs", "ec 132", "lc 214"]

            if any(k in txt for k in keywords):
                return "usar_biblioteca"

            return "gerar_resposta_sem_contexto"

        tool = resp.tool_calls[0]["name"]

        if tool == "biblioteca_fiscal":
            return "usar_biblioteca"

        if tool == "tavily_search_results":
            return "usar_web"

        return "gerar_resposta_sem_contexto"

    def no_biblioteca(state: AgentState):
        pergunta = state["messages"][-1]["content"]
        perfil = state["perfil_cliente"]
        query = f"{pergunta} (Contexto: {perfil})"

        try:
            docs = retriever_obj.invoke(query) or []
        except:
            docs = []

        contexto = "\n\n".join([f"Fonte: {d.metadata.get('source','Lei')} | {d.page_content}" for d in docs])
        fontes = [{"source": d.metadata.get("source","Lei"), "page": d.metadata.get("page"), "type": d.metadata.get("document_type","Lei")} for d in docs]

        return {"sources_data": fontes, "contexto_juridico_bruto": contexto}

    def no_web(state: AgentState):
        pergunta = state["messages"][-1]["content"]

        try:
            docs = web_tool.invoke(pergunta) or []
        except:
            docs = []

        contexto = "\n---\n".join([d.get("content","") for d in docs])
        fontes = [{"source": "Web", "page": None, "type": "WEB", "content": d.get("content","")} for d in docs]

        return {"sources_data": fontes, "contexto_juridico_bruto": contexto}

    def no_gerar(state: AgentState):
        perfil = state["perfil_cliente"]
        contexto = state["contexto_juridico_bruto"] or "Nenhuma legislação encontrada."
        pergunta = state["messages"][-1]["content"]

        system_prompt = f"""
Você é um consultor tributário sênior.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO ENCONTRADO:
{contexto}

DIRETRIZES:
1. Responda com base no contexto acima.
2. Cite artigos e leis quando mencionados.
3. Se o contexto não ajudar, diga isso ao usuário.
"""

        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=pergunta)
        ]

        out = llm.invoke(msgs)

        return {
            "messages": [
                {"role": "assistant", "content": out.content}
            ],
            "mcp_data": ""
        }

    # -------------------------------------------------
    # COMPILAÇÃO DO GRAFO
    # -------------------------------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node("usar_biblioteca", no_biblioteca)
    workflow.add_node("usar_web", no_web)
    workflow.add_node("gerar_resposta", no_gerar)
    workflow.add_node("gerar_resposta_sem_contexto", no_gerar)

    workflow.add_conditional_edges(
        START, roteador,
        {
            "usar_biblioteca": "usar_biblioteca",
            "usar_web": "usar_web",
            "gerar_resposta_sem_contexto": "gerar_resposta_sem_contexto"
        }
    )

    workflow.add_edge("usar_biblioteca", "gerar_resposta")
    workflow.add_edge("usar_web", "gerar_resposta")
    workflow.add_edge("gerar_resposta", END)
    workflow.add_edge("gerar_resposta_sem_contexto", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # contagem do banco
    try:
        c = qdrant_client.count(NOME_DA_COLECAO, exact=True)
        st.session_state.db_count = c.count
    except:
        st.session_state.db_count = 0

    return app, True

# ----------------------------------------------------
# CARREGA SERVIÇOS
# ----------------------------------------------------
agente, langfuse_ok = carregar_servicos_e_grafo()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configurações")
    st.subheader("Perfil do Cliente")

    perfil_edit = st.text_area("JSON do Perfil:", st.session_state.client_profile, height=200)

    if st.button("Salvar Perfil"):
        st.session_state.client_profile = perfil_edit
        st.success("Perfil atualizado!")

    st.divider()
    st.info(f"Documentos na Base: {st.session_state.db_count}")

# ----------------------------------------------------
# CHAT MAIN
# ----------------------------------------------------
st.title("🤖 Agente Fiscal v4.3")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Digite sua dúvida tributária...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação..."):

            mensagens_lc = [message_to_lc(m) for m in st.session_state.messages]

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }

            if langfuse_ok:
                config["callbacks"] = [
                    CallbackHandler(
                        user_id="user",
                        session_id=st.session_state.thread_id
                    )
                ]

            entrada = {
                "messages": st.session_state.messages,
                "perfil_cliente": st.session_state.client_profile,
                "thread_id": st.session_state.thread_id,
                "contexto_juridico_bruto": "",
                "sources_data": [],
                "mcp_data": ""
            }

            try:
                resposta = ""

                for event in agente.stream(entrada, config, stream_mode="values"):
                    if "messages" in event:
                        last = event["messages"][-1]
                        resposta = last["content"]

                if resposta:
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                else:
                    st.warning("Nenhuma resposta foi gerada.")

            except Exception as e:
                st.error(f"Erro inesperado: {e}")