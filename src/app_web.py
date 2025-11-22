import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient, models
from typing import Union, TypedDict, Annotated, List, Dict, Any
import os

# LangChain / LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

# Langfuse novo
from langfuse.callbacks import CallbackHandler

# Protocol
try:
    from protocol import ConsultaContext, FonteDocumento
except ImportError:
    from pydantic import BaseModel
    class FonteDocumento(BaseModel):
        document_source: str
        page_number: Any
        chunk_index: int
        document_type: str
    class ConsultaContext(BaseModel):
        trace_id: str
        perfil_cliente: str
        pergunta_cliente: str
        contexto_juridico_bruto: str
        fontes_detalhadas: List[FonteDocumento]
        prompt_mestre: str


# -----------------------------------------------------------
# FUNÇÃO ROLE
# -----------------------------------------------------------
def get_streamlit_role(message: dict) -> str:
    """Padroniza role para chat_message"""
    return message["role"]


# -----------------------------------------------------------
# CONFIG STREAMLIT
# -----------------------------------------------------------
st.set_page_config(page_title="Agente Fiscal v4.4", page_icon="⚖️", layout="wide")

# Session State - GARANTIDO IMEDIATAMENTE
if "messages" not in st.session_state:
    st.session_state.messages = []

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


# -----------------------------------------------------------
# CONSTANTES
# -----------------------------------------------------------
NOME_DA_COLECAO = "leis_fiscais_v1"
MODELO_LLM = st.secrets.get("MODELO_LLM", "gpt-4o")
MODELO_EMBEDDING = st.secrets.get("MODELO_EMBEDDING", "text-embedding-3-large")


# -----------------------------------------------------------
# ESTADO DO AGENTE
# -----------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    perfil_cliente: str
    sources_data: List[Dict[str, Any]]
    thread_id: str
    contexto_juridico_bruto: str
    mcp_data: str


# -----------------------------------------------------------
# CARREGAMENTO DOS SERVIÇOS
# -----------------------------------------------------------
@st.cache_resource
def carregar_servicos_e_grafo():
    try:
        required = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY"]
        for s in required:
            if s not in st.secrets:
                st.error(f"Secret faltando: {s}")
                return None, None

        api_key = st.secrets["OPENAI_API_KEY"]
        q_url = st.secrets["QDRANT_URL"]
        q_key = st.secrets["QDRANT_API_KEY"]

        llm = ChatOpenAI(api_key=api_key, model=MODELO_LLM, temperature=0)
        embeddings = OpenAIEmbeddings(api_key=api_key, model=MODELO_EMBEDDING)
        client_qdrant = QdrantClient(url=q_url, api_key=q_key)

        # TAVILY
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

        # Langfuse novo
        lf_public = st.secrets.get("LANGFUSE_PUBLIC_KEY", "")
        lf_secret = st.secrets.get("LANGFUSE_SECRET_KEY", "")
        lf_host = st.secrets.get("LANGFUSE_BASE_URL", "")

        langfuse = None
        if lf_public:
            langfuse = CallbackHandler(
                public_key=lf_public,
                secret_key=lf_secret,
                host=lf_host
            )

        # Retriever SIMPLIFICADO (SEM FILTRO)
        qdrant_store = Qdrant(
            client=client_qdrant,
            collection_name=NOME_DA_COLECAO,
            embeddings=embeddings
        )

        retriever_obj = qdrant_store.as_retriever(search_kwargs={"k": 5})

        biblioteca_tool = create_retriever_tool(
            retriever_obj,
            "biblioteca_fiscal",
            "Use esta ferramenta para consultar leis fiscais."
        )

        web_tool = TavilySearchResults(max_results=3)
        tools = [biblioteca_tool, web_tool]

        # -----------------------------------------------------------
        # DEFINIÇÃO DOS NÓS DO GRAFO
        # -----------------------------------------------------------

        def roteador_de_ferramentas(state: AgentState) -> str:
            last = state["messages"][-1]
            router_llm = llm.bind_tools(tools)

            sm = SystemMessage(content="Você é um assistente tributário.")

            resp = router_llm.invoke([sm, last])

            if not resp.tool_calls:
                txt = last.content.lower()
                if any(k in txt for k in ["lei", "cbs", "ibs", "tribut", "ec 132", "lc 214"]):
                    return "usar_biblioteca"
                return "gerar_resposta_sem_contexto"

            tool = resp.tool_calls[0]["name"]

            if tool == "biblioteca_fiscal":
                return "usar_biblioteca"

            if tool == "tavily_search_results":
                return "usar_web"

            return "gerar_resposta_sem_contexto"

        def no_busca_biblioteca(state: AgentState):
            pergunta = state["messages"][-1].content
            perfil = state["perfil_cliente"]
            query = f"{pergunta} (contexto do cliente: {perfil})"

            try:
                docs = retriever_obj.invoke(query) or []
            except:
                docs = []

            contexto = "\n\n".join(
                f"Fonte: {d.metadata.get('source','Lei')}\n{d.page_content}"
                for d in docs
            )

            metadados = [
                {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "type": d.metadata.get("document_type")}
                for d in docs
            ]

            return {
                "sources_data": metadados,
                "contexto_juridico_bruto": contexto
            }

        def no_busca_web(state: AgentState):
            pergunta = state["messages"][-1].content

            try:
                docs = web_tool.invoke(pergunta) or []
            except:
                docs = []

            contexto = "\n---\n".join(doc.get("content", "") for doc in docs)

            metadados = [
                {"source": "WEB", "page": None, "type": "WEB", "content": doc.get("content")}
                for doc in docs
            ]

            return {
                "sources_data": metadados,
                "contexto_juridico_bruto": contexto
            }

        def no_gerador_resposta(state: AgentState):
            perfil = state["perfil_cliente"]
            contexto = state.get("contexto_juridico_bruto", "")
            pergunta = state["messages"][-1].content

            if not contexto:
                contexto = "Nenhum dado encontrado na base específica."

            fontes = []
            for i, f in enumerate(state.get("sources_data", [])):
                fontes.append(FonteDocumento(
                    document_source=f.get("source"),
                    page_number=f.get("page"),
                    chunk_index=i + 1,
                    document_type=f.get("type")
                ))

            try:
                ctx = ConsultaContext(
                    trace_id=state["thread_id"],
                    perfil_cliente=perfil,
                    pergunta_cliente=pergunta,
                    contexto_juridico_bruto=contexto,
                    fontes_detalhadas=fontes,
                    prompt_mestre="Agente Fiscal"
                )
                mcp_json = ctx.model_dump_json()
            except:
                mcp_json = "{}"

            s_prompt = f"""
Você é um consultor tributário sênior.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO:
{contexto}

Diretrizes:
1. Baseie sua resposta no contexto acima.
2. Cite explicitamente artigos se aparecerem no contexto.
3. Seja direto e profissional.
"""

            msgs = [
                SystemMessage(content=s_prompt),
                HumanMessage(content=pergunta)
            ]

            resposta = llm.invoke(msgs)

            return {
                "messages": [AIMessage(content=resposta.content)],
                "mcp_data": mcp_json
            }

        # -----------------------------------------------------------
        # CONSTRÓI O GRAFO
        # -----------------------------------------------------------
        g = StateGraph(AgentState)

        g.add_node("usar_biblioteca", no_busca_biblioteca)
        g.add_node("usar_web", no_busca_web)
        g.add_node("gerar_resposta", no_gerador_resposta)
        g.add_node("gerar_resposta_sem_contexto", no_gerador_resposta)

        g.add_conditional_edges(
            START,
            roteador_de_ferramentas,
            {
                "usar_biblioteca": "usar_biblioteca",
                "usar_web": "usar_web",
                "gerar_resposta_sem_contexto": "gerar_resposta_sem_contexto"
            }
        )

        g.add_edge("usar_biblioteca", "gerar_resposta")
        g.add_edge("usar_web", "gerar_resposta")
        g.add_edge("gerar_resposta", END)
        g.add_edge("gerar_resposta_sem_contexto", END)

        memory = MemorySaver()
        app = g.compile(checkpointer=memory)

        # DB COUNT
        try:
            c = client_qdrant.count(collection_name=NOME_DA_COLECAO, exact=True)
            st.session_state.db_count = c.count
        except:
            st.session_state.db_count = 0

        return app, langfuse

    except Exception as e:
        st.error(f"Erro crítico ao carregar serviços: {e}")
        return None, None


# -----------------------------------------------------------
# CARREGA SERVIÇOS
# -----------------------------------------------------------
agente, langfuse = carregar_servicos_e_grafo()


# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configurações")
    perfil_texto = st.text_area("Perfil do cliente:", value=st.session_state.client_profile, height=200)

    if st.button("Salvar Perfil"):
        st.session_state.client_profile = perfil_texto
        st.success("Perfil atualizado!")

    st.info(f"Documentos na base: {st.session_state.db_count}")


# -----------------------------------------------------------
# INTERFACE DO CHAT
# -----------------------------------------------------------
st.title("🤖 Agente Fiscal v4.4")

for m in st.session_state.messages:
    with st.chat_message(get_streamlit_role(m)):
        st.markdown(m["content"])


# -----------------------------------------------------------
# PROCESSA MENSAGEM DO USUÁRIO
# -----------------------------------------------------------
if prompt := st.chat_input("Digite sua dúvida tributária..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if agente:
        with st.chat_message("assistant"):
            with st.spinner("Consultando legislação..."):

                # Converte dict → BaseMessage
                msgs = []
                for msg in st.session_state.messages:
                    if msg["role"] == "assistant":
                        msgs.append(AIMessage(content=msg["content"]))
                    else:
                        msgs.append(HumanMessage(content=msg["content"]))

                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                # Langfuse
                if langfuse:
                    config["callbacks"] = [langfuse]

                inputs = {
                    "messages": msgs,
                    "perfil_cliente": st.session_state.client_profile,
                    "thread_id": st.session_state.thread_id,
                    "contexto_juridico_bruto": "",
                    "sources_data": [],
                    "mcp_data": ""
                }

                try:
                    final = ""
                    for ev in agente.stream(inputs, config, stream_mode="values"):
                        if "messages" in ev:
                            last = ev["messages"][-1]
                            if last.type == "ai":
                                final = last.content

                    if final:
                        st.markdown(final)
                        st.session_state.messages.append({"role": "assistant", "content": final})
                    else:
                        st.warning("Nenhuma resposta foi gerada.")

                except Exception as e:
                    st.error(f"Erro na execução: {e}")