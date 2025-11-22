import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient, models
from typing import Union, TypedDict, Annotated, List, Dict, Any
import os

# Imports do LangChain e LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langfuse import Langfuse

# Tenta importar o protocolo, se não existir, define classes dummy para não quebrar
try:
    from protocol import ConsultaContext, FonteDocumento
except ImportError:
    # Classes Dummy caso o arquivo protocol.py não exista no ambiente
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

# --- FUNÇÃO UTILITY: TRADUZ O TIPO DA MENSAGEM ---
def get_streamlit_role(message: Union[dict, BaseMessage]) -> str:
    """Converte o objeto LangChain/LangGraph de volta para um role do Streamlit."""
    if isinstance(message, dict):
        return message["role"].replace('human', 'user')
    return message.type.replace('human', 'user')

# --- 1. CONFIGURAÇÃO DA PÁGINA E INICIALIZAÇÃO DE ESTADO ---
st.set_page_config(
    page_title="Agente Fiscal v4.3 (Fixed)",
    page_icon="⚖️",
    layout="wide"
)

# Inicialização de variáveis de sessão
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

# --- Constantes de Serviço ---
NOME_DA_COLECAO = "leis_fiscais_v1"
# Tenta pegar dos secrets, se não, usa placeholders para evitar crash imediato
MODELO_LLM = st.secrets.get("MODELO_LLM", "gpt-4o")
MODELO_EMBEDDING = st.secrets.get("MODELO_EMBEDDING", "text-embedding-3-large")

# --- 2. DEFINIÇÃO DE ESTADO DO LANGGRAPH ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    perfil_cliente: str
    sources_data: List[Dict[str, Any]]
    thread_id: str
    contexto_juridico_bruto: str
    mcp_data: str  # Campo adicionado para evitar erros de chave

# --- 3. CARREGAR OS SERVIÇOS (CACHED) E CONSTRUIR O GRAFO ---
@st.cache_resource
def carregar_servicos_e_grafo():
    try:
        # Validação básica de secrets
        required_secrets = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY"]
        for secret in required_secrets:
            if secret not in st.secrets:
                st.error(f"Secret faltando: {secret}")
                return None, None

        secrets_dict = {
            "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
            "QDRANT_URL": st.secrets["QDRANT_URL"],
            "QDRANT_API_KEY": st.secrets["QDRANT_API_KEY"],
            "TAVILY_API_KEY": st.secrets["TAVILY_API_KEY"],
            "LANGFUSE_PUBLIC_KEY": st.secrets.get("LANGFUSE_PUBLIC_KEY", ""),
            "LANGFUSE_SECRET_KEY": st.secrets.get("LANGFUSE_SECRET_KEY", ""),
            "LANGFUSE_BASE_URL": st.secrets.get("LANGFUSE_BASE_URL", "")
        }

        # 1. Inicializar Clientes
        llm = ChatOpenAI(api_key=secrets_dict["OPENAI_API_KEY"], model=MODELO_LLM, temperature=0)
        embeddings = OpenAIEmbeddings(api_key=secrets_dict["OPENAI_API_KEY"], model=MODELO_EMBEDDING)
        qdrant_client = QdrantClient(url=secrets_dict["QDRANT_URL"], api_key=secrets_dict["QDRANT_API_KEY"])
        
        os.environ["TAVILY_API_KEY"] = secrets_dict["TAVILY_API_KEY"]
        
        langfuse = None
        if secrets_dict["LANGFUSE_PUBLIC_KEY"]:
            langfuse = Langfuse(
                public_key=secrets_dict["LANGFUSE_PUBLIC_KEY"], 
                secret_key=secrets_dict["LANGFUSE_SECRET_KEY"], 
                host=secrets_dict["LANGFUSE_BASE_URL"]
            )

        # 2. Configuração do Retriever (Qdrant)
        qdrant_store = Qdrant(client=qdrant_client, collection_name=NOME_DA_COLECAO, embeddings=embeddings)
        
        # --- CORREÇÃO RAG: REMOVIDO FILTRO RESTRITIVO ---
        # O filtro metadata.regime_tax estava bloqueando resultados se os metadados não fossem exatos.
        # Usamos k=5 para trazer mais contexto.
        retriever_biblioteca_obj = qdrant_store.as_retriever(search_kwargs={"k": 5})
        
        biblioteca_tool = create_retriever_tool(
            retriever_biblioteca_obj,
            "biblioteca_fiscal",
            "OBRIGATÓRIO: Use esta ferramenta para buscar leis, artigos, impostos (IBS, CBS), EC 132 e LC 214."
        )
        
        web_search_tool = TavilySearchResults(max_results=3)
        tools = [biblioteca_tool, web_search_tool]

        # --- DEFINIÇÃO DOS NÓS ---

        def roteador_de_ferramentas(state: AgentState) -> str:
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_router_llm = llm.bind_tools(tools)
            
            # Prompt de sistema para forçar decisão correta
            system_msg = SystemMessage(content="""Você é um assistente jurídico tributário.
            Se a pergunta for sobre leis, impostos, reforma tributária ou regras fiscais, você DEVE chamar a ferramenta 'biblioteca_fiscal'.
            Se for sobre notícias atuais ou cotações, use 'tavily_search_results'.
            Caso contrário, responda diretamente.""")
            
            response = tool_router_llm.invoke([system_msg, last_message])
            
            if not response.tool_calls:
                # Fallback: Se contiver palavras-chave fiscais, força a busca
                content_lower = last_message.content.lower()
                keywords = ['lei', 'imposto', 'tribut', 'alíquota', 'ibs', 'cbs', 'artigo', 'ec 132', 'lc 214']
                if any(k in content_lower for k in keywords):
                    return "usar_biblioteca_fiscal"
                return "gerar_resposta_sem_contexto"
            
            tool_name = response.tool_calls[0]["name"]
            if tool_name == "tavily_search_results": 
                return "usar_busca_web"
            elif tool_name == "biblioteca_fiscal":
                return "usar_biblioteca_fiscal"
            else:
                return "gerar_resposta_sem_contexto"

        def no_busca_biblioteca(state: AgentState):
            pergunta = state["messages"][-1].content
            perfil = state["perfil_cliente"]
            query = f"{pergunta} (Contexto: {perfil})"
            
            try:
                # Invoca o retriever diretamente
                docs = retriever_biblioteca_obj.invoke(query)
                if not docs:
                    docs = []
            except Exception as e:
                print(f"Erro Qdrant: {e}")
                docs = []
            
            contexto_text = "\n\n".join([f"Fonte: {d.metadata.get('source', 'Lei')} | Conteúdo: {d.page_content}" for d in docs])
            
            metadados = [{"source": doc.metadata.get('document_type', 'Lei'), "page": doc.metadata.get('page'), "type": doc.metadata.get('document_type')} for doc in docs]
            
            # CORREÇÃO: Não retorna "messages" para evitar duplicação
            return {"sources_data": metadados, "contexto_juridico_bruto": contexto_text}

        def no_busca_web(state: AgentState):
            pergunta = state["messages"][-1].content
            try:
                docs = web_search_tool.invoke(pergunta)
                if docs is None: docs = []
            except:
                docs = []
                
            contexto_text = "\n---\n".join([doc.get('content', '') for doc in docs])
            metadados = [{"source": "Web", "page": None, "type": "WEB", "content": doc.get('content')} for doc in docs]
            
            # CORREÇÃO: Não retorna "messages"
            return {"sources_data": metadados, "contexto_juridico_bruto": contexto_text}

        def no_gerador_resposta(state: AgentState):
            perfil = state["perfil_cliente"]
            contexto = state.get("contexto_juridico_bruto", "")
            pergunta_usuario = state["messages"][-1].content
            
            if not contexto:
                contexto = "Nenhuma legislação específica encontrada na busca automática."

            # Prepara dados para MCP (se necessário)
            fontes_detalhadas = []
            for i, fonte in enumerate(state.get("sources_data", [])):
                fontes_detalhadas.append(FonteDocumento(
                    document_source=str(fonte.get("source", "N/A")),
                    page_number=fonte.get("page"),
                    chunk_index=i + 1,
                    document_type=str(fonte.get("type", "DESCONHECIDO"))
                ))

            try:
                context_protocol = ConsultaContext(
                    trace_id=state["thread_id"],
                    perfil_cliente=perfil,
                    pergunta_cliente=pergunta_usuario,
                    contexto_juridico_bruto=contexto,
                    fontes_detalhadas=fontes_detalhadas,
                    prompt_mestre="Agente Fiscal Advisor"
                )
                mcp_json = context_protocol.model_dump_json()
            except Exception:
                mcp_json = "{}"

            # CORREÇÃO: Prompt com SystemMessage para priorizar o contexto
            prompt_sistema = f"""Você é um consultor tributário sênior.
            
            PERFIL DO CLIENTE:
            {perfil}
            
            CONTEXTO LEGISLATIVO/INFORMATIVO (Use estas informações para responder):
            {contexto}
            
            DIRETRIZES:
            1. Responda à pergunta do usuário baseando-se PRIMORDIALMENTE no Contexto acima.
            2. Se o contexto citar artigos ou leis, mencione-os explicitamente.
            3. Se o contexto for vazio ou irrelevante, use seu conhecimento mas avise que não encontrou na base específica.
            4. Seja direto e profissional.
            """
            
            mensagens_llm = [
                SystemMessage(content=prompt_sistema),
                HumanMessage(content=pergunta_usuario)
            ]
            
            response = llm.invoke(mensagens_llm)
            
            return {"messages": [AIMessage(content=response.content)], "mcp_data": mcp_json}

        # --- COMPILAÇÃO DO GRAFO ---
        workflow = StateGraph(AgentState)
        workflow.add_node("usar_biblioteca_fiscal", no_busca_biblioteca)
        workflow.add_node("usar_busca_web", no_busca_web)
        workflow.add_node("gerar_resposta", no_gerador_resposta)
        workflow.add_node("gerar_resposta_sem_contexto", no_gerador_resposta)
        
        workflow.add_conditional_edges(
            START, roteador_de_ferramentas,
            {"usar_biblioteca_fiscal": "usar_biblioteca_fiscal", "usar_busca_web": "usar_busca_web", "gerar_resposta_sem_contexto": "gerar_resposta_sem_contexto"}
        )
        workflow.add_edge("usar_biblioteca_fiscal", "gerar_resposta")
        workflow.add_edge("usar_busca_web", "gerar_resposta")
        workflow.add_edge("gerar_resposta", END)
        workflow.add_edge("gerar_resposta_sem_contexto", END)
        
        memory = MemorySaver()
        app_graph = workflow.compile(checkpointer=memory)

        # Verifica contagem DB
        try:
            count = qdrant_client.count(collection_name=NOME_DA_COLECAO, exact=True)
            st.session_state.db_count = count.count
        except:
            st.session_state.db_count = 0

        return app_graph, langfuse

    except Exception as e:
        st.error(f"ERRO CRÍTICO AO CARREGAR AGENTE: {e}")
        return None, None

# --- 4. CARREGAR OS SERVIÇOS ---
agente, langfuse = carregar_servicos_e_grafo()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configurações")
    st.subheader("Perfil do Cliente")
    perfil_texto = st.text_area("JSON do Perfil:", value=st.session_state.client_profile, height=200)
    if st.button("Atualizar Perfil"):
        st.session_state.client_profile = perfil_texto
        st.success("Atualizado!")
    
    st.divider()
    st.info(f"📚 Documentos na Base: {st.session_state.get('db_count', 0)}")

# --- 6. CHAT PRINCIPAL ---
st.title("🤖 Agente Fiscal v4.3")

for message in st.session_state.messages:
    with st.chat_message(get_streamlit_role(message)):
        content = message.content if hasattr(message, 'content') else message['content']
        st.markdown(content)

if prompt := st.chat_input("Digite sua dúvida tributária..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if agente:
        with st.chat_message("assistant"):
            with st.spinner("Consultando legislação e analisando..."):
                
                # Prepara mensagens para o grafo (converte dict -> BaseMessage)
                mensagens_para_grafo = []
                for msg in st.session_state.messages:
                    content = msg['content'] if isinstance(msg, dict) else msg.content
                    role = msg['role'] if isinstance(msg, dict) else msg.type
                    if role in ['assistant', 'ai']:
                        mensagens_para_grafo.append(AIMessage(content=content))
                    else:
                        mensagens_para_grafo.append(HumanMessage(content=content))
                
                # Configuração de execução
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                if langfuse:
                    callbacks = [langfuse.get_langchain_callback(user_id="user", session_id=st.session_state.thread_id)]
                    config["callbacks"] = callbacks

                # INPUTS CORRIGIDOS: sources_data inicializado vazio
                inputs = {
                    "messages": mensagens_para_grafo,
                    "perfil_cliente": st.session_state.client_profile,
                    "thread_id": st.session_state.thread_id,
                    "contexto_juridico_bruto": "",
                    "sources_data": [], # CORREÇÃO CRÍTICA
                    "mcp_data": ""
                }
                
                try:
                    resposta_final = ""
                    for event in agente.stream(inputs, config, stream_mode="values"):
                        if "messages" in event:
                            msg = event["messages"][-1]
                            if msg.type == "ai":
                                resposta_final = msg.content
                    
                    if resposta_final:
                        st.markdown(resposta_final)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_final})
                    else:
                        st.warning("O agente não gerou uma resposta final.")
                        
                except Exception as e:
                    st.error(f"Erro na execução: {e}")