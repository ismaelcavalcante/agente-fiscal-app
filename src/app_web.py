import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient, models
from typing import Union 
from langchain_core.messages import BaseMessage
import os
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from protocol import ConsultaContext, FonteDocumento # Assumindo que 'protocol' está definido

# --- FUNÇÃO UTILITY: TRADUZ O TIPO DA MENSAGEM ---
def get_streamlit_role(message: Union[dict, BaseMessage]) -> str:
    """Converte o objeto LangChain/LangGraph de volta para um role do Streamlit."""
    if isinstance(message, dict):
        # Para mensagens salvas como Dicionário (Formato Streamlit)
        return message["role"].replace('human', 'user') 
    
    # Se for um objeto BaseMessage, usamos o atributo .type
    return message.type.replace('human', 'user') 
# ------------------------------------------------

# --- 1. CONFIGURAÇÃO DA PÁGINA E INICIALIZAÇÃO DE ESTADO ---
st.set_page_config(
    page_title="Agente Fiscal v4.2 (LangGraph + MCP)",
    page_icon="🤖",
    layout="wide"
)

# Inicialização de variáveis de sessão (REFORÇADO)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client_profile" not in st.session_state:
    st.session_state.client_profile = """{
"nome_empresa": "Construtora Alfa Ltda",
"cnae_principal": "4120-4/00 (Construção de Edifícios)",
"regime_tributario": "Simples Nacional",
"faturamento_anual": "R$ 3.000.000,00"
}"""
# A chave "thread_id" é crucial para o erro; garantimos sua inicialização.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1" 
if "db_count" not in st.session_state:
    st.session_state.db_count = 0


# --- Constantes de Serviço ---
NOME_DA_COLECAO = "leis_fiscais_v1"
MODELO_LLM = st.secrets["MODELO_LLM"]
MODELO_EMBEDDING = st.secrets["MODELO_EMBEDDING"]

# --- 2. DEFINIÇÃO DE ESTADO DO LANGGRAPH ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    perfil_cliente: str
    sources_data: List[Dict[str, Any]]
    thread_id: str
    # CORREÇÃO 1: Adicionar campo de estado explícito para o contexto
    contexto_juridico_bruto: str 

# --- 3. CARREGAR OS SERVIÇOS (CACHED) E CONSTRUIR O GRAFO ---

@st.cache_resource
def carregar_servicos_e_grafo():
    try:
        # --- BLOCO DE LEITURA E VALIDAÇÃO DE SECRETS ---
        secrets_dict = {
            "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
            "QDRANT_URL": st.secrets["QDRANT_URL"],
            "QDRANT_API_KEY": st.secrets["QDRANT_API_KEY"],
            "TAVILY_API_KEY": st.secrets["TAVILY_API_KEY"],
            "LANGFUSE_PUBLIC_KEY": st.secrets["LANGFUSE_PUBLIC_KEY"],
            "LANGFUSE_SECRET_KEY": st.secrets["LANGFUSE_SECRET_KEY"],
            "LANGFUSE_BASE_URL": st.secrets["LANGFUSE_BASE_URL"]
        }

        # 1. Carregar Clientes Principais
        llm = ChatOpenAI(api_key=secrets_dict["OPENAI_API_KEY"], model=MODELO_LLM, temperature=0)
        embeddings = OpenAIEmbeddings(api_key=secrets_dict["OPENAI_API_KEY"], model=MODELO_EMBEDDING)
        qdrant_client = QdrantClient(url=secrets_dict["QDRANT_URL"], api_key=secrets_dict["QDRANT_API_KEY"])
        
        # 2. Inicializar Langfuse
        os.environ["TAVILY_API_KEY"] = secrets_dict["TAVILY_API_KEY"]
        langfuse = Langfuse(public_key=secrets_dict["LANGFUSE_PUBLIC_KEY"], secret_key=secrets_dict["LANGFUSE_SECRET_KEY"], host=secrets_dict["LANGFUSE_BASE_URL"])

        # 3. Configuração do Retriever (Qdrant)
        qdrant_store = Qdrant(client=qdrant_client, collection_name=NOME_DA_COLECAO, embeddings=embeddings)
        qdrant_filter = models.Filter(should=[
            models.FieldCondition(key="metadata.regime_tax", match=models.MatchText(text="Simples Nacional")),
            models.FieldCondition(key="metadata.regime_tax", match=models.MatchText(text="Geral/Presumido")),
        ])
        
        # --- Definição da Ferramenta de Biblioteca ---
        retriever_biblioteca_obj = qdrant_store.as_retriever(search_kwargs={"k": 7, "filter": qdrant_filter})
        
        # Transformar o Retriever em uma Tool
        biblioteca_tool = create_retriever_tool(
            retriever_biblioteca_obj,
            "biblioteca_fiscal",
            "Use para buscar e responder perguntas sobre artigos da Emenda Constitucional 132 e a Lei Complementar 214."
        )
        # -------------------------------------------------------------
        
        # 4. Ferramenta de Web Search
        web_search_tool = TavilySearchResults(max_results=3)
        
        # Lista final de ferramentas que o roteador pode usar
        tools = [biblioteca_tool, web_search_tool]


        # --- DEFINIÇÃO DOS NÓS DO LANGGRAPH (Lógica de Decisão) ---
        def roteador_de_ferramentas(state: AgentState) -> str:
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_router_llm = llm.bind_tools(tools)
            
            content = last_message.content
            
            # O roteador considera o perfil do cliente
            response = tool_router_llm.invoke(f"Perfil do Cliente: {state['perfil_cliente']}\n\nPergunta: {content}")
            
            if not response.tool_calls:
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
            query = f"Perfil: {perfil}\nPergunta: {pergunta}"
            
            # Correção para o erro 'NoneType' object is not iterable
            try:
                docs = biblioteca_tool.invoke(query) 
                if docs is None:
                    docs = []
            except Exception as e:
                print(f"Erro ao invocar biblioteca_tool: {e}")
                docs = []
            
            contexto_text = "\n---\n".join([doc.page_content for doc in docs])
            
            metadados = [{"source": doc.metadata.get('document_type', 'Lei'), "page": doc.metadata.get('page'), "type": doc.metadata.get('document_type')} for doc in docs]
            
            # CORREÇÃO 6: Retornar 'messages' (aqui vazia, LangGraph cuidará da concatenação)
            return {"messages": [], "sources_data": metadados, "contexto_juridico_bruto": contexto_text}

        def no_busca_web(state: AgentState):
            pergunta = state["messages"][-1].content
            
            # Correção para o erro 'NoneType' object is not iterable
            try:
                docs = web_search_tool.invoke(pergunta)
                if docs is None:
                    docs = []
            except Exception as e:
                print(f"Erro ao invocar web_search_tool: {e}")
                docs = []
                
            contexto_text = "\n---\n".join([str(doc) for doc in docs])
            metadados = [{"source": "Tavily Web Search", "page": None, "type": "WEB", "content": doc.get('content')} for doc in docs]
            
            # CORREÇÃO 7: Retornar 'messages' (aqui vazia, LangGraph cuidará da concatenação)
            return {"messages": [], "sources_data": metadados, "contexto_juridico_bruto": contexto_text}

        def no_gerador_resposta(state: AgentState):
            messages = state["messages"]
            perfil = state["perfil_cliente"]
            
            # CORREÇÃO 4: Lê o contexto bruto diretamente do estado, ignorando a mensagem intermediária
            contexto_juridico_bruto = state.get("contexto_juridico_bruto", "Nenhuma fonte relevante encontrada.")
            
            # Pega a última HumanMessage original do usuário
            pergunta_cliente_msg = messages[-1].content 

            # Formatar Fontes Detalhadas (para o MCP)
            fontes_detalhadas = []
            for i, fonte in enumerate(state.get("sources_data", [])):
                try:
                    page_num = fonte.get("page")
                    if isinstance(page_num, str) and page_num.isdigit():
                        page_num = int(page_num)
                    elif page_num is not None and not isinstance(page_num, int):
                        page_num = None
                        
                    fontes_detalhadas.append(FonteDocumento(
                        document_source=fonte.get("source", "N/A"),
                        page_number=page_num,
                        chunk_index=i + 1,
                        document_type=fonte.get("type", "DESCONHECIDO")
                    ))
                except Exception as e:
                    print(f"Erro ao formatar FonteDocumento: {e}") 
                    pass 

            # CONSTRUIR E VALIDAR O PROTOCOLO (MCP)
            try:
                # CORREÇÃO: Lê o thread_id DIRETAMENTE do state (AgentState), não do st.session_state
                current_thread_id = state["thread_id"]
                
                context_protocol = ConsultaContext(
                    # Usa o thread_id que veio no state
                    trace_id=current_thread_id, 
                    perfil_cliente=perfil, 
                    pergunta_cliente=pergunta_cliente_msg,
                    contexto_juridico_bruto=contexto_juridico_bruto, 
                    fontes_detalhadas=fontes_detalhadas,
                    prompt_mestre="O Agente Fiscal Advisor, especialista em reforma tributária."
                )
            except Exception as e:
                # Garante que o erro de validação do MCP é repassado
                raise ValueError(f"Falha na validação do MCP (ContextProtocolModel): {e}")

            # Gerar Resposta (Usando o Protocolo Validado)
            # CORREÇÃO 5: Injetar o Perfil do Cliente diretamente no System Prompt para que o modelo possa responder 
            # perguntas sobre o perfil, mesmo sem contexto RAG.
            prompt_mestre_msg = HumanMessage(
                content=f"""
                {context_protocol.prompt_mestre}
                Você está respondendo à pergunta de um cliente. Use as informações de perfil e contexto fornecidas para formular sua resposta.

                **Perfil do Cliente:** {perfil}
                **Contexto Jurídico Validado (RAG):** {contexto_juridico_bruto}

                Com base nas informações acima, responda à última pergunta do usuário. Se o Contexto Jurídico Validado for "Nenhuma fonte relevante encontrada.", responda apenas com base no seu conhecimento geral ou no Perfil do Cliente.
                """
            )
            
            ultima_mensagem_usuario = HumanMessage(content=pergunta_cliente_msg)
            
            response = llm.invoke([ultima_mensagem_usuario, prompt_mestre_msg])
            
            # CORREÇÃO 8: O nó de geração deve retornar a mensagem final.
            return {"messages": [AIMessage(content=response.content)], "mcp_data": context_protocol.model_dump_json()}

        # --- COMPILAÇÃO DO GRAFO (O MAESTRO) ---
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

        # Verifica a contagem de fatias no DB
        try:
            count = qdrant_client.count(collection_name=NOME_DA_COLECAO, exact=True)
            st.session_state.db_count = count.count
        except Exception:
            st.session_state.db_count = 0

        return app_graph, langfuse 

    except KeyError as e:
        st.error(f"ERRO FATAL: Secret {e} não encontrada. Verifique as 6 chaves no painel do Streamlit.")
        return None, None
    except Exception as e:
        st.error(f"ERRO DE CONEXÃO: O agente não pôde ser carregado. Detalhe: {e}")
        return None, None

# --- 4. CARREGAR OS SERVIÇOS NA INICIALIZAÇÃO ---
agente, langfuse = carregar_servicos_e_grafo()

# --- 5. INTERFACE DA BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/128/10573/10573788.png", width=80)
    st.title("Perfil do Cliente")
    st.markdown("O Agente usará este perfil para todas as consultas.")
    
    perfil_texto = st.text_area(
        "Edite o JSON do Perfil:",
        value=st.session_state.client_profile,
        height=250
    )
    
    if st.button("Salvar Perfil"):
        st.session_state.client_profile = perfil_texto
        st.success("Perfil salvo para esta sessão!")
    
    st.markdown("---")
    st.subheader("Base de Conhecimento")
    st.markdown(f"**Fatias na Biblioteca:** `{st.session_state.get('db_count', 0)}`")
    st.markdown("EC 132 e LC 214")
    st.markdown("**Ferramenta de Web:** `Tavily Search`")
    st.markdown("**Monitoramento:** `Langfuse Ativo`")


# --- 6. INTERFACE PRINCIPAL DO CHAT ---
st.title("🤖 Agente Fiscal v4.2 (LangGraph + MCP)")
st.markdown("Pergunte sobre as leis (EC 132/LC 214) ou sobre notícias do congresso.")

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(get_streamlit_role(message)): 
        content = message.content if hasattr(message, 'content') else message['content']
        st.markdown(content)

# Recebe a nova pergunta do usuário
if prompt := st.chat_input("O que o congresso decidiu hoje sobre o cashback?"):
    
    # 1. Adiciona a mensagem do usuário ao histórico (Formato seguro de DICT)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if agente:
        with st.chat_message("assistant"):
            with st.spinner("O Agente está pensando..."):
                
                # --- CRIAÇÃO DA LISTA DE MENSAGENS LIMPA (Conversão para BaseMessage) ---
                mensagens_para_grafo = []
                for msg in st.session_state.messages:
                    content = getattr(msg, 'content', msg.get('content', ''))
                    role = getattr(msg, 'role', msg.get('role', 'user')) 
                    
                    if role in ['assistant', 'ai']:
                        mensagens_para_grafo.append(AIMessage(content=content))
                    else: 
                        mensagens_para_grafo.append(HumanMessage(content=content))
                
                # -------------------------------------------------------------
                
                # Prepara os Callbacks do Langfuse (Mantido)
                langfuse_callbacks = [] 
                if langfuse:
                    try:
                        # O Langfuse também precisa do thread_id, acessamos com .get() por segurança
                        safe_thread_id = st.session_state.get("thread_id", "fallback_id_00")
                        langfuse_callbacks = [langfuse.get_langchain_callback(
                            user_id="usuario_streamlit",
                            session_id=safe_thread_id
                        )]
                    except AttributeError:
                        pass
                
                config = {
                        # O LangGraph pode gerenciar a thread_id via checkpointer, mas mantemos aqui para compatibilidade
                        "configurable": {"thread_id": st.session_state.get("thread_id", "fallback_id_00")},
                        "callbacks": langfuse_callbacks
                    }
                inputs = {
                        "messages": mensagens_para_grafo, 
                        "perfil_cliente": st.session_state.client_profile,
                        "thread_id": st.session_state.thread_id,
                        # Inicializa o novo campo de estado, será populado pelo nó de busca se acionado.
                        "contexto_juridico_bruto": "" 
                    }
                
                try:
                    resposta_final = ""
                    mcp_output = None
                    
                    # Loop de streaming do LangGraph
                    for event in agente.stream(inputs, config, stream_mode="values"):
                        
                        if not event or "messages" not in event:
                            continue
                            
                        # LangGraph concatena o histórico aqui.
                        new_message = event["messages"][-1]
                        
                        # Correção do .role para .type
                        if new_message and new_message.type == "ai":
                            resposta_final = new_message.content
                        
                        if "mcp_data" in event:
                             mcp_output = event["mcp_data"]

                    st.markdown(resposta_final)
                    
                    # 2. Salva a resposta no histórico no formato seguro de DICT
                    st.session_state.messages.append({"role": "assistant", "content": resposta_final})

                except Exception as e:
                    # Se ocorrer um erro no LangGraph, exibe a mensagem de erro
                    st.error(f"Erro ao executar o agente: {e}")
                    print(f"Erro na execução do grafo: {e}")
    else:
        st.error("O Agente não pôde ser carregado. Verifique as 'Secrets' e recarregue a página.")
