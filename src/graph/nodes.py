from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.logs import logger


# ===========================
# NODE: Roteador
# ===========================
def node_router(state):
    from graph.router import node_router as router
    return router(state)


# ===========================
# NODE: RAG com QDRANT
# ===========================
def node_rag_qdrant(state, retriever):
    logger.debug("📌 [node_rag_qdrant] Recebendo state...")

    question = state["messages"][-1].content
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.retrieve_documents(question, perfil)

        state["contexto_juridico_bruto"] = contexto
        state["sources_data"] = fontes

        return state

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro ao recuperar docs: {e}")
        state["contexto_juridico_bruto"] = ""
        state["sources_data"] = []
        return state


# ===========================
# NODE: Web Search (Tavily)
# ===========================
def node_web_search(state, web_tool):
    question = state["messages"][-1].content
    result = web_tool.invoke({"query": question})

    state["contexto_juridico_bruto"] = result.get("answer", "")
    state["sources_data"] = result.get("sources", [])

    return state


# ===========================
# NODE: Resposta Direta (sem RAG)
# ===========================
def node_direct_answer(state, llm):
    logger.debug("📌 [node_direct_answer] Recebendo state...")

    question = state["messages"][-1].content

    resposta = llm.invoke([
        SystemMessage(content="Você é um consultor tributário especializado em IBS e CBS."),
        HumanMessage(content=question)
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state


# ===========================
# NODE: Resposta Final
# ===========================
def node_generate_final(state, llm):

    logger.debug("📌 [node_generate_final] Gerando resposta final...")

    question = state["messages"][-1].content
    perfil = state.get("perfil_cliente", "")
    contexto = state.get("contexto_juridico_bruto", "")
    fontes = state.get("sources_data", [])

    prompt = f"""
Você é um consultor tributário sênior.

REGRAS:
- Use EXCLUSIVAMENTE o contexto abaixo se ele existir.
- Se o contexto estiver vazio, diga:
  "Com base no meu corpus atual (Qdrant), não encontrei fundamento jurídico para responder."
- Sempre conecte a resposta ao perfil da empresa.
- Seja objetivo, técnico e juridicamente preciso.
- Cite as fontes retornadas pelo RAG.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO JURÍDICO RECUPERADO (RAG):
{contexto}

FONTES:
{fontes}

Pergunta:
{question}
"""

    final = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ])

    state["messages"].append(AIMessage(content=final.content))
    return state