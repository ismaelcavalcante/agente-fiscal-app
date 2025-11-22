from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.logs import logger


def node_rag_qdrant(state, retriever):
    logger.debug("📌 node_rag_qdrant…")

    question = state["messages"][-1].content
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.retrieve_documents(question, perfil)

        if not isinstance(state, dict):
            logger.error("STATE CORROMPIDO NO RAG – RECRIANDO")
            state = {}

        state["contexto_juridico_bruto"] = contexto if isinstance(contexto, str) else ""
        state["sources_data"] = fontes if isinstance(fontes, list) else []

        return state

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro: {e}")

        if not isinstance(state, dict):
            state = {}

        state["contexto_juridico_bruto"] = ""
        state["sources_data"] = []
        return state


def node_web_search(state, web_tool):
    question = state["messages"][-1].content
    result = web_tool.invoke({"query": question})

    state["contexto_juridico_bruto"] = result.get("answer", "")
    state["sources_data"] = result.get("sources", [])

    return state


def node_direct_answer(state, llm):
    question = state["messages"][-1].content

    resposta = llm.invoke([
        SystemMessage(content="Você é um consultor tributário especialista."),
        HumanMessage(content=question)
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state


def node_generate_final(state, llm):

    question = state["messages"][-1].content
    perfil = state.get("perfil_cliente", {})
    contexto = state.get("contexto_juridico_bruto", "")
    fontes = state.get("sources_data", [])

    prompt = f"""
Você é um consultor tributário sênior.

Regra:
- Use SOMENTE o contexto RAG.
- Se vazio: diga que não encontrou fundamentos no corpus atual.
- Não invente leis.
- Relacione ao perfil do cliente.
"""

    resposta = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"PERFIL: {perfil}\n\nCONTEXTO:\n{contexto}\n\nFONTES:\n{fontes}\n\nPergunta:{question}")
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state


def node_router(state):
    from graph.router import node_router as router
    return router(state)