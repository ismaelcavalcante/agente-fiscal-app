from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.logs import logger


def node_rag_qdrant(state, retriever):
    logger.debug("📌 node_rag_qdrant…")

    question = state["messages"][-1].content
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.retrieve_documents(question, perfil)

        state["contexto_juridico_bruto"] = contexto or ""
        state["sources_data"] = fontes or []

        return state

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro: {e}")
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
- Use SOMENTE o contexto do RAG.
- Se vazio, informe que não encontrou base jurídica.
- Nunca invente leis.
- Relacione com o perfil do cliente.
"""

    resposta = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(
            content=f"PERFIL:\n{perfil}\n\nCONTEXTO RAG:\n{contexto}\n\nFONTES:\n{fontes}"
        )
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state