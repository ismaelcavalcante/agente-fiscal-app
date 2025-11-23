from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.logs import logger


def node_rag_qdrant(state, retriever):
    logger.debug("📌 node_rag_qdrant…")

    question = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.retrieve_documents(question, perfil)
        state["contexto_juridico_bruto"] = contexto or ""
        state["sources_data"] = fontes or []
        state["rag_ok"] = bool(contexto.strip())

        return state

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro: {e}")
        state["contexto_juridico_bruto"] = ""
        state["sources_data"] = []
        state["rag_ok"] = False
        return state


def node_web_search(state, web_tool):
    logger.debug("📌 node_web_search…")

    if state.get("rag_ok"):
        return state

    question = state.get("ultima_pergunta", "")
    result = web_tool.invoke({"query": question})

    state["contexto_juridico_bruto"] = result.get("answer", "")
    state["sources_data"] = result.get("sources", [])
    state["rag_ok"] = bool(state["contexto_juridico_bruto"].strip())

    return state


def node_generate_final(state, llm):
    logger.debug("📌 node_generate_final…")

    question = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", {})
    contexto = state.get("contexto_juridico_bruto", "")
    fontes = state.get("sources_data", [])

    system_prompt = """
Você é um consultor tributário sênior.

Regras:
- Use exclusivamente o contexto fornecido abaixo.
- Se o contexto estiver vazio, diga que não encontrou base jurídica.
- Não invente leis.
- Relacione a resposta ao perfil do cliente.
"""

    resposta = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"PERGUNTA:\n{question}\n\nPERFIL:\n{perfil}\n\nCONTEXTO RAG:\n{contexto}\n\nFONTES:\n{fontes}")
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state