from utils.logs import logger

def node_router(state: dict):

    logger.info("Roteador analisando pergunta...")

    # Garantir que sempre existe messages
    if "messages" not in state or not state["messages"]:
        state["__route__"] = "DIRECT"
        return state

    question = state["messages"][-1].content.lower()

    gatilhos_rag = [
        "ibs", "cbs", "ec 132", "lc 214",
        "crédito", "insumo", "benefício fiscal",
        "tribut", "imposto", "não cumulatividade"
    ]

    if any(g in question for g in gatilhos_rag):
        state["__route__"] = "RAG"
        return state

    if "lei" in question or "artigo" in question:
        state["__route__"] = "RAG"
        return state

    if "pesquise" in question or "notícia" in question:
        state["__route__"] = "WEB"
        return state

    state["__route__"] = "DIRECT"
    return state