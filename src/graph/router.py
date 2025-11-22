from utils.logs import logger


def node_router(state: dict):
    logger.info("Roteador analisando pergunta...")

    if not state or "messages" not in state:
        return "DIRECT"

    question = state["messages"][-1].content.lower()

    gatilhos_rag = [
        "ibs", "cbs", "lc 214", "ec 132",
        "tribut", "imposto", "crédito",
        "benefício fiscal", "não cumulatividade",
        "regime", "substituição tributária",
        "crédito de ibs", "crédito de cbs",
        "insumo", "custo", "dedução",
    ]

    # se houver qualquer termo tributário → ir para RAG
    if any(g in question for g in gatilhos_rag):
        return "RAG"

    if "lei" in question or "artigo" in question:
        return "RAG"

    if "pesquise" in question or "notícia" in question:
        return "WEB"

    return "DIRECT"