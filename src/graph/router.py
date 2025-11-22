from utils.logs import logger


def node_router(state: dict):
    logger.info("Roteador analisando pergunta...")

    if not state or "messages" not in state:
        return "DIRECT"

    question = state["messages"][-1].content.lower()

    GATILHOS_RAG = [
        "ibs", "cbs", "lc 214", "ec 132",
        "reforma tributária", "crédito", "tribut",
        "benefício fiscal", "não cumulatividade",
        "substituição tributária", "imposto",
    ]

    if any(w in question for w in GATILHOS_RAG):
        return "RAG"

    if "lei" in question or "artigo" in question:
        return "RAG"

    if "pesquise" in question or "notícia" in question:
        return "WEB"

    return "DIRECT"