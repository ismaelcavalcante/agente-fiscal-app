from utils.logs import logger


def router_node(state: dict):
    logger.info("Roteador analisando pergunta...")

    if not state or "messages" not in state or not state["messages"]:
        return "direct_answer"

    question = state["messages"][-1].content.lower()

    gatilhos_tributarios = [
        "ibs", "cbs", "icms", "pis", "cofins",
        "ec 132", "lc 214", "alíquota", "imposto",
        "tribut", "fiscal", "não cumul", "crédito",
        "benefício", "isenção", "substituição tributária",
        "crédito de insumo", "crédito financeiro"
    ]

    if any(t in question for t in gatilhos_tributarios):
        logger.info("Roteador → RAG_QDRANT")
        return "rag_qdrant"

    logger.info("Roteador → DIRECT")
    return "direct_answer"