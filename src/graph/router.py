from utils.logs import logger

def node_router(state: dict):

    logger.info("Roteador analisando pergunta...")

    msg = state.get("messages", [])
    if not msg:
        return "DIRECT"

    question = msg[-1].content.lower()

    gatilhos_rag = [
        "ibs", "cbs", "ec 132", "lc 214",
        "crédito", "insumo", "tribut", "imposto",
        "benefício fiscal", "não cumulatividade",
    ]

    if any(g in question for g in gatilhos_rag):
        return "RAG"

    if "lei" in question or "artigo" in question:
        return "RAG"

    if "pesquise" in question or "notícia" in question:
        return "WEB"

    return "DIRECT"