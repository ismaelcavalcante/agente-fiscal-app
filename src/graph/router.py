from utils.logs import logger

def node_router(state: dict):

    logger.info("Roteador analisando pergunta...")

    messages = state.get("messages", [])
    if not messages:
        return {"__route__": "DIRECT"}

    question = messages[-1].content.lower()

    gatilhos_rag = [
        "ibs", "cbs", "ec 132", "lc 214",
        "crédito", "insumo", "benefício fiscal",
        "tribut", "imposto", "não cumulatividade"
    ]

    if any(g in question for g in gatilhos_rag):
        return {"__route__": "RAG"}

    if "lei" in question or "artigo" in question:
        return {"__route__": "RAG"}

    if "pesquise" in question or "notícia" in question:
        return {"__route__": "WEB"}

    return {"__route__": "DIRECT"}