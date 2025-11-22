from utils.logs import logger

def router_node(state: dict):
    logger.info("Roteador analisando pergunta...")

    if not state or "messages" not in state or not state["messages"]:
        logger.error("State inválido — enviando para direct_answer")
        return "direct_answer"

    question = state["messages"][-1].content.lower()

    # 🔥 DETECÇÃO ABRANGENTE DE ASSUNTOS TRIBUTÁRIOS
    gatilhos_tributarios = [
        "ibs", "cbs", "ec 132", "lc 214",
        "reforma", "tribut", "imposto",
        "crédito", "não cumul", "fiscal",
        "icms", "pis", "cofins"
    ]

    # Se aparecer QUALQUER termo tributário → ir para o RAG Qdrant
    if any(g in question for g in gatilhos_tributarios):
        return "rag_qdrant"

    # Perguntas explícitas sobre leis/atos → RAG
    if "lei" in question or "artigo" in question or "parágrafo" in question:
        return "rag_qdrant"

    # fallback
    logger.info("Roteador → DIRECT (nenhum gatilho encontrado)")
    return "direct_answer"