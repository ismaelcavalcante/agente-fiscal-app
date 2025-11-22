# graph/router.py

from utils.logs import logger


def router_node(state: dict) -> str:
    """
    Roteador oficial:
    Decide qual nó executar com base no conteúdo da pergunta.
    Retorna APENAS o nome do nó do grafo.
    """

    # Proteção contra estado vazio
    if not state or "messages" not in state or len(state["messages"]) == 0:
        logger.warning("Roteador recebeu state vazio → fallback para DIRECT")
        return "direct_answer"

    question = state["messages"][-1].content.lower()
    logger.info("Roteador analisando pergunta...")

    # Caminhos específicos
    if "ec 132" in question or "lc 214" in question or "emenda constitucional" in question:
        logger.info("Roteador → RAG_RULES")
        return "rag_rules"

    if "últimas notícias" in question or "novidades" in question or "site" in question:
        logger.info("Roteador → RAG_WEB")
        return "rag_web"

    if "artigo" in question or "lei" in question or "tribut" in question:
        logger.info("Roteador → RAG_QDRANT")
        return "rag_qdrant"

    # Default
    logger.info("Roteador → DIRECT (nenhuma regra específica detectada)")
    return "direct_answer"