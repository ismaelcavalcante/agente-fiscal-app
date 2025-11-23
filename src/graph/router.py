from utils.logs import logger


def node_router(state: dict):

    logger.info("Roteador analisando pergunta...")

    question = state.get("ultima_pergunta", "").lower()

    if not question.strip():  # <-- pergunta vazia
        state["__route__"] = "WEB"
        return state

    gatilhos_rag = [
        "ibs", "cbs", "ec 132", "lc 214",
        "crédito", "insumo", "benefício fiscal",
        "tribut", "imposto", "não cumulatividade",
        "ncm", "alíquota", "isenção", "substituição tributária",
        "lc", "lei complementar", "art", "art.",
        "base de cálculo", "regime", "simples nacional",
        "fato gerador", "jurídico"
    ]

    if any(g in question for g in gatilhos_rag):
        state["__route__"] = "RAG"
        return state

    gatilhos_web = ["pesquise", "pesquisar", "notícia", "busque", "procure"]
    if any(w in question for w in gatilhos_web):
        state["__route__"] = "WEB"
        return state

    state["__route__"] = "RAG"
    return state