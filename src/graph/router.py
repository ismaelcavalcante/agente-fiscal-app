from utils.logs import logger


def node_router(state: dict):

    logger.info("Roteador analisando pergunta...")

    # Garantir que sempre existe messages
    if "messages" not in state or not state["messages"]:
        # Sem mensagem → nada a analisar → usar WEB
        state["__route__"] = "WEB"
        return state

    question = state["messages"][-1].content.lower()

    # Palavras-chave que ativam o RAG jurídico/fiscal
    gatilhos_rag = [
        "ibs", "cbs", "ec 132", "lc 214",
        "crédito", "insumo", "benefício fiscal",
        "tribut", "imposto", "não cumulatividade",
        "ncm", "alíquota", "isenção", "substituição tributária",
        "lc", "lei complementar", "art.", "artigo",
        "base de cálculo", "regime", "simples nacional",
        "fato gerador", "jurídico"
    ]

    # Se contém gatilho → RAG
    if any(g in question for g in gatilhos_rag):
        state["__route__"] = "RAG"
        return state

    # Pedidos explícitos de lei/artigo → RAG
    if "lei" in question or "artigo" in question or "norma" in question:
        state["__route__"] = "RAG"
        return state

    # Perguntas de pesquisa geral → WEB
    gatilhos_web = ["pesquise", "pesquisar", "notícia", "busque", "procure"]
    if any(g in question for g in gatilhos_web):
        state["__route__"] = "WEB"
        return state

    # Caso nenhum gatilho seja detectado → usar RAG por padrão
    # (isso garante respostas mais jurídicas/estruturadas)
    state["__route__"] = "RAG"
    return state