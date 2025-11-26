# graph/router.py

import re
from utils.logs import logger


def _match(patterns, text):
    return any(re.search(p, text) for p in patterns)


def node_router(state: dict):
    """
    Roteador jurídico inteligente.
    Decide entre RAG e WEB com base em padrões jurídicos e comandos do usuário.
    """

    pergunta = (state.get("ultima_pergunta") or "").lower().strip()

    if not pergunta:
        state["__route__"] = "WEB"
        return state

    # ---------------------------
    # Gatilhos jurídicos (RAG)
    # ---------------------------
    padroes_rag = [
        r"\bibs\b", r"\bcbs\b",
        r"ec\s?132", r"lc\s?214",
        r"\bpis\b", r"\bcofins\b",
        r"não\s+cumulatividade",
        r"\bal[ií]quota\b",
        r"\bimposto\b",
        r"\bicms\b", r"\biss\b",
        r"\bncm\b",
        r"\bsubstitui[cç][aã]o tribut[áa]ria\b",
        r"\bart(\.|igo)?\b"
    ]

    if _match(padroes_rag, pergunta):
        logger.info("🔀 Roteador: caminho → RAG")
        state["__route__"] = "RAG"
        return state

    # ---------------------------
    # Gatilhos explícitos de busca (WEB)
    # ---------------------------
    padroes_web = [
        r"pesquis", r"busque", r"procure", r"not[íi]cia"
    ]

    if _match(padroes_web, pergunta):
        logger.info("🔀 Roteador: caminho → WEB")
        state["__route__"] = "WEB"
        return state

    # Default → Jurídico
    logger.info("🔀 Roteador: caminho padrão → RAG")
    state["__route__"] = "RAG"
    return state