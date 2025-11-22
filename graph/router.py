from langchain_core.messages import BaseMessage, HumanMessage
from rag.rules import identify_fixed_rule
from utils.logs import logger


def classify_question(text: str) -> str:
    """
    Classificador simples baseado em regras jurídicas e heurísticas tributárias.
    Retorna um dos tipos:
        - "qdrant"
        - "web"
        - "rules"
        - "direct"
    """

    text_lower = text.lower()

    # 🔎 1. Perguntas claramente normativas → Qdrant
    keywords_qdrant = [
        "art.", "artigo", "lei", "lc", "lcp", "ec",
        "parágrafo", "caput", "inciso", "alínea",
        "regulamento", "norma", "complementar"
    ]
    if any(k in text_lower for k in keywords_qdrant):
        logger.info("Roteador → QDRANT (palavras normativas detectadas)")
        return "qdrant"

    # 🌐 2. Perguntas sobre atualidade → Web search
    keywords_web = [
        "últimas notícias", "atual", "2024", "2025", "projeto de lei",
        "alteração recente", "andamento", "hoje", "cotação", "valor atual"
    ]
    if any(k in text_lower for k in keywords_web):
        logger.info("Roteador → WEB (tema atual detectado)")
        return "web"

    # 📘 3. Fallback tributário → Regras consolidadas
    if identify_fixed_rule(text):
        logger.info("Roteador → REGRAS FIXAS (tema recorrente detectado)")
        return "rules"

    # 💬 4. LLM direto
    logger.info("Roteador → DIRECT (nenhuma regra específica detectada)")
    return "direct"


def router_node(state: dict) -> str:
    """
    Nó de roteamento do LangGraph.
    Recebe o estado com mensagens e decide qual nó executar.
    """
    messages: list[BaseMessage] = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, HumanMessage):
        # Caso raro: última mensagem não é humana
        logger.info("Roteador recebeu mensagem não-humana; enviando para resposta direta.")
        return "direct_answer"

    question = last_message.content

    decision = classify_question(question)

    # Mapeamento para nós do grafo
    mapping = {
        "qdrant": "rag_qdrant",
        "web": "rag_web",
        "rules": "rag_rules",
        "direct": "direct_answer"
    }

    return mapping.get(decision, "direct_answer")