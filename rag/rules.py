from utils.logs import logger


# Base de regras fixas — você pode expandir conforme necessário.
# Essas regras entram APENAS quando:
# - Qdrant não retorna contexto útil
# - E a Web não é apropriada
# - Ou quando a pergunta requer "regra tributária consolidada"

FIXED_TAX_RULES = {
    "ibs": """
O IBS (Imposto sobre Bens e Serviços) é um tributo criado pela EC 132/2023 e regulamentado pela LC 214/2024.
Ele substituirá ICMS + ISS de forma gradual, com base ampla sobre consumo.
A alíquota padrão estimada é composta por:
- Parcela Federal (CBS)
- Parcela Estadual/Municipal (IBS)
""",

    "cbs": """
A CBS (Contribuição sobre Bens e Serviços) substituirá PIS + COFINS.
Regida principalmente pela LC 214/2024.
A CBS possui base ampla e não cumulativa, com créditos amplos.
""",

    "simples": """
O Simples Nacional é regido pela LC 123/2006.
O regime é opcional para empresas com faturamento até R$ 4,8 milhões.
A tributação é feita por anexos, variando conforme atividade.
""",

    "substituicao": """
A Substituição Tributária (ST) tem regras específicas:
- Aplica-se a mercadorias ou serviços definidos por convênio.
- O recolhimento é antecipado pelo contribuinte substituto.
""",

    "imunidade_livros": """
A Constituição estabelece imunidade de impostos para:
- Livros
- Jornais
- Periódicos
- Papel destinado à impressão
(Regra tradicionalmente mantida após a reforma.)
"""
}


def identify_fixed_rule(query: str) -> str | None:
    """
    Identifica qual regra aplicar baseado em palavras‑chave.
    Retorna a chave da regra ou None se não houver correspondência.
    """
    q = query.lower()

    if any(t in q for t in ["ibs", "imposto sobre bens", "reforma", "lc 214", "ec 132"]):
        return "ibs"

    if any(t in q for t in ["cbs", "contribuição sobre bens", "pis", "cofins"]):
        return "cbs"

    if any(t in q for t in ["simples", "mei", "anexo i", "anexo v"]):
        return "simples"

    if any(t in q for t in ["substituição tributária", "st", "st-ret", "antecipação"]):
        return "substituicao"

    if any(t in q for t in ["livro", "impresso", "periódico", "jornal"]):
        return "imunidade_livros"

    return None


def get_fixed_rule_response(query: str) -> str:
    """
    Retorna uma resposta consolidada com base jurídica fixa.
    Usada como fallback quando nenhum RAG fornece dados suficientes.
    """
    chave = identify_fixed_rule(query)

    if not chave:
        logger.info("Nenhuma regra fixa aplicável encontrada.")
        return ""

    logger.info(f"Regra tributária fixa aplicada: {chave}")

    return FIXED_TAX_RULES[chave]