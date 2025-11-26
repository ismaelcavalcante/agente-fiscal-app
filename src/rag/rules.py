from utils.logs import logger
import re


FIXED_TAX_RULES = {
    "ibs": """
O IBS (Imposto sobre Bens e Serviços) foi instituído pela EC 132/2023 e regulamentado pela LC 214/2024.
Ele substituirá ICMS + ISS de forma gradual.
A base é ampla, incidindo sobre bens, serviços e direitos.
""",

    "cbs": """
A CBS (Contribuição sobre Bens e Serviços) substituirá PIS + COFINS.
Regida principalmente pela LC 214/2024.
O regime é não cumulativo com créditos amplos.
""",

    "simples": """
O Simples Nacional é disciplinado pela LC 123/2006.
A tributação depende do anexo e faturamento.
Limite de receita bruta: R$ 4,8 milhões.
""",

    "substituicao": """
A Substituição Tributária (ST) transfere a responsabilidade de recolhimento para o substituto tributário.
Regida por convênios e protocolos, com base no ICMS.
""",

    "imunidade_livros": """
A Constituição garante imunidade de impostos sobre:
- Livros
- Jornais
- Periódicos
- Papel destinado à impressão
(Art. 150, VI, d).
"""
}


def match_any(patterns, text):
    return any(re.search(pattern, text) for pattern in patterns)


def identify_fixed_rule(query: str) -> str | None:
    q = query.lower().strip()

    # IBS / Reforma Tributária
    if match_any([r"\bibs\b", r"imposto sobre bens", r"lc 214", r"ec 132", r"reforma tribut[aá]ria"], q):
        return "ibs"

    # CBS / PIS / COFINS
    if match_any([r"\bcbs\b", r"contribui[cç][aã]o sobre bens", r"\bpis\b", r"\bcofins\b"], q):
        return "cbs"

    # ST
    if match_any([r"substitui[cç][aã]o tribut[áa]ria", r"\bst\b", r"st-?ret", r"antecipação"], q):
        return "substituicao"

    # Simples Nacional
    if match_any([r"simples nacional", r"\bmei\b", r"anexo [ivx]+"], q):
        return "simples"

    # Imunidade de livros
    if match_any([r"\blivros?\b", r"per[ií]odic", r"\bjornal\b"], q):
        return "imunidade_livros"

    return None


def get_fixed_rule_response(query: str) -> str:
    chave = identify_fixed_rule(query)

    if not chave:
        logger.info("Nenhuma regra fixa aplicável.")
        return ""

    logger.info(f"Regra fixa utilizada: {chave}")
    return FIXED_TAX_RULES[chave]