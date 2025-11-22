import re


def formatar_cnae(valor: str) -> str:
    """Aplica máscara automática ao CNAE."""
    if not valor:
        return ""
    digitos = re.sub(r"\D", "", valor)

    # CNAE completo possui 7 dígitos
    if len(digitos) < 7:
        return valor

    return f"{digitos[:4]}-{digitos[4]}/{digitos[5:7]}"


def formatar_moeda(valor: str) -> str:
    """Formata automaticamente valores como R$ 1.234.567,89 sem usar locale."""
    if not valor:
        return ""

    # Mantém somente números
    digitos = re.sub(r"\D", "", valor)

    if not digitos:
        return ""

    # Garante pelo menos 2 dígitos (centavos)
    if len(digitos) == 1:
        digitos = "0" + digitos
    if len(digitos) == 2:
        inteiro = "0"
        centavos = digitos
    else:
        inteiro = digitos[:-2]
        centavos = digitos[-2:]

    # Adiciona separadores
    inteiro_formatado = ""
    for i, c in enumerate(reversed(inteiro)):
        if i != 0 and i % 3 == 0:
            inteiro_formatado = "." + inteiro_formatado
        inteiro_formatado = c + inteiro_formatado

    return f"R$ {inteiro_formatado},{centavos}"