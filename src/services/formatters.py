import re
import locale

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")


def formatar_cnae(valor: str) -> str:
    if not valor:
        return ""
    digitos = re.sub(r"\D", "", valor)
    if len(digitos) < 7:
        return valor
    return f"{digitos[:4]}-{digitos[4]}/{digitos[5:7]}"


def formatar_moeda(valor: str) -> str:
    if not valor:
        return ""
    digitos = re.sub(r"\D", "", valor)
    if not digitos:
        return ""
    numero = int(digitos)
    return f"R$ {numero / 100:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")