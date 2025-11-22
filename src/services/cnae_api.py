import requests


def buscar_cnae(query: str):
    digitos = "".join([c for c in query if c.isdigit()])

    if len(digitos) < 2:
        return []

    try:
        url = "https://servicodados.ibge.gov.br/api/v2/cnae/subclasses"
        resp = requests.get(url).json()

        resultados = []
        for item in resp:
            if query.replace("-", "").replace("/", "") in item["id"]:
                resultados.append({
                    "code": item["id"],
                    "title": item["title"]
                })

        return resultados[:5]

    except Exception:
        return []