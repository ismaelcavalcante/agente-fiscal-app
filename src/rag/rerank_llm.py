from utils.logs import logger


class LLMJudgeReranker:
    """
    LLM-Judge definitivo:
    - Usa JSON Mode nativo do modelo
    - Zero parsing manual
    - Zero risco de texto fora do JSON
    - Totalmente determinístico
    """

    def __init__(self, llm):
        """
        O LLM recebido deve ser um modelo que suporte JSON Mode (gpt-4o).
        No app principal, você segue usando gpt-4o-mini.
        Aqui usamos gpt‑4o apenas neste componente.
        """
        self.llm = llm

    def _build_prompt(self, pergunta, docs):
        lista_docs = ""
        for d in docs:
            lista_docs += (
                f'\n{{"doc_id": {d["index"]}, '
                f'"texto": """{d["page_content"]}""" }}'
            )

        return f"""
Avalie a relevância dos documentos abaixo para a pergunta fornecida.

RETORNE APENAS JSON.
NÃO escreva nenhuma frase, explicação, comentário ou texto fora do JSON.

Formato obrigatório:
{{
  "scores": [
    {{"doc_id": 0, "score": 0.00}},
    {{"doc_id": 1, "score": 0.00}}
  ]
}}

Regras:
- "score" deve ser número entre 0.00 e 1.00.
- Nenhum outro campo deve ser incluído.
- JSON deve ser válido.

PERGUNTA:
"{pergunta}"

DOCUMENTOS:
{lista_docs}
"""

    def rerank(self, pergunta, docs, top_k=4):
        if not docs:
            return []

        prompt = self._build_prompt(pergunta, docs)

        try:
            response = self.llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "Você é um avaliador de relevância documental. "
                            "Retorne SOMENTE JSON válido, sem texto adicional."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            parsed = response

        except Exception as e:
            logger.error(
                f"[LLM-JUDGE] Erro durante julgamento. "
                f"Usando reranking vetorial. Detalhes: {e}"
            )
            return docs[:top_k]

        try:
            scores_list = parsed.get("scores", [])
            scores_map = {
                int(item["doc_id"]): float(item["score"])
                for item in scores_list
                if "doc_id" in item and "score" in item
            }
        except Exception as e:
            logger.error(
                f"[LLM-JUDGE] Falha ao interpretar JSON. "
                f"Usando reranking vetorial. Motivo: {e}"
            )
            return docs[:top_k]

        if not scores_map:
            logger.error("[LLM-JUDGE] Nenhum score retornado. Fallback ativado.")
            return docs[:top_k]

        ordered = sorted(
            docs,
            key=lambda d: scores_map.get(d["index"], 0),
            reverse=True,
        )

        return ordered[:top_k]