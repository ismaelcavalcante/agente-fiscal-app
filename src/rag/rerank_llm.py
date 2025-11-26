# rag/judge_reranker.py

import json
from utils.logs import logger


class LLMJudgeReranker:
    def __init__(self, llm):
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
NÃO escreva texto fora do JSON.

Formato obrigatório:
{{
  "scores": [
    {{"doc_id": 0, "score": 0.00}},
    {{"doc_id": 1, "score": 0.00}}
  ]
}}

Regras:
- score entre 0.00 e 1.00
- JSON válido
- Sem campos extras

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

            parsed = json.loads(response.content)

        except Exception as e:
            logger.error(
                f"[LLM-JUDGE] Erro ao interpretar JSON. "
                f"Usando reranking vetorial. Motivo: {e}"
            )
            return docs[:top_k]

        scores_list = parsed.get("scores", [])
        if not scores_list:
            logger.error("[LLM-JUDGE] Nenhum score retornado.")
            return docs[:top_k]

        try:
            scores_map = {
                int(item["doc_id"]): float(item["score"])
                for item in scores_list
            }
        except Exception as e:
            logger.error(f"[LLM-JUDGE] Score inválido. Fallback. {e}")
            return docs[:top_k]

        ordered = sorted(
            docs,
            key=lambda d: scores_map.get(d["index"], 0),
            reverse=True,
        )

        return ordered[:top_k]