from langchain_core.messages import SystemMessage, HumanMessage
from utils.logs import logger


class LLMJudgeReranker:

    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, docs: list, top_k=4):
        if not docs:
            return []

        prompt = """
Você é um avaliador jurídico especializado.

Avalie cada documento de 0 a 10 considerando:
- Relevância jurídica
- Relação com a pergunta
- Precisão normativa
- Pertinência tributária

Retorne SOMENTE um JSON:
[
  {"index": X, "score": Y},
  ...
]
"""

        context = "\n\n".join(
            [f"DOC {i}:\n{d['page_content']}" for i, d in enumerate(docs)]
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"PERGUNTA:\n{query}\n\nDOCUMENTOS:\n{context}")
        ])

        try:
            ranking = eval(response.content)  # seguro porque é sandbox de LLM
        except:
            logger.error("Erro no parsing da resposta do LLM‑judge. Mantendo reranking vetorial.")
            return docs[:top_k]

        ranking_sorted = sorted(ranking, key=lambda x: x["score"], reverse=True)
        selected_indices = [r["index"] for r in ranking_sorted[:top_k]]

        top_docs = [docs[i] for i in selected_indices]
        logger.info(f"⚖️ LLM‑as‑judge selecionou {len(top_docs)} documentos.")
        return top_docs