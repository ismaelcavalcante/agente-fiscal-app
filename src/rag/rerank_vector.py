# rag/rerank_vector.py

from sentence_transformers import CrossEncoder
from utils.logs import logger


class VectorReranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"🔁 Carregando CrossEncoder {model_name} para reranking vetorial...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_k=6):
        if not docs:
            return []

        pairs = [[query, d["page_content"]] for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        top_docs = [doc for score, doc in ranked[:top_k]]
        logger.info(f"🔁 Reranking vetorial selecionou {len(top_docs)} documentos.")
        return top_docs