# rag/pipeline.py

from utils.logs import logger
from rag.qdrant import QdrantRetriever
from rag.rerank_vector import VectorReranker
from rag.rerank_llm import LLMJudgeReranker


class HybridRAGPipeline:

    def __init__(self, qdrant_retriever: QdrantRetriever, llm, vector_top_k=6, final_top_k=4):
        self.retriever = qdrant_retriever
        self.vector_reranker = VectorReranker()
        self.llm_reranker = LLMJudgeReranker(llm)
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k

    def run(self, question: str, perfil: str):
        logger.info("⚙️ Executando pipeline híbrido de RAG...")

        # -------------------------------------------------------------
        # 1. Recuperação inicial (Qdrant)
        # -------------------------------------------------------------
        try:
            raw_docs = self.retriever.query(question, perfil, limit=12)
        except Exception as e:
            logger.error(f"[RAG] Falha ao consultar Qdrant: {e}")
            return [], ""

        if not raw_docs:
            logger.warning("⚠️ Qdrant não retornou documentos. RAG desativado.")
            return [], ""

        # Filtra textos vazios ou de baixa qualidade
        raw_docs = [d for d in raw_docs if d.get("page_content", "").strip()]
        if not raw_docs:
            logger.warning("⚠️ Todos os documentos retornados estavam vazios.")
            return [], ""

        logger.info(f"📄 Documentos após filtragem inicial: {len(raw_docs)}")

        # -------------------------------------------------------------
        # 2. Reranking Vetorial (Cross‑Encoder)
        # -------------------------------------------------------------
        try:
            vector_docs = self.vector_reranker.rerank(
                question,
                raw_docs,
                top_k=min(self.vector_top_k, len(raw_docs))
            )
        except Exception as e:
            logger.error(f"[RAG] Erro no reranking vetorial: {e}")
            # fallback = pegar documentos crus
            vector_docs = raw_docs[:self.vector_top_k]

        if not vector_docs:
            logger.warning("⚠️ Reranking vetorial retornou zero documentos.")
            vector_docs = raw_docs[:self.vector_top_k]

        logger.info(f"🔁 Documentos pós‑reranking vetorial: {len(vector_docs)}")

        # -------------------------------------------------------------
        # 3. Reranking LLM‑as‑Judge
        # -------------------------------------------------------------
        try:
            final_docs = self.llm_reranker.rerank(
                question,
                vector_docs,
                top_k=min(self.final_top_k, len(vector_docs))
            )
        except Exception as e:
            logger.error(f"[RAG] Erro no LLM‑as‑Judge: {e}")
            # fallback
            final_docs = vector_docs[:self.final_top_k]

        if not final_docs:
            logger.warning("⚠️ LLM‑Judge retornou zero documentos.")
            final_docs = vector_docs[:self.final_top_k]

        logger.info(f"⚖️ Documentos pós‑LLM‑Judge: {len(final_docs)}")

        # -------------------------------------------------------------
        # 4. Consolidação final do contexto
        # -------------------------------------------------------------
        contexto = "\n\n".join(
            (doc.get("page_content") or "").strip()
            for doc in final_docs
        ).strip()

        if not contexto:
            logger.warning("⚠️ Contexto final vazio após pipeline RAG.")
            return [], ""

        # Metadados das fontes
        fontes = [d.get("metadata", {}) for d in final_docs]

        logger.info("✅ Pipeline híbrido RAG concluído com sucesso.")
        return fontes, contexto