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

        raw_docs = self.retriever.query(question, perfil, limit=12)
        logger.info("Passo 1: Qdrant OK.")

        vector_docs = self.vector_reranker.rerank(
            question,
            raw_docs,
            top_k=self.vector_top_k
        )
        logger.info("Passo 2: Reranking vetorial OK.")

        final_docs = self.llm_reranker.rerank(
            question,
            vector_docs,
            top_k=self.final_top_k
        )
        logger.info("Passo 3: LLM‑as‑Judge OK.")

        contexto = "\n\n".join(d["page_content"] for d in final_docs)
        metadata = [d["metadata"] for d in final_docs]

        logger.info("Pipeline híbrido concluído.")
        return metadata, contexto