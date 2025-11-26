# rag/qdrant.py

from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from utils.logs import logger


class QdrantRetriever:

    def __init__(self, url, api_key, collection, embedding_model, openai_key):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_key)

    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)

    def query(self, text: str, perfil: str, limit=12):
        enriched = f"{text}\n\nPerfil: {perfil}"
        logger.info("🔎 Gerando embedding para RAG...")

        try:
            vector = self.embed_query(enriched)
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            return []

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                vector_name="default",
                query_filter=None,
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            logger.error(f"[RAG] Erro ao consultar Qdrant: {e}")
            raise

        docs = []
        for i, point in enumerate(results.points):
            payload = point.payload or {}
            text = payload.get("page_content", "")

            docs.append({
                "index": i,
                "page_content": text,
                "metadata": payload
            })

        logger.info(f"🔎 Qdrant retornou {len(docs)} documentos.")
        return docs