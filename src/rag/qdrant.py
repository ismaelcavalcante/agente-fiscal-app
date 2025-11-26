from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from langchain_openai import OpenAIEmbeddings
from utils.logs import logger


class QdrantRetriever:

    def __init__(self, url, api_key, collection, embedding_model, openai_key):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_key)
        self.collection = collection

    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)

    def query(self, text: str, perfil: str, limit=12):
        enriched = f"{text}\n\nPerfil: {perfil}"
        logger.info("🔎 Gerando embedding para RAG...")
        query_vector = self.embed_query(enriched)

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            )
        except Exception as e:
            logger.error(f"[RAG] Erro ao consultar Qdrant: {e}")
            raise

        docs = []
        for i, point in enumerate(results):
            payload = point.payload or {}
            text = payload.get("page_content", "") or ""

            docs.append({
                "index": i,
                "page_content": text,
                "metadata": payload,
            })

        logger.info(f"🔎 Qdrant retornou {len(docs)} documentos (pré‑reranking).")
        return docs