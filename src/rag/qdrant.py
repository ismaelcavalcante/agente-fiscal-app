from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from langchain_openai import OpenAIEmbeddings
from utils.logs import logger


class RetrieverWrapper:

    def __init__(self, client, embeddings, collection):
        self.client = client
        self.embeddings = embeddings
        self.collection = collection

    def retrieve_documents(self, query, perfil):

        enriched = f"{query}\n\nPerfil:{perfil}"
        logger.error("=== DEBUG RAG (manual) ===")
        logger.error(f"Consulta enriquecida: {enriched}")

        vector = self.embeddings.embed_query(enriched)

        logger.error("=== VECTOR ===")
        logger.error(f"Consulta enriquecida: {vector}")

        # ============================================================
        # AQUI ESTÁ A CHAMADA CORRETA PARA SUA COLLECTION
        # ============================================================
        try:
            results = self.client.query_points(
                collection_name=self.collection,                
                vector_name="default",
                limit=6,
                with_payload=False,                
            )
        except Exception as e:
            logger.error(f"[RAG_QDRANT] Erro: {e}")
            raise

        docs = results.points
        logger.error(f"Docs retornados: {len(docs)}")

        metadata = []
        contexto = []

        for i, p in enumerate(docs):
            payload = p.payload or {}
            text = payload.get("page_content", "")
            logger.error(f"[DOC {i}] {text[:200]}")
            metadata.append(payload)
            contexto.append(text)

        return metadata, "\n\n".join(contexto)


def build_retriever(url, api_key, collection, embedding_model, openai_key):
    logger.info("Inicializando retriever Qdrant...")

    client = QdrantClient(url=url, api_key=api_key)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_key)

    return RetrieverWrapper(client, embeddings, collection)