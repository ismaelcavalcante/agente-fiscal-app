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

        results = self.client.search(
            collection_name=self.collection,
            query_vector=("default", vector),
            limit=6,
            search_params=SearchParams(hnsw_ef=128)
        )

        docs = results.points

        logger.error(f"Docs retornados: {len(docs)}")

        metadata = []
        contexto = []

        for i, point in enumerate(docs):
            payload = point.payload or {}
            texto = payload.get("page_content", "")

            logger.error(f"[DOC {i}] {texto[:200]}")

            metadata.append(payload)
            contexto.append(texto)

        return metadata, "\n\n".join(contexto)


def build_retriever(url, api_key, collection, embedding_model, openai_key):
    logger.info("Inicializando retriever Qdrant...")

    client = QdrantClient(url=url, api_key=api_key)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_key)

    return RetrieverWrapper(client, embeddings, collection)