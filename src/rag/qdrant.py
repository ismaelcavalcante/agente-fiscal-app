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

        # ============================================================
        # AUTO-DETECÇÃO DE API DO QDRANT
        # ============================================================

        # 🔥 Caso moderno (v1.x)
        if hasattr(self.client, "query_points"):
            try:
                results = self.client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    vector_name="default",
                    limit=6,
                    search_params=SearchParams(hnsw_ef=128)
                )
                docs = results.points
                return self._format_docs(docs)

            except TypeError:
                # versão moderna porém sem vector_name
                results = self.client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    limit=6,
                )
                docs = results.points
                return self._format_docs(docs)

        # 🔥 API intermediária (v0.9)
        if hasattr(self.client, "search"):
            try:
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=("default", vector),
                    limit=6
                )
                return self._legacy_format(results)

            except TypeError:
                # sem vector_name
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=6
                )
                return self._legacy_format(results)

        # 🔥 API legacy de verdade (v0.3.x)
        if hasattr(self.client, "search_collection"):
            results = self.client.search_collection(
                collection_name=self.collection,
                query_vector=vector,
                limit=6
            )
            return self._legacy_format(results)

        # 🔥 API pré-histórica (v0.1.x – v0.2.x)
        if hasattr(self.client, "search"):
            results = self.client.search(
                self.collection,  # positional
                vector,           # embedding
                6                 # top-k
            )
            return self._legacy_format(results)

        raise RuntimeError("Nenhuma API conhecida encontrada no QdrantClient")


    def _format_docs(self, docs):
        metadata = []
        contexto = []

        logger.error(f"Docs retornados: {len(docs)}")
        for i, p in enumerate(docs):
            payload = p.payload or {}
            text = payload.get("page_content", "")
            logger.error(f"[DOC {i}] {text[:200]}")
            metadata.append(payload)
            contexto.append(text)

        return metadata, "\n\n".join(contexto)


    def _legacy_format(self, docs):
        metadata = []
        contexto = []

        logger.error(f"Docs retornados (legacy): {len(docs)}")
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