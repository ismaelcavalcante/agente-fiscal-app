from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from langchain_openai import OpenAIEmbeddings
from utils.logs import logger


TRIBUTARY_EXPANSION = [
    "IBS", "CBS", "EC 132", "LC 214",
    "tributação", "crédito", "não cumulatividade",
    "benefício fiscal", "regra fiscal", "ICMS"
]


def expand_query(query, client_profile):
    return (
        f"{query}\n\n"
        f"Perfil: {client_profile}\n"
        f"Palavras-chave: {', '.join(TRIBUTARY_EXPANSION)}"
    )


def build_retriever(url, api_key, collection, embedding_model, openai_key):
    logger.info("Inicializando retriever Qdrant...")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_key
    )

    client = QdrantClient(
        url=url,
        api_key=api_key,
    )

    return RetrieverWrapper(client, embeddings, collection)


class RetrieverWrapper:

    def __init__(self, client, embeddings, collection):
        self.client = client
        self.embeddings = embeddings
        self.collection = collection

    def retrieve_documents(self, query, client_profile=""):

        enriched = expand_query(query, client_profile)

        logger.error("=== DEBUG RAG (manual retriever) ===")
        logger.error(f"Consulta enriquecida: {enriched}")

        # 🔥 1. Gerar embedding da consulta
        vector = self.embeddings.embed_query(enriched)

        # 🔥 2. Buscar pontos usando API MODERNA
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=6,
            search_params=SearchParams(
                hnsw_ef=128
            )
        )

        docs = results.points

        logger.error(f"Quantidade de documentos retornados: {len(docs)}")

        # 🔥 3. Extrair contexto
        metadata_list = []
        full_context = []

        for i, point in enumerate(docs):
            md = point.payload or {}
            texto = md.get("page_content", "")

            logger.error(
                f"[DOC {i}] page={md.get('page')} "
                f"type={md.get('document_type')} "
                f"conteudo={texto[:200]}..."
            )

            metadata_list.append({
                "source": md.get("source", "QDRANT"),
                "page": md.get("page"),
                "document_type": md.get("document_type")
            })

            full_context.append(
                f"[{md.get('document_type')} pg {md.get('page')}] {texto}"
            )

        return metadata_list, "\n\n".join(full_context)