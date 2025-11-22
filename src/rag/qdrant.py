from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from utils.logs import logger

TRIBUTARY_EXPANSION = [
    "IBS", "CBS", "EC 132", "LC 214",
    "tributação", "crédito", "não cumulatividade",
    "benefício fiscal", "regra fiscal"
]


def expand_query(query, client_profile):
    return (
        f"{query}\n\n"
        f"Perfil: {client_profile}\n"
        f"Keywords: {', '.join(TRIBUTARY_EXPANSION)}"
    )


def load_qdrant_client(url, api_key):
    client = QdrantClient(url=url, api_key=api_key)
    logger.info("QdrantClient conectado.")
    return client


def build_retriever(url, api_key, collection, embedding_model, openai_key):
    logger.info("Inicializando retriever Qdrant...")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_key
    )

    client = load_qdrant_client(url, api_key)

    store = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name=collection,
        vector_name="default"
    )

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 25, "lambda_mult": 0.3}
    )

    return RetrieverWrapper(retriever)


class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve_documents(self, query, client_profile=""):
        enriched = expand_query(query, client_profile)

        docs = self.retriever.invoke(enriched)

        logger.error("=== DEBUG RAG ===")
        logger.error(f"Query enriquecida: {enriched}")
        logger.error(f"Docs retornados: {len(docs)}")

        for i, d in enumerate(docs):
            logger.error(
                f"[DOC {i}] page={d.metadata.get('page')} "
                f"type={d.metadata.get('document_type')} "
                f"conteudo={d.page_content[:200]}..."
            )

        metadata = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "document_type": d.metadata.get("document_type", "Lei")
            }
            for d in docs
        ]

        contexto = "\n\n".join(
            f"[{d.metadata.get('document_type','Lei')} pg {d.metadata.get('page')}] "
            f"{d.page_content}"
            for d in docs
        )

        return metadata, contexto