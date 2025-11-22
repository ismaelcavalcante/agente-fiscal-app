from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from utils.logs import logger 



TRIBUTARY_EXPANSION = [
    "IBS", "CBS", "EC 132", "LC 214",
    "tributação", "não cumulatividade",
    "crédito", "compensação", "fiscal",
    "imposto", "ICMS", "PIS", "COFINS"
]


def expand_query(query: str, client_profile: str) -> str:
    return (
        f"{query}\n\n"
        f"Perfil do cliente: {client_profile}\n"
        f"Palavras-chave relacionadas: {', '.join(TRIBUTARY_EXPANSION)}"
    )


def load_qdrant_client(url: str, api_key: str):
    try:
        client = QdrantClient(url=url, api_key=api_key)
        logger.info("QdrantClient conectado com sucesso.")
        return client
    except Exception as e:
        logger.error(f"Erro ao conectar Qdrant: {e}")
        raise


def build_retriever(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    embedding_model_name: str,
    openai_api_key: str
):
    logger.info("Inicializando retriever Qdrant...")

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=openai_api_key
    )

    client = load_qdrant_client(qdrant_url, qdrant_api_key)

    store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        vector_name="default"
    )

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 25, "lambda_mult": 0.3}
    )

    logger.info("Retriever Qdrant carregado com sucesso.")
    return RetrieverWrapper(retriever)


class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve_documents(self, query, client_profile=""):

        enriched = expand_query(query, client_profile)

        try:
            docs = self.retriever.invoke(enriched)

            # DEBUG COMPLETO
            logger.error("=== DEBUG RAG ===")
            logger.error(f"Consulta enriquecida: {enriched}")
            logger.error(f"Quantidade de documentos retornados: {len(docs)}")

            for i, d in enumerate(docs):
                logger.error(
                    f"[DOC {i}] page={d.metadata.get('page')} "
                    f"type={d.metadata.get('document_type')} "
                    f"conteudo_inicio={d.page_content[:200]}..."
                )

            metadata_list = [
                {
                    "source": d.metadata.get("source", "QDRANT"),
                    "page": d.metadata.get("page"),
                    "document_type": d.metadata.get("document_type", "LEI"),
                }
                for d in docs
            ]

            contexto = "\n\n".join(
                [
                    f"[{d.metadata.get('document_type', 'Lei')} pg {d.metadata.get('page', '?')}] "
                    f"{d.page_content}"
                    for d in docs
                ]
            )

            return metadata_list, contexto

        except Exception as e:
            logger.error(f"Erro durante busca no Qdrant: {e}")
            return [], ""