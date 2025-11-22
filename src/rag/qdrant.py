from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from utils.logs import logger


# Palavras-chave tributárias úteis para reforçar a busca
TRIBUTARY_EXPANSION = [
    "reforma tributária",
    "IBS",
    "CBS",
    "LC 214",
    "EC 132",
    "tributação",
    "alíquota",
    "isenção",
    "substituição tributária",
    "benefício fiscal",
    "regra fiscal",
    "imposto"
]


def expand_query(query: str, client_profile: str) -> str:
    """
    Enriquecimento da consulta antes de enviar para o retriever.
    Aumenta recall no Qdrant (essencial para linguagem jurídica).
    """
    enriched = (
        f"{query}\n\n"
        f"Contexto do cliente: {client_profile}\n"
        f"Palavras-chave relacionadas: {', '.join(TRIBUTARY_EXPANSION)}"
    )
    return enriched


def load_qdrant_client(url: str, api_key: str) -> QdrantClient:
    """
    Carrega o cliente Qdrant com logs seguros.
    """
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
    """
    Cria o retriever do Qdrant com MMR otimizado.
    """
    logger.info("Inicializando retriever Qdrant...")

    # Embeddings
    embeddings = OpenAIEmbeddings(
        api_key=openai_api_key,
        model=embedding_model_name
    )

    # Cliente
    client = load_qdrant_client(qdrant_url, qdrant_api_key)

    # Store
    store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # Retriever MMR
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.35
        }
    )

    logger.info("Retriever Qdrant carregado com sucesso.")
    return retriever


def retrieve_documents(retriever, query: str, client_profile: str) -> tuple[list, str]:
    """
    Executa busca no Qdrant com enriquecimento.
    Retorna:
        - lista de documentos
        - contexto concatenado para o LLM
    """

    enriched_query = expand_query(query, client_profile)

    try:
        docs_metadata = []
        for i, d in enumerate(docs_metadata):
            docs_metadata.append({
                "source": d.metadata.get("source", "QDRANT"),
                "page": d.metadata.get("page", None),
                "document_type": d.metadata.get("document_type", "LEI"),
                "chunk_index": i
            })

        # construir contexto legível
        contexto = "\n\n".join(
            [
                f"[Fonte: {d.metadata.get('document_type', 'Lei')} | "
                f"Página: {d.metadata.get('page', '?')}] — "
                f"{d.page_content}"
                for d in docs
            ]
        )

        return docs_metadata, contexto

    except Exception as e:
        logger.error(f"Erro durante busca no Qdrant: {e}")
        return [], ""