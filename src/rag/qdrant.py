import json
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from utils.logs import logger


def build_retriever(qdrant_url, qdrant_api_key, collection_name, embedding_model_name, openai_api_key):

    logger.info("Iniciando retriever Qdrant...")

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=openai_api_key
    )

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        vector_name="default"   # 🔥 obrigatório
    )

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}   # número de chunks
    )

    logger.info("Retriever Qdrant carregado com sucesso.")

    return RetrieverWrapper(retriever)


class RetrieverWrapper:

    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve_documents(self, query):
        docs = self.retriever.invoke(query)

        # --- extrair texto (page_content) ---
        contexto_concat = "\n\n".join([d.page_content for d in docs])

        # --- extrair metadados ---
        metadata_list = [dict(d.metadata) for d in docs]

        return metadata_list, contexto_concat