from langchain_community.tools.tavily_search import TavilySearchResults
from utils.logs import logger


def build_web_tool(tavily_api_key: str) -> TavilySearchResults:
    """
    Inicializa o cliente Tavily com número reduzido de resultados
    para melhor performance no Streamlit Cloud.
    """
    try:
        tool = TavilySearchResults(
            max_results=3,
            api_key=tavily_api_key
        )
        logger.info("Tavily Web Search iniciado com sucesso.")
        return tool
    except Exception as e:
        logger.error(f"Erro ao inicializar Tavily: {e}")
        raise


def execute_web_search(tool: TavilySearchResults, query: str) -> tuple[list, str]:
    """
    Executa busca na Web com fallback silencioso.
    Retorna:
        - lista de documentos
        - contexto formatado para LLM
    """
    try:
        results = tool.invoke(query)

        if not results:
            logger.info("Web search retornou vazio.")
            return [], ""

        context_blocks = []
        docs_metadata = []

        for item in results:
            content = item.get("content", "")
            url = item.get("url", "")
            snippet = item.get("snippet", "")

            bloco = f"WEB Result — URL: {url}\n{snippet}\n\n{content}"
            context_blocks.append(bloco)

            docs_metadata.append({
                "source": "WEB",
                "url": url,
                "page": None,
                "document_type": "WEB",
                "chunk_index": len(docs_metadata)
            })

        contexto = "\n---\n".join(context_blocks)
        return docs_metadata, contexto

    except Exception as e:
        logger.error(f"Erro executando Tavily Search: {e}")
        return [], ""