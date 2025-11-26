from langchain_community.tools.tavily_search import TavilySearchResults
from utils.logs import logger


class WebSearch:

    def __init__(self, api_key: str, max_results: int = 3):
        """
        Inicializa wrapper para Tavily.
        """
        try:
            self.tool = TavilySearchResults(
                api_key=api_key,
                max_results=max_results
            )
            logger.info("🌐 Tavily WebSearch inicializado.")
        except Exception as e:
            logger.error(f"Erro ao inicializar Tavily: {e}")
            raise

    def execute(self, query: str) -> dict:
        """
        Executa busca e retorna:
        {
          "answer": "texto concatenado",
          "sources": [ { ... }, ... ]
        }
        """
        try:
            # Formato correto para tools LangChain: dict(query=foo)
            results = self.tool.invoke({"query": query}) or []
        except Exception as e:
            logger.error(f"Erro executando web search: {e}")
            return {"answer": "", "sources": []}

        if not results:
            logger.info("🌐 Web search retornou vazio.")
            return {"answer": "", "sources": []}

        context_blocks = []
        sources = []

        for idx, item in enumerate(results):

            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or "")[:300].strip()
            content = (item.get("content") or "")[:1200].strip()

            bloco = (
                f"WEB RESULTADO {idx+1}\n"
                f"URL: {url}\n\n"
                f"{snippet}\n\n"
                f"{content}"
            )

            context_blocks.append(bloco)

            sources.append({
                "source": "WEB",
                "url": url,
                "chunk_index": idx,
                "document_type": "WEB",
            })

        final_context = "\n\n---\n\n".join(context_blocks)
        return {"answer": final_context, "sources": sources}


def build_web_tool(api_key: str) -> WebSearch:
    """
    Função auxiliar usada em app_web e no LangGraph.
    """
    return WebSearch(api_key=api_key)