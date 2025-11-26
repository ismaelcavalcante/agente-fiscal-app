from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from functools import partial

from graph.router import node_router
from graph.nodes import node_rag_qdrant, node_web_search, node_generate_final
from utils.logs import logger


class GraphState(TypedDict, total=False):
    messages: list
    ultima_pergunta: str
    perfil_cliente: Any
    contexto_juridico_bruto: str
    sources_data: list
    rag_ok: bool
    __route__: str


def build_graph(llm, rag_pipeline, web_tool):
    """
    Constrói o grafo principal da aplicação.
    """

    logger.info("⚙️ Construindo LangGraph...")

    workflow = StateGraph(GraphState)

    workflow.add_node("router", node_router)
    workflow.add_node("rag_qdrant", partial(node_rag_qdrant, retriever=rag_pipeline))
    workflow.add_node("web_search", partial(node_web_search, web_tool=web_tool))
    workflow.add_node("generate_final", partial(node_generate_final, llm=llm))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda s: s.get("__route__", "RAG"),
        {"RAG": "rag_qdrant", "WEB": "web_search"},
    )

    workflow.add_edge("rag_qdrant", "generate_final")
    workflow.add_edge("web_search", "generate_final")
    workflow.add_edge("generate_final", END)

    graph = workflow.compile()
    logger.info("🧠 LangGraph compilado com sucesso.")
    return graph