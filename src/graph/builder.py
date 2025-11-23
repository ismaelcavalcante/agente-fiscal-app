from langgraph.graph import StateGraph, END
from graph.router import node_router
from graph.nodes import (
    node_rag_qdrant,
    node_web_search,
    node_generate_final,
)
from utils.logs import logger

state_schema = dict


def build_graph(llm, retriever, web_tool):
    logger.info("⛓️  Construindo LangGraph (versão profissional)...")

    workflow = StateGraph(state_schema)

    workflow.add_node("router", node_router)
    workflow.add_node("rag_qdrant", lambda s: node_rag_qdrant(s, retriever))
    workflow.add_node("web_search", lambda s: node_web_search(s, web_tool))
    workflow.add_node("generate_final", lambda s: node_generate_final(s, llm))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["__route__"],
        {
            "RAG": "rag_qdrant",
            "WEB": "web_search",
        }
    )

    workflow.add_edge("rag_qdrant", "generate_final")
    workflow.add_edge("web_search", "generate_final")
    workflow.add_edge("generate_final", END)

    graph = workflow.compile()
    logger.info("🧠 Grafo compilado com sucesso.")

    return graph