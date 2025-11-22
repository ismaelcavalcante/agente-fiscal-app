from langgraph.graph import StateGraph, END
from graph.nodes import (
    node_router,
    node_rag_qdrant,
    node_web_search,
    node_direct_answer,
    node_generate_final,
)
from utils.logs import logger


def build_graph(llm, retriever, web_tool):

    logger.info("⛓️  Construindo LangGraph (versão profissional)...")

    workflow = StateGraph()

    # Nós
    workflow.add_node("router", lambda state: node_router(state))
    workflow.add_node("rag_qdrant", lambda state: node_rag_qdrant(state, retriever))
    workflow.add_node("web_search", lambda state: node_web_search(state, web_tool))
    workflow.add_node("direct_answer", lambda state: node_direct_answer(state, llm))
    workflow.add_node("generate_final", lambda state: node_generate_final(state, llm))

    # Estado inicial
    workflow.set_entry_point("router")

    # Transições
    workflow.add_conditional_edges(
        "router",
        lambda output: output,
        {
            "RAG": "rag_qdrant",
            "WEB": "web_search",
            "DIRECT": "direct_answer",
        }
    )

    workflow.add_edge("rag_qdrant", "generate_final")
    workflow.add_edge("web_search", "generate_final")
    workflow.add_edge("direct_answer", "generate_final")

    # Encerrar
    workflow.add_edge("generate_final", END)

    graph = workflow.compile()

    logger.info("🧠 Grafo compilado com sucesso.")

    return graph