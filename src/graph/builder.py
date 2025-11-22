from langgraph.graph import StateGraph, END
from graph.nodes import (
    node_direct_answer,
    node_rag_qdrant,
    node_generate_final
)
from graph.router import router_node
from utils.logs import logger


def build_graph(llm, retriever, web_tool=None):

    logger.info("⛓️  Construindo LangGraph (versão profissional)...")

    workflow = StateGraph(dict)

    # Nós
    workflow.add_node("direct_answer", lambda s: node_direct_answer(s, llm))
    workflow.add_node("rag_qdrant", lambda s: node_rag_qdrant(s, retriever))
    workflow.add_node("generate_final", lambda s: node_generate_final(s, llm))

    # Roteador
    workflow.set_entry_point(router_node)

    # Encadeamento
    workflow.add_edge("direct_answer", "generate_final")
    workflow.add_edge("rag_qdrant", "generate_final")
    workflow.add_edge("generate_final", END)

    graph = workflow.compile()

    logger.info("🧠 Grafo compilado com sucesso.")
    return graph