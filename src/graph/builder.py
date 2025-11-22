from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils.logs import logger

from graph.router import router_node
from graph.nodes import (
    node_direct_answer,
    node_rag_qdrant,
    node_rag_web,
    node_rag_rules,
    node_generate_final,
)


# ===============================
# 🔐 Validação automática do state
# ===============================
def validate_state(state: dict):
    if state is None:
        raise ValueError("State chegou como None no nó do grafo.")

    if not isinstance(state, dict):
        raise ValueError("State deve ser dict.")

    if "messages" not in state:
        raise ValueError("State sem campo 'messages'.")

    return state


# ===============================
# 🧠 Builder do grafo profissional
# ===============================
def build_graph(llm, retriever, web_tool):
    logger.info("⛓️  Construindo LangGraph (versão profissional)...")

    workflow = StateGraph(dict)

    # Checkpointer
    checkpointer = MemorySaver()

    # Nós
    workflow.add_node("direct_answer", lambda s: node_direct_answer(validate_state(s), llm))
    workflow.add_node("rag_qdrant", lambda s: node_rag_qdrant(validate_state(s), retriever))
    workflow.add_node("rag_web", lambda s: node_rag_web(validate_state(s), web_tool))
    workflow.add_node("rag_rules", lambda s: node_rag_rules(validate_state(s)))
    workflow.add_node("generate_final", lambda s: node_generate_final(validate_state(s), llm))

    # Roteador
    workflow.add_conditional_edges(
        "START",
        router_node,
        {
            "direct_answer": "direct_answer",
            "rag_qdrant": "rag_qdrant",
            "rag_web": "rag_web",
            "rag_rules": "rag_rules",
        },
    )

    # Fluxo RAG → final
    workflow.add_edge("rag_qdrant", "generate_final")
    workflow.add_edge("rag_web", "generate_final")
    workflow.add_edge("rag_rules", "generate_final")
    workflow.add_edge("direct_answer", "generate_final")

    workflow.add_edge("generate_final", END)

    app = workflow.compile(checkpointer=checkpointer)

    logger.info("🧠 Grafo compilado com sucesso.")
    return app