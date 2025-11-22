from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.router import router_node
from graph.nodes import (
    node_rag_qdrant,
    node_rag_web,
    node_rag_rules,
    node_direct_answer,
    node_generate_final
)

from utils.logs import logger


def build_graph(llm, retriever, web_tool):
    """
    Monta o grafo completo do agente fiscal profissional.
    """

    logger.info("⛓️  Construindo LangGraph (versão profissional)...")

    class AgentState(dict):
        """
        Estado do LangGraph.
        Campos usados:
            - messages
            - perfil_cliente
            - sources_data
            - contexto_juridico_bruto
            - thread_id
        """
        pass

    workflow = StateGraph(AgentState)

    # Nós
    workflow.add_node("rag_qdrant", lambda s: node_rag_qdrant(s, retriever))
    workflow.add_node("rag_web", lambda s: node_rag_web(s, web_tool))
    workflow.add_node("rag_rules", node_rag_rules)
    workflow.add_node("direct_answer", lambda s: node_direct_answer(s, llm))
    workflow.add_node("generate_final", lambda s: node_generate_final(s, llm))

    # Roteador
    workflow.add_conditional_edges(
        START,
        router_node,
        {
            "rag_qdrant": "rag_qdrant",
            "rag_web": "rag_web",
            "rag_rules": "rag_rules",
            "direct_answer": "direct_answer",
        },
    )

    # Após cada caminho, gerar resposta final
    workflow.add_edge("rag_qdrant", "generate_final")
    workflow.add_edge("rag_web", "generate_final")
    workflow.add_edge("rag_rules", "generate_final")
    workflow.add_edge("direct_answer", END)

    # Nó final
    workflow.add_edge("generate_final", END)

    # Checkpoint (memória leve)
    memory = MemorySaver()

    logger.info("🧠 Grafo compilado com sucesso.")

    return workflow.compile(checkpointer=memory)