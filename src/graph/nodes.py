from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from utils.logs import logger


# ==============================
# ESTADO GLOBAL DO GRAPH
# ==============================

def initial_state(pergunta, perfil):
    return {
        "messages": [HumanMessage(content=pergunta)],
        "ultima_pergunta": pergunta,
        "perfil_cliente": perfil,

        # preenchidos durante a execução
        "contexto_juridico_bruto": "",
        "sources_data": [],
        "rag_ok": False,
    }


# ==============================
# NÓ 1 — RAG COM QDRANT
# ==============================

def node_rag_qdrant(state, retriever):
    logger.debug("📌 node_rag_qdrant…")

    question = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.retrieve_documents(question, perfil)

        state["contexto_juridico_bruto"] = contexto or ""
        state["sources_data"] = fontes or []
        state["rag_ok"] = bool(contexto.strip())

        return state

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro: {e}")
        state["contexto_juridico_bruto"] = ""
        state["sources_data"] = []
        state["rag_ok"] = False
        return state


# ==============================
# NÓ 2 — WEB SEARCH FALLBACK
# ==============================

def node_web_search(state, web_tool):
    logger.debug("📌 node_web_search…")

    # Se o RAG funcionou, não faz fallback
    if state.get("rag_ok"):
        return state

    question = state.get("ultima_pergunta", "")
    result = web_tool.invoke({"query": question})

    state["contexto_juridico_bruto"] = result.get("answer", "")
    state["sources_data"] = result.get("sources", [])
    state["rag_ok"] = bool(state["contexto_juridico_bruto"].strip())

    return state


# ==============================
# NÓ 3 — GERAÇÃO FINAL DO LLM
# ==============================

def node_generate_final(state, llm):
    logger.debug("📌 node_generate_final…")

    question = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", {})
    contexto = state.get("contexto_juridico_bruto", "")
    fontes = state.get("sources_data", [])

    system_prompt = """
Você é um consultor tributário sênior.

Regras:
- Use exclusivamente o contexto fornecido abaixo.
- Se o contexto estiver vazio, diga que não encontrou base jurídica.
- Não invente leis.
- Relacione a resposta ao perfil do cliente.
"""

    resposta = llm.invoke([
        HumanMessage(
            content=(
                f"{system_prompt}\n\n"
                f"PERGUNTA DO CLIENTE:\n{question}\n\n"
                f"PERFIL DO CLIENTE:\n{perfil}\n\n"
                f"CONTEXTO LEGAL DO RAG:\n{contexto}\n\n"
                f"FONTES:\n{fontes}"
            )
        )
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state


# ==============================
# CONSTRUÇÃO DO GRAFO
# ==============================

def create_graph(retriever, web_tool, llm):
    graph = StateGraph(dict)

    graph.add_node("rag", lambda s: node_rag_qdrant(s, retriever))
    graph.add_node("web", lambda s: node_web_search(s, web_tool))
    graph.add_node("final", lambda s: node_generate_final(s, llm))

    graph.set_entry_point("rag")

    graph.add_edge("rag", "web")
    graph.add_edge("web", "final")

    graph.set_finish_point("final")

    return graph.compile()


# ==============================
# EXECUÇÃO (EXEMPLO)
# ==============================

def executar_fluxo(pergunta, perfil, retriever, web_tool, llm):
    state = initial_state(pergunta, perfil)
    graph = create_graph(retriever, web_tool, llm)
    result = graph.invoke(state)
    return result["messages"][-1].content