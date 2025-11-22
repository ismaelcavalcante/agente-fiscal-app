from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.logs import logger


# ===========================
# ROUTER NODE
# ===========================
def node_router(state):
    from graph.router import node_router as router
    return router(state)


# ===========================
# RAG NODE (QDRANT MANUAL)
# ===========================
def node_rag_qdrant(state, retriever):
    logger.debug("📌 [node_rag_qdrant] iniciando...")

    pergunta = state["messages"][-1].content
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.retrieve_documents(pergunta, perfil)

        state["contexto_juridico_bruto"] = contexto
        state["sources_data"] = fontes

        return state

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro: {e}")

        state["contexto_juridico_bruto"] = ""
        state["sources_data"] = []
        return state


# ===========================
# WEB SEARCH NODE
# ===========================
def node_web_search(state, web_tool):
    pergunta = state["messages"][-1].content
    result = web_tool.invoke({"query": pergunta})

    state["contexto_juridico_bruto"] = result.get("answer", "")
    state["sources_data"] = result.get("sources", [])

    return state


# ===========================
# DIRECT ANSWER NODE
# ===========================
def node_direct_answer(state, llm):

    pergunta = state["messages"][-1].content

    resposta = llm.invoke([
        SystemMessage(content="Você é um consultor tributário especializado."),
        HumanMessage(content=pergunta)
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state


# ===========================
# FINAL ANSWER NODE
# ===========================
def node_generate_final(state, llm):

    pergunta = state["messages"][-1].content
    perfil = state.get("perfil_cliente", "")
    contexto = state.get("contexto_juridico_bruto", "")
    fontes = state.get("sources_data", [])

    prompt = f"""
Você é um consultor tributário sênior.

REGRAS:
- Use exclusivamente o CONTEXTO JURÍDICO abaixo.
- Se o contexto estiver vazio, responda:
  "Com base no meu corpus atual (Qdrant), não encontrei fundamento jurídico para responder."
- Não invente leis, artigos ou jurisprudência.
- Conecte a resposta ao PERFIL da empresa.
- Cite as FONTES retornadas pelo Qdrant.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO JURÍDICO RECUPERADO:
{contexto}

FONTES:
{fontes}

Pergunta:
{pergunta}
"""

    resposta = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=pergunta)
    ])

    state["messages"].append(AIMessage(content=resposta.content))
    return state