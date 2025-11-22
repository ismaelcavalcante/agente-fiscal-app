# graph/nodes.py

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.logs import logger
from rag.qdrant import retrieve_documents
from rag.web import execute_web_search
from rag.rules import get_fixed_rule_response
from protocol import ConsultaContext, FonteDocumento


# -------------------------------
# 🔰 Proteção universal do state
# -------------------------------

def ensure_state(state: dict) -> dict:
    """Garante que o state tenha sempre os campos mínimos."""
    if state is None:
        return {"messages": []}

    if "messages" not in state:
        state["messages"] = []

    if "perfil_cliente" not in state:
        state["perfil_cliente"] = "Perfil não informado."

    return state


# -------------------------------
# 🔎 RAG QDRANT
# -------------------------------

def node_rag_qdrant(state: dict, retriever):
    state = ensure_state(state)

    if not state["messages"]:
        return {"messages": [AIMessage(content="Nenhuma pergunta recebida.")]}

    question = state["messages"][-1].content
    logger.info("Executando RAG (Qdrant)...")

    docs, contexto = retrieve_documents(retriever, question, state["perfil_cliente"])

    return {
        "sources_data": docs,
        "contexto_juridico_bruto": contexto
    }


# -------------------------------
# 🌐 RAG WEB
# -------------------------------

def node_rag_web(state: dict, web_tool):
    state = ensure_state(state)

    if not state["messages"]:
        return {"messages": [AIMessage(content="Nenhuma pergunta recebida.")]}

    question = state["messages"][-1].content
    logger.info("Executando Web Search...")

    docs, contexto = execute_web_search(web_tool, question)

    return {
        "sources_data": docs,
        "contexto_juridico_bruto": contexto
    }


# -------------------------------
# 📘 Regras fixas
# -------------------------------

def node_rag_rules(state: dict):
    state = ensure_state(state)

    question = state["messages"][-1].content if state["messages"] else ""
    logger.info("Aplicando regras tributárias fixas...")

    contexto = get_fixed_rule_response(question)

    return {
        "sources_data": [],
        "contexto_juridico_bruto": contexto
    }


# -------------------------------
# ✏️ Resposta direta (sem RAG)
# -------------------------------

def node_direct_answer(state: dict, llm):
    state = ensure_state(state)

    if not state["messages"]:
        return {
            "messages": [
                AIMessage(content="Preciso que você envie uma pergunta.")
            ]
        }

    question = state["messages"][-1].content
    logger.info("Gerando resposta direta (sem RAG)...")

    response = llm.invoke([HumanMessage(content=question)])

    return {
        "messages": [AIMessage(content=response.content)]
    }


# -------------------------------
# 🧠 Nó Final + MCP
# -------------------------------

def node_generate_final(state: dict, llm):
    state = ensure_state(state)

    if not state["messages"]:
        return {"messages": [AIMessage(content="Nenhuma pergunta recebida.")]}

    question = state["messages"][-1].content
    perfil = state["perfil_cliente"]
    contexto = state.get("contexto_juridico_bruto", "")
    fontes = state.get("sources_data", [])

    logger.info("Gerando resposta FINAL com MCP...")

    prompt = f"""
Você é um consultor tributário sênior.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO RECUPERADO:
{contexto}

INSTRUÇÕES:
1. Use o contexto quando relevante.
2. Cite dispositivos legais quando aplicável.
3. Seja preciso, objetivo e técnico.
"""

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ])

    # Construção das fontes MCP
    fontes_mcp = []
    for idx, f in enumerate(fontes):
        fontes_mcp.append(
            FonteDocumento(
                document_source=str(f.get("source", "DESCONHECIDO")),
                page_number=f.get("page"),
                chunk_index=f.get("chunk_index", idx),
                document_type=f.get("document_type", "LEI")
            )
        )

    # Construção do MCP
    mcp = ConsultaContext(
        trace_id=state.get("thread_id"),
        perfil_cliente=perfil,
        pergunta_cliente=question,
        contexto_juridico_bruto=contexto,
        fontes_detalhadas=fontes_mcp,
        prompt_mestre=prompt
    )

    return {
        "messages": [AIMessage(content=response.content)],
        "mcp": mcp
    }