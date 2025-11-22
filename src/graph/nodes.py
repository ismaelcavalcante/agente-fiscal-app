from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag.qdrant import retrieve_documents
from rag.web import execute_web_search
from rag.rules import get_fixed_rule_response
from utils.logs import logger
from protocol import ConsultaContext, FonteDocumento


def node_rag_qdrant(state: dict, retriever) -> dict:
    """
    Executa RAG usando Qdrant.
    """
    question = state["messages"][-1].content
    client_profile = state["perfil_cliente"]

    logger.info("Executando RAG (Qdrant)...")

    docs, contexto = retrieve_documents(retriever, question, client_profile)

    return {
        "sources_data": docs,
        "contexto_juridico_bruto": contexto
    }


def node_rag_web(state: dict, web_tool) -> dict:
    """
    Executa busca na Web via Tavily.
    """
    question = state["messages"][-1].content

    logger.info("Executando Web Search...")

    docs, contexto = execute_web_search(web_tool, question)

    return {
        "sources_data": docs,
        "contexto_juridico_bruto": contexto
    }


def node_rag_rules(state: dict) -> dict:
    """
    Retorna regras tributárias fixas.
    """
    question = state["messages"][-1].content

    logger.info("Aplicando regras tributárias fixas...")

    contexto = get_fixed_rule_response(question)

    return {
        "sources_data": [],
        "contexto_juridico_bruto": contexto
    }


def node_direct_answer(state: dict, llm) -> dict:
    """
    Resposta direta do LLM *sem* contexto externo.
    """
    question = state["messages"][-1].content

    logger.info("Gerando resposta direta (sem RAG)...")

    response = llm.invoke([HumanMessage(content=question)])

    return {
        "messages": [AIMessage(content=response.content)]
    }


def node_generate_final(state: dict, llm) -> dict:
    """
    Nó final: combina contexto + pergunta + LLM + gera MCP.
    """
    question = state["messages"][-1].content
    perfil = state["perfil_cliente"]
    contexto = state.get("contexto_juridico_bruto", "")
    sources = state.get("sources_data", []) or []

    # Monta system prompt
    prompt_sistema = f"""
Você é um consultor tributário sênior especializado em legislação brasileira.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO ENCONTRADO (use somente se relevante):
{contexto}

INSTRUÇÕES:
1. Use o contexto acima como fonte primária quando fizer sentido.
2. Se o contexto citar artigos ou leis, mencione‑os explicitamente.
3. Se o contexto estiver vazio, responda com base no seu conhecimento geral.
4. Seja objetivo, profissional e juridicamente preciso.
"""

    # ---------- LLM ----------
    response = llm.invoke([
        SystemMessage(content=prompt_sistema),
        HumanMessage(content=question)
    ])

    # ---------- MCP ----------
    fontes_mcp = []
    for i, fonte in enumerate(sources):
        fontes_mcp.append(
            FonteDocumento(
                document_source=str(fonte.get("source", "desconhecido")),
                page_number=fonte.get("page", None),
                chunk_index=i,
                document_type=str(fonte.get("document_type", "WEB" if fonte.get("source")=="WEB" else "LEI"))
            )
        )

    mcp = ConsultaContext(
        trace_id=state.get("thread_id"),
        perfil_cliente=perfil,
        pergunta_cliente=question,
        contexto_juridico_bruto=contexto,
        fontes_detalhadas=fontes_mcp,
        prompt_mestre=prompt_sistema
    )

    # Salva MCP no estado
    state["mcp"] = mcp

    return {
        "messages": [AIMessage(content=response.content)],
        "mcp": mcp
    }