import traceback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.logs import logger
from protocol import ConsultaContext, FonteDocumento

# ===============================
# 🔐 Segurança: valida estado
# ===============================
def require_messages(state: dict):
    if state is None:
        raise ValueError("STATE chegou como None")

    if "messages" not in state:
        raise ValueError("STATE não possui campo 'messages'")

    msgs = state["messages"]

    if msgs is None:
        raise ValueError("'messages' chegou como None")

    if len(msgs) == 0:
        raise ValueError("'messages' está vazio")

    return msgs


# ===============================
# 🟦 Nó: resposta direta
# ===============================
def node_direct_answer(state: dict, llm):
    logger.debug("📌 [node_direct_answer] Recebendo state...")
    logger.debug(f"STATE: {state}")

    try:
        msgs = require_messages(state)
        question = msgs[-1].content
    except Exception as e:
        logger.error(f"direct_answer → erro ao validar state: {e}")
        return {
            "messages": [AIMessage(content="Não consegui interpretar sua pergunta.")],
            "mcp": None,
        }

    sys_prompt = """
Você é um consultor tributário profissional.  
Responda com clareza, precisão técnica e objetividade.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=question)
        ])
    except Exception as e:
        logger.error(f"Erro no LLM direct_answer: {e}")
        traceback.print_exc()
        return {
            "messages": [AIMessage(content="Erro ao gerar resposta.")],
            "mcp": None,
        }

    return {
        "messages": [AIMessage(content=response.content)],
        "mcp": None
    }


# ===============================
# 🟦 Nó: RAG Qdrant
# ===============================
def node_rag_qdrant(state: dict, retriever):
    logger.debug("📌 [node_rag_qdrant] Recebendo state...")

    try:
        msgs = require_messages(state)
        question = msgs[-1].content
    except Exception as e:
        logger.error(f"rag_qdrant → state inválido: {e}")
        return {"messages": [AIMessage(content="Erro interno no RAG Qdrant.")], "mcp": None}

    try:
        docs_metadata, contexto_texto = retriever.retrieve_documents(question)
    except Exception as e:
        logger.error(f"[RAG_QDRANT] Erro ao recuperar docs: {e}")
        return {
            "messages": [AIMessage(content="Não foi possível recuperar documentos jurídicos.")],
            "mcp": None
        }

    state["contexto_juridico_bruto"] = contexto_texto
    state["sources_data"] = docs_metadata

    return state


# ===============================
# 🟦 Nó: RAG Web Search (Tavily)
# ===============================
def node_rag_web(state: dict, webtool):
    logger.debug("📌 [node_rag_web] Recebendo state...")

    try:
        msgs = require_messages(state)
        question = msgs[-1].content
    except Exception as e:
        logger.error(f"rag_web → state inválido: {e}")
        return {"messages": [AIMessage(content="Erro interno no RAG Web.")], "mcp": None}

    try:
        docs_metadata, contexto_texto = webtool.execute_web_search(question)
    except Exception as e:
        logger.error(f"[RAG_WEB] erro: {e}")
        return {
            "messages": [AIMessage(content="Não foi possível realizar a busca web.")],
            "mcp": None
        }

    state["contexto_juridico_bruto"] = contexto_texto
    state["sources_data"] = docs_metadata

    return state


# ===============================
# 🟦 Nó: RAG Regras Fixas
# ===============================
def node_rag_rules(state: dict):
    logger.debug("📌 [node_rag_rules] Executando regras fixas...")

    state["contexto_juridico_bruto"] = """
EC 132 — Mudanças estruturais no IBS e CBS.
LC 214 — Regulamentação operacional, fiscalização compartilhada,
obrigações acessórias e distribuição da receita.
"""
    state["sources_data"] = [
        {"source": "EC 132", "page": None, "document_type": "LEI"},
        {"source": "LC 214", "page": None, "document_type": "LEI"},
    ]

    return state


# ===============================
# 🟦 Nó final: gera resposta + MCP
# ===============================
def node_generate_final(state: dict, llm):
    logger.debug("📌 [node_generate_final] Gerando resposta final...")

    try:
        msgs = require_messages(state)
        question = msgs[-1].content
    except Exception as e:
        logger.error(f"generate_final → state inválido: {e}")
        return {
            "messages": [AIMessage(content="Erro ao gerar resposta final.")],
            "mcp": None
        }

    perfil = state.get("perfil_cliente", "Perfil não informado")
    contexto = state.get("contexto_juridico_bruto", "")
    sources_list = state.get("sources_data", [])

    prompt_sistema = f"""
Você é um consultor tributário sênior.
Use o contexto jurídico somente quando útil.

PERFIL DO CLIENTE:
{perfil}

CONTEXTO:
{contexto}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=prompt_sistema),
            HumanMessage(content=question)
        ])
    except Exception as e:
        logger.error(f"Erro no LLM final: {e}")
        return {
            "messages": [AIMessage(content="Erro ao gerar resposta final.")],
            "mcp": None
        }

    fontes_mcp = []
    for i, f in enumerate(sources_list):
        fontes_mcp.append(
            FonteDocumento(
                document_source=str(f.get("source", "DESCONHECIDO")),
                page_number=f.get("page", None),
                chunk_index=i,
                document_type=str(f.get("document_type", "LEI"))
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

    return {
        "messages": [AIMessage(content=response.content)],
        "mcp": mcp
    }