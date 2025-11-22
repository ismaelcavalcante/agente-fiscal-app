from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.logs import logger
from protocol import ConsultaContext, FonteDocumento


def require_messages(state):
    if "messages" not in state or not state["messages"]:
        raise ValueError("State sem mensagens.")
    return state["messages"]


# ============================
#   NODE — DIRECT ANSWER
# ============================

def node_direct_answer(state: dict, llm):
    logger.debug("[node_direct_answer] Respondendo direto (sem RAG).")

    msgs = require_messages(state)
    question = msgs[-1].content

    resposta = llm.invoke([
        SystemMessage(content="Você é um consultor tributário experiente."),
        HumanMessage(content=question)
    ])

    return {
        "messages": [AIMessage(content=resposta.content)]
    }


# ============================
#   NODE — RAG QDRANT (MANUAL)
# ============================

def node_rag_qdrant(state: dict, retriever):
    logger.debug("[node_rag_qdrant] Executando busca Qdrant...")

    msgs = require_messages(state)
    question = msgs[-1].content
    perfil = state.get("perfil_cliente", "")

    try:
        metadata, contexto = retriever.retrieve_documents(
            query=question,
            client_profile=perfil
        )

        logger.debug(f"[node_rag_qdrant] Recuperou {len(metadata)} documentos.")
        return {
            "contexto_juridico_bruto": contexto,
            "sources_data": metadata
        }

    except Exception as e:
        logger.error(f"[RAG_QDRANT] Falha: {e}")
        return {
            "contexto_juridico_bruto": "",
            "sources_data": []
        }


# ============================
#   NODE — GERAÇÃO FINAL
# ============================

def node_generate_final(state: dict, llm):
    logger.debug("[node_generate_final] Gerando resposta final...")

    msgs = require_messages(state)
    question = msgs[-1].content
    perfil = state.get("perfil_cliente", "")
    contexto = state.get("contexto_juridico_bruto", "")
    sources = state.get("sources_data", [])

    fontes_txt = "\n".join(
        [
            f"- {s.get('document_type')} (página {s.get('page')}, fonte {s.get('source')})"
            for s in sources
        ]
    )

    prompt = f"""
Você é um consultor tributário sênior especializado em IBS/CBS/EC 132/LC 214.

REGRAS:
1. Só use o contexto jurídico recuperado (NÃO invente).
2. Se o contexto não contiver base suficiente, responda:
   "Com base no meu corpus atual (Qdrant), não encontrei fundamento jurídico para responder."
3. Responda de forma objetiva, jurídica e vinculada ao perfil abaixo.
4. Sempre cite as fontes no final.

PERFIL DO CLIENTE:
{perfil}

TRECHOS RECUPERADOS:
{contexto}

FONTES:
{fontes_txt}

Pergunta:
"{question}"
"""

    resposta = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ])

    fontes_mcp = [
        FonteDocumento(
            document_source=d.get("source"),
            page_number=d.get("page"),
            chunk_index=i,
            document_type=d.get("document_type")
        )
        for i, d in enumerate(sources)
    ]

    mcp = ConsultaContext(
        trace_id=state.get("thread_id"),
        perfil_cliente=perfil,
        pergunta_cliente=question,
        contexto_juridico_bruto=contexto,
        fontes_detalhadas=fontes_mcp,
        prompt_mestre=prompt
    )

    return {
        "messages": [AIMessage(content=resposta.content)],
        "mcp": mcp
    }