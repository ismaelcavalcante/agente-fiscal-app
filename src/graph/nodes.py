from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.logs import logger

from rag.pipeline import HybridRAGPipeline
from rag.web import WebSearch

from protocol import ConsultaContext
from mcp_converters import convert_sources
from prompt_hierarchy import montar_prompt_mestre


# ============================================================
# NODE 01 — RAG (pipeline híbrido: Qdrant → Vetorial → LLM-Judge)
# ============================================================

def node_rag_qdrant(state, retriever: HybridRAGPipeline):
    """
    Executa pipeline completo de RAG híbrido:
    Qdrant → reranking vetorial → reranking LLM.
    Retorna novo estado contendo:
      - contexto_juridico_bruto
      - sources_data
      - rag_ok
    """

    pergunta = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", "")

    logger.info("🔎 NODE_RAG — iniciando pipeline híbrido...")

    try:
        fontes, contexto = retriever.run(pergunta, perfil)

        return {
            "contexto_juridico_bruto": contexto or "",
            "sources_data": fontes or [],
            "rag_ok": bool((contexto or "").strip()),
        }

    except Exception as e:
        logger.error(f"❌ [NODE_RAG] Erro no pipeline híbrido: {e}")

        return {
            "contexto_juridico_bruto": "",
            "sources_data": [],
            "rag_ok": False,
        }



# ============================================================
# NODE 02 — WEB SEARCH (fallback quando RAG não é suficiente)
# ============================================================

def node_web_search(state, web_tool: WebSearch):
    """
    Executa WebSearch somente quando:
    - RAG falhou (rag_ok=False)
    - OU o roteador indicou caminho WEB

    Retorna:
      - contexto_juridico_bruto
      - sources_data
      - rag_ok
    """

    if state.get("rag_ok"):   # RAG já resolveu → skip
        logger.info("🌐 NODE_WEB — ignorado (RAG já encontrou resposta)")
        return {}

    pergunta = state.get("ultima_pergunta", "")

    logger.info("🌐 NODE_WEB — executando Tavily WebSearch...")

    result = web_tool.execute(pergunta)

    contexto = result.get("answer", "") or ""
    fontes = result.get("sources", []) or []

    return {
        "contexto_juridico_bruto": contexto,
        "sources_data": fontes,
        "rag_ok": bool(contexto.strip()),
    }



# ============================================================
# NODE 03 — GERADOR FINAL (MCP + Prompt Hierárquico + Resposta do LLM)
# ============================================================

def node_generate_final(state, llm):
    """
    Node final:
      1. Constrói o MCP (Model Context Protocol)
      2. Converte fontes do RAG/WEB para FonteDocumento
      3. Gera prompt hierárquico profissional
      4. Invoca LLM com system prompt robusto
      5. Retorna mensagem gerada e atualiza o histórico
    """

    logger.info("🧠 NODE_FINAL — Montando MCP e gerando resposta final...")

    pergunta = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", "")
    contexto = state.get("contexto_juridico_bruto", "")
    fontes_raw = state.get("sources_data", [])
    historico = list(state.get("messages", []))

    # --------------------------------------------------------
    # 1) Converter fontes → MCP FonteDocumento
    # --------------------------------------------------------
    fontes = convert_sources(fontes_raw)

    # --------------------------------------------------------
    # 2) Criar prompt hierárquico
    # --------------------------------------------------------
    prompt_mestre = montar_prompt_mestre(
        pergunta=pergunta,
        perfil=perfil,
        contexto=contexto,
        fontes=fontes
    )

    # --------------------------------------------------------
    # 3) Construir MCP consolidado
    # --------------------------------------------------------
    mcp = ConsultaContext(
        trace_id=None,
        perfil_cliente=perfil,
        pergunta_cliente=pergunta,
        contexto_juridico_bruto=contexto,
        fontes_detalhadas=fontes,
        prompt_mestre=prompt_mestre,
    )

    # --------------------------------------------------------
    # 4) Invocar LLM (system + human)
    # --------------------------------------------------------
    resposta = llm.invoke([
        SystemMessage(content=mcp.prompt_mestre),
        HumanMessage(content="Gere a resposta final seguindo rigorosamente as instruções acima.")
    ])

    # --------------------------------------------------------
    # 5) Atualizar histórico
    # --------------------------------------------------------
    historico.append(AIMessage(content=resposta.content))

    return {"messages": historico}