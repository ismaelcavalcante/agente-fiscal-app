# graph/nodes.py

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.logs import logger

from protocol import ConsultaContext
from mcp_converters import convert_sources
from prompts.hierarchy import montar_prompt_mestre

from rag.pipeline import HybridRAGPipeline
from rag.web import WebSearch


def node_rag_qdrant(state, retriever: HybridRAGPipeline):
    """
    Executa o pipeline RAG híbrido completo:
    Qdrant → Reranking Vetorial → LLM‑as‑Judge.
    """
    pergunta = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", "")

    try:
        fontes, contexto = retriever.run(pergunta, perfil)
        return {
            "contexto_juridico_bruto": contexto or "",
            "sources_data": fontes or [],
            "rag_ok": bool(contexto.strip()),
        }
    except Exception as e:
        logger.error(f"[NODE_RAG] Erro interno: {e}")
        return {
            "contexto_juridico_bruto": "",
            "sources_data": [],
            "rag_ok": False,
        }


def node_web_search(state, web_tool: WebSearch):
    """
    WebSearch como fallback (casos que o Router indica WEB ou quando o RAG falha).
    """
    if state.get("rag_ok"):
        return {}

    result = web_tool.execute(state.get("ultima_pergunta", ""))

    return {
        "contexto_juridico_bruto": result.get("answer", ""),
        "sources_data": result.get("sources", []),
        "rag_ok": bool(result.get("answer", "").strip()),
    }


def node_generate_final(state, llm):
    """
    Monta o MCP, aplica o Prompt Hierárquico SOP e gera a resposta final.
    """
    pergunta = state.get("ultima_pergunta", "")
    perfil = state.get("perfil_cliente", "")
    contexto = state.get("contexto_juridico_bruto", "")
    fontes_raw = state.get("sources_data", [])
    historico = list(state.get("messages", []))

    fontes = convert_sources(fontes_raw)
    prompt_mestre = montar_prompt_mestre(pergunta, perfil, contexto, fontes)

    mcp = ConsultaContext(
        trace_id=None,
        perfil_cliente=perfil,
        pergunta_cliente=pergunta,
        contexto_juridico_bruto=contexto,
        fontes_detalhadas=fontes,
        prompt_mestre=prompt_mestre,
    )

    resposta = llm.invoke([
        SystemMessage(content=mcp.prompt_mestre),
        HumanMessage(content="Gere a resposta final seguindo estritamente as instruções.")
    ])

    historico.append(AIMessage(content=resposta.content))

    return {"messages": historico}