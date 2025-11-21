from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FonteDocumento(BaseModel):
    """Metadados de uma fatia de lei recuperada."""
    document_source: str = Field(..., description="Nome da fonte (EC 132, LC 214 ou Tavily Web Search).")
    page_number: int | None = Field(None, description="Número da página (se aplicável).")
    chunk_index: int | None = Field(None, description="Índice da fatia no documento original.")
    document_type: str = Field(..., description="Tipo de documento (LEI, EMENDA, WEB).")

class ConsultaContext(BaseModel):
    """
    Model Context Protocol (MCP) - Define o contexto completo para a resposta final.
    """
    # 1. IDENTIDADE E AUDITORIA
    trace_id: str | None = Field(None, description="ID de rastreamento do Langfuse/LangGraph.")
    perfil_cliente: str = Field(..., description="Perfil do cliente em JSON (CNAE, Regime).")
    pergunta_cliente: str = Field(..., description="A pergunta original do usuário.")
    
    # 2. CONHECIMENTO
    contexto_juridico_bruto: str = Field(..., description="O texto bruto das fatias (chunks) recuperadas.")
    fontes_detalhadas: List[FonteDocumento] = Field(..., description="Metadados das fontes usadas.")
    
    # 3. GOVERNANÇA
    prompt_mestre: str = Field(..., description="O system prompt usado na geração.")