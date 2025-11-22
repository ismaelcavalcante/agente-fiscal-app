from pydantic import BaseModel, Field
from typing import List, Optional


class FonteDocumento(BaseModel):
    """
    Metadados de cada fonte usada no contexto jurídico final.
    """
    document_source: str = Field(..., description="Nome da fonte (EC 132, LC 214, Web Search, etc.)")
    page_number: Optional[int] = Field(None, description="Número da página quando aplicável.")
    chunk_index: Optional[int] = Field(None, description="Índice/número do chunk no documento.")
    document_type: str = Field(..., description="Tipo de documento (LEI, EMENDA, WEB).")


class ConsultaContext(BaseModel):
    """
    Model Context Protocol (MCP) – encapsula todo o contexto usado para gerar a resposta final.
    """

    # IDENTIDADE E GOVERNANÇA
    trace_id: Optional[str] = Field(
        None,
        description="ID de rastreamento do LangGraph/Langfuse (útil para auditoria)."
    )

    perfil_cliente: str = Field(
        ...,
        description="Perfil do cliente. Pode incluir regime, CNAE, porte, UF etc."
    )

    pergunta_cliente: str = Field(
        ...,
        description="Pergunta original feita pelo usuário."
    )

    # CONHECIMENTO
    contexto_juridico_bruto: str = Field(
        ...,
        description="Texto jurídico bruto recuperado pelo RAG (Qdrant/Web/Regras Fixas)."
    )

    fontes_detalhadas: List[FonteDocumento] = Field(
        ...,
        description="Lista de metadados das fontes utilizadas."
    )

    # GOVERNANÇA DO PROMPT
    prompt_mestre: str = Field(
        ...,
        description="O system prompt final usado para gerar a resposta."
    )