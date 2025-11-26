from pydantic import BaseModel, Field
from typing import List, Optional, Union


class FonteDocumento(BaseModel):
    """
    Metadados padronizados para qualquer fonte utilizada pelo sistema.
    """
    document_source: str = Field(..., description="Nome da fonte (LC 214/2024, EC 132/2023, WEB, etc.)")
    document_type: str = Field(..., description="Tipo (LEI, WEB, RAG, FIXED_RULE).")
    chunk_index: Optional[int] = None
    page_number: Optional[int] = None
    url: Optional[str] = None


class ConsultaContext(BaseModel):
    """
    MCP — Model Context Protocol.
    Estrutura completa enviada ao gerador final.
    """

    # Governança / rastreamento
    trace_id: Optional[str] = Field(
        None, description="ID de rastreamento (LangGraph / Langfuse)"
    )

    # Identidade
    perfil_cliente: Union[str, dict] = Field(
        ..., description="Perfil do cliente (estrutura livre: CNAE, regime, UF etc.)"
    )
    pergunta_cliente: str = Field(..., description="Pergunta original do usuário.")

    # Conhecimento
    contexto_juridico_bruto: str = Field(
        ...,
        description="Texto bruto consolidado do RAG, WebSearch ou Regras Fixas."
    )
    fontes_detalhadas: List[FonteDocumento] = Field(
        ..., description="Lista padronizada de metadados das fontes utilizadas."
    )

    # Governança de prompt
    prompt_mestre: str = Field(
        ..., description="Prompt final usado para gerar a resposta."
    )