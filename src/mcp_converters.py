from typing import List, Dict
from protocol import FonteDocumento


def convert_sources(sources_raw: List[Dict]) -> List[FonteDocumento]:
    """
    Converte metadados soltos (RAG, WEB, regras fixas) no formato FonteDocumento.
    """
    fontes = []

    for src in sources_raw:
        fontes.append(
            FonteDocumento(
                document_source=src.get("source") or src.get("document_source") or "DESCONHECIDO",
                document_type=src.get("document_type", "RAG"),
                chunk_index=src.get("chunk_index"),
                page_number=src.get("page"),
                url=src.get("url"),
            )
        )

    return fontes