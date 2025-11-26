from protocol import ConsultaContext, FonteDocumento


def test_fonte_documento():
    f = FonteDocumento(
        document_source="LC 214",
        document_type="LEI",
        chunk_index=2,
        page_number=10,
        url="https://example.com"
    )
    assert f.document_source == "LC 214"
    assert f.document_type == "LEI"


def test_mcp_valid():
    f = FonteDocumento(
        document_source="WEB",
        document_type="WEB",
    )

    ctx = ConsultaContext(
        trace_id=None,
        perfil_cliente="Simples Nacional",
        pergunta_cliente="Como fica o IBS?",
        contexto_juridico_bruto="Texto legal...",
        fontes_detalhadas=[f],
        prompt_mestre="PROMPT..."
    )

    assert ctx.perfil_cliente == "Simples Nacional"
    assert ctx.fontes_detalhadas[0].document_type == "WEB"