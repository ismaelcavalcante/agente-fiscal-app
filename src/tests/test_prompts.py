from prompts.hierarchy import montar_prompt_mestre
from protocol import FonteDocumento


def test_prompt_contains_profile_question_context():
    fontes = [
        FonteDocumento(document_source="LC 214", document_type="LEI"),
        FonteDocumento(document_source="WEB", document_type="WEB")
    ]

    prompt = montar_prompt_mestre(
        pergunta="Como fica o IBS?",
        perfil="Simples Nacional",
        contexto="Texto jurídico sobre IBS.",
        fontes=fontes
    )

    assert "Simples Nacional" in prompt
    assert "Como fica o IBS?" in prompt
    assert "Texto jurídico sobre IBS." in prompt
    assert "LC 214" in prompt
    assert "WEB" in prompt


def test_prompt_has_structure():
    prompt = montar_prompt_mestre("X", "Y", "Z", [])
    assert "# PAPEL" in prompt
    assert "# OBJETIVO" in prompt
    assert "# FORMAT"[:7] in prompt.upper()