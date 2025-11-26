from prompt_hierarchy import montar_prompt_mestre
from protocol import FonteDocumento


def test_prompt_mestre():
    fontes = [FonteDocumento(document_source="LC 214", document_type="LEI")]
    prompt = montar_prompt_mestre("Pergunta", "Perfil", "Contexto", fontes)

    assert "Perfil" in prompt
    assert "Contexto" in prompt
    assert "LC 214" in prompt