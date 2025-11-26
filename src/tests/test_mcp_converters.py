from mcp_converters import convert_sources


def test_convert_sources():
    raw = [{
        "source": "WEB",
        "document_type": "WEB",
        "url": "https://foo"
    }]

    out = convert_sources(raw)

    assert len(out) == 1
    assert out[0].document_source == "WEB"
    assert out[0].url == "https://foo"