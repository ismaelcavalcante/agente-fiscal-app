import pytest
from rag.web import WebSearch


class MockTool:
    """
    Mock do Tavily tool.
    """
    def __init__(self, results):
        self._results = results

    def invoke(self, params):
        assert isinstance(params, dict)
        assert "query" in params
        return self._results


def test_web_search_success(monkeypatch):
    results = [
        {
            "url": "https://example.com",
            "snippet": "Um snippet útil.",
            "content": "Conteúdo completo da página.",
        }
    ]

    ws = WebSearch(api_key="dummy")
    ws.tool = MockTool(results)

    output = ws.execute("ICMS sobre energia")

    assert "answer" in output
    assert "sources" in output
    assert "example.com" in output["answer"]
    assert len(output["sources"]) == 1


def test_web_search_empty(monkeypatch):
    ws = WebSearch(api_key="dummy")
    ws.tool = MockTool([])

    output = ws.execute("test")
    assert output["answer"] == ""
    assert output["sources"] == []


def test_web_search_exception(monkeypatch):
    class ErrTool:
        def invoke(self, x):
            raise RuntimeError("fail")

    ws = WebSearch(api_key="dummy")
    ws.tool = ErrTool()

    output = ws.execute("x")
    assert output["answer"] == ""
    assert output["sources"] == []


def test_web_search_limits(monkeypatch):
    long_snippet = "s" * 10000
    long_content = "c" * 10000

    ws = WebSearch(api_key="dummy")
    ws.tool = MockTool([{
        "url": "x",
        "snippet": long_snippet,
        "content": long_content
    }])

    output = ws.execute("x")
    assert len(output["answer"]) < 2000
    assert len(output["sources"]) == 1