# CHANGELOG
Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

---

## [1.1.0] — 2025-02-12
### Added
- Pipeline híbrido RAG (Qdrant + Reranking Vetorial + LLM-as-Judge)
- MCP (Model Context Protocol)
- Prompts hierárquicos SOP
- Web Search com normalização total
- Novo app_web com estado imutável
- Testes de integração do LangGraph
- Testes unitários do RAG, Web e MCP
- Conversores de fontes (RAG/Web → MCP)

### Changed
- Reescrita completa dos nodes do LangGraph
- Router juridicamente mais preciso
- Estrutura interna reorganizada
- Compatibilidade plena com Langfuse

### Fixed
- Problemas de mutação de histórico
- Cálculo incorreto de routes
- Execuções inconsistentes no Streamlit
- Falsos positivos do roteador ("lc", "reforma", etc.)

---

## [1.0.0] — 2025-01-20
### Added
- Versão inicial com Qdrant simples, Web Search e Streamlit básico.