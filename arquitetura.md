# Arquitetura do Sistema Fiscal Inteligente

---

# 🧠 Visão Macro

O sistema é dividido em 6 camadas:
Interface (Streamlit)
Orquestração (LangGraph)
Recuperação Híbrida (RAG)
Web Search
Geração Final (LLM + MCP)
Auditoria (Langfuse)
Cada layer é desacoplada e testada isoladamente. --- # 🔗 LangGraph — Fluxo completo
Entrada ↓ router ↓ ┌───────────── RAG → node_rag_qdrant → node_generate_final │ └───────────── WEB → node_web_search → node_generate_final

--- # 🔍 Detalhamento do RAG 1. Embed query (OpenAI) 2. Qdrant top-12 3. CrossEncoder reranker top-6 4. LLM-as-Judge top-4 5. Consolidação de contexto 6. Normalização de fontes 7. MCP 8. Geração final --- # 🧠 MCP — Estrutura final
ConsultaContext: trace_id perfil_cliente pergunta_cliente contexto_juridico_bruto fontes_detalhadas[] prompt_mestre

--- # 📄 Prompts Hierárquicos (SOP) O prompt final é composto de:
system_base.txt tax_rules.txt contexto (+ perfil) fontes format_output.json

--- # 🧪 Testes Testes cobrem: - RAG (Qdrant mock + Rerankers) - Web Search - Router - Nodes - MCP - Prompts - LangGraph integration - Estado do app