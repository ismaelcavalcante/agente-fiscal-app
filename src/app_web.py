import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient

# --- 0. Constantes ---
NOME_DA_COLECAO = "leis_fiscais_v1"
MODELO_EMBEDDING = st.secrets["MODELO_EMBEDDING"]
OPENAI_MODEL=st.secrets["OPENAI_MODEL"]


# --- 1. CONFIGURAÇÃO (AGORA COM CACHE) ---

@st.cache_resource
def carregar_cerebro_e_executor():
    """Conecta ao Qdrant e ao LLM da OpenAI."""
    print("Conectando aos serviços...")
    try:
        # Conecta ao Qdrant (serviço 'qdrant' no docker-compose)
        qdrant_client = QdrantClient(
            url=st.secrets["QDRANT_URL"], 
            api_key=st.secrets["QDRANT_API_KEY"]
        )
        print("✅ Cérebro (Qdrant) carregado.")
        
        # Conecta ao LLM
        llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("✅ Executor (OpenAI LLM) conectado.")
        
        # Verifica se a coleção existe
        try:
            count = qdrant_client.count(collection_name=NOME_DA_COLECAO, exact=True)
            st.session_state.db_count = count.count
        except Exception as e:
            st.session_state.db_count = 0
            st.error(f"Coleção '{NOME_DA_COLECAO}' não encontrada no Qdrant! Você já rodou o 'processador.py'?")
            print(e)

        return qdrant_client, llm_client
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        st.error(f"Erro fatal ao conectar aos serviços: {e}")
        return None, None

# Carrega os serviços
qdrant_client, llm_client = carregar_cerebro_e_executor()


# --- 2. INTERFACE WEB (STREAMLIT) ---

st.set_page_config(layout="wide")
st.title("🤖 Agente Fiscal v2.0 (Qdrant Engine)")
st.markdown(f"Alimentado com a **EC 132** e **LC 214**. Fatias no Cérebro: **{st.session_state.get('db_count', 0)}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Perfil do Cliente")
    perfil_cliente = st.text_area(
        "Descreva a empresa:",
        height=150,
        value="""{
"nome_empresa": "Construtora Alfa Ltda",
"cnae_principal": "4120-4/00 (Construção de Edifícios)",
"regime_tributario": "Simples Nacional",
"faturamento_anual": "R$ 3.000.000,00"
}"""
    )
    
    st.subheader("2. Pergunta Específica")
    pergunta_cliente = st.text_input(
        "Faça sua pergunta:",
        value="Eu terei direito ao crédito de IBS e CBS?"
    )

    run_button = st.button("Executar Análise", type="primary")

with col2:
    st.subheader("3. Resposta do Agente")
    
    if run_button and llm_client and qdrant_client and st.session_state.db_count > 0:
        with st.spinner("Analisando... (Isso pode levar até 30 segundos)"):
            try:
                # --- 3. LÓGICA RAG (QDRANT) ---
                
                # ETAPA DE RECUPERAÇÃO (RAG)
                print("Iniciando consulta RAG...")
                query_text = f"Perfil: {perfil_cliente}\nPergunta: {pergunta_cliente}"
                
                # 1. Criar o vetor da *pergunta* do usuário
                print("Criando vetor para a pergunta...")
                embedding_response = llm_client.embeddings.create(
                    input=query_text,
                    model=MODELO_EMBEDDING
                )
                query_vector = embedding_response.data[0].embedding
                
                # 2. Buscar no Qdrant pelo vetor mais próximo
                print("Buscando no Qdrant...")
                resultados = qdrant_client.search(
                    collection_name=NOME_DA_COLECAO,
                    query_vector=query_vector,
                    limit=7  # Pede os 7 chunks mais relevantes
                )
                
                # O Qdrant retorna os 'payloads' (metadados)
                contexto_juridico = "\n---\n".join(
                    [hit.payload['texto'] for hit in resultados]
                )
                print(f"Contexto RAG recuperado ({len(resultados)} fatias).")

                # ETAPA DE GERAÇÃO (LLM) - (Idêntica a antes)
                PROMPT_MESTRE = """
                Você é o "IA Fiscal Advisor", um consultor tributário Sênior.
                Responda a pergunta do cliente com base *exclusivamente* no Perfil do Cliente e no Contexto Jurídico (fatias das leis) fornecido.
                Seja direto, claro e cite os artigos ou seções do contexto que fundamentam sua resposta.
                """
                
                prompt_usuario = f"""
                **Perfil do Cliente:**
                {perfil_cliente}

                **Pergunta do Cliente:**
                "{pergunta_cliente}"

                **Contexto Jurídico Recuperado da Base (Use APENAS isso):**
                ---
                {contexto_juridico}
                ---

                **Sua Resposta (seja direto e fundamente no contexto):**
                """
                
                print("Enviando para o LLM...")
                completion = llm_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": PROMPT_MESTRE},
                        {"role": "user", "content": prompt_usuario}
                    ]
                )
                
                resposta_final = completion.choices[0].message.content
                st.markdown(resposta_final)
                
                with st.expander("Ver fontes (payloads) usadas pelo RAG"):
                    st.json([hit.payload for hit in resultados])

            except Exception as e:
                st.error(f"Erro durante a execução: {e}")
    elif st.session_state.db_count == 0:
        st.error("O Cérebro (Qdrant) está vazio. Rode o `processador.py` primeiro!")
    else:
        st.info("Preencha o perfil e a pergunta, depois clique em 'Executar Análise'.")