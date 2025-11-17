import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient

# --- 0. GARANTIR A INICIALIZAÇÃO DO SESSION STATE ---
# Esta é a correção principal. Nós garantimos que 'db_count'
# sempre exista, mesmo antes de tentar a conexão.
if "db_count" not in st.session_state:
    st.session_state.db_count = -1  # Usamos -1 para significar "não verificado"
# ----------------------------------------------------

# --- Constantes ---
NOME_DA_COLECAO = "leis_fiscais_v1"
MODELO_EMBEDDING = st.secrets['MODELO_EMBEDDING']
OPENAI_MODEL=st.secrets['OPENAI_MODEL']


# --- 1. CONFIGURAÇÃO (LENDO st.secrets) ---

@st.cache_resource
def carregar_cerebro_e_executor():
    """Conecta ao Qdrant e ao LLM da OpenAI usando st.secrets."""
    print("Conectando aos serviços...")
    try:
        # 1. Conectar ao Qdrant
        qdrant_client = QdrantClient(
            url=st.secrets['QDRANT_URL'], 
            api_key=st.secrets['QDRANT_API_KEY']
        )
        print("✅ Cérebro (Qdrant Cloud) carregado.")

        # 2. Conectar ao LLM
        llm_client = OpenAI(
            api_key=st.secrets['OPENAI_API_KEY']
        )
        print("✅ Executor (OpenAI LLM) conectado.")

        # 3. Verificar contagem (só se as conexões funcionarem)
        try:
            count = qdrant_client.count(collection_name=NOME_DA_COLECAO, exact=True)
            st.session_state.db_count = count.count
        except Exception as e:
            st.session_state.db_count = 0 # DB conectado, mas coleção vazia
            st.error(f"Coleção '{NOME_DA_COLECAO}' não encontrada no Qdrant! Você 'encheu' o Cérebro na Nuvem?")
            print(f"Erro Qdrant: {e}")

        return qdrant_client, llm_client
    
    except KeyError as e:
        st.error(f"Erro: A 'Secret' {e} não foi definida no painel do Streamlit Cloud!")
        st.session_state.db_count = -2 # Código de erro: Secret faltando
        return None, None
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        st.error(f"Erro fatal ao conectar aos serviços: {e}")
        st.session_state.db_count = -3 # Código de erro: Outra falha de conexão
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
    
    # --- LÓGICA DE EXIBIÇÃO CORRIGIDA ---
    # Nós reestruturamos este bloco 'if/else' para ser mais claro
    # e para usar o método .get() que é mais seguro.
    
    db_count = st.session_state.get('db_count', -1) # Pega o valor com segurança

    if run_button:
        # O botão FOI clicado
        if llm_client and qdrant_client and db_count > 0:
            with st.spinner("Analisando... (Isso pode levar até 30 segundos)"):
                try:
                    # --- 3. LÓGICA RAG (QDRANT) ---
                    print("Iniciando consulta RAG...")
                    query_text = f"Perfil: {perfil_cliente}\nPergunta: {pergunta_cliente}"
                    
                    print("Criando vetor para a pergunta...")
                    embedding_response = llm_client.embeddings.create(
                        input=query_text,
                        model=MODELO_EMBEDDING
                    )
                    query_vector = embedding_response.data[0].embedding
                    
                    print("Buscando no Qdrant...")
                    resultados = qdrant_client.search(
                        collection_name=NOME_DA_COLECAO,
                        query_vector=query_vector,
                        limit=7
                    )
                    
                    contexto_juridico = "\n---\n".join(
                        [hit.payload['texto'] for hit in resultados]
                    )
                    print(f"Contexto RAG recuperado ({len(resultados)} fatias).")

                    # --- ETAPA DE GERAÇÃO (LLM) ---
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
        
        elif db_count == 0:
            st.error("O Cérebro (Qdrant) está vazio. Verifique se o 'processador.py' foi executado corretamente.")
        else:
            # Se chegou aqui, é porque llm_client ou qdrant_client falharam na conexão
            st.error("Erro de conexão. Verifique suas 'Secrets' no painel do Streamlit Cloud e recarregue a página.")
    
    else:
        # O botão NÃO foi clicado. A página está apenas esperando.
        st.info("Preencha o perfil e a pergunta, depois clique em 'Executar Análise'.")