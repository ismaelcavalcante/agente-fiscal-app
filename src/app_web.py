import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
# st.set_page_config DEVE ser o primeiro comando Streamlit.
st.set_page_config(
    page_title="Agente Fiscal v3.0",
    page_icon="🤖",
    layout="wide"
)

# --- 2. INICIALIZAÇÃO DO "SESSION STATE" (A Memória do Chat) ---
# Nós garantimos que as variáveis de chat e conexão existam.

if "db_count" not in st.session_state:
    st.session_state.db_count = -1  # -1 = não verificado
if "messages" not in st.session_state:
    st.session_state.messages = [] # Lista para guardar o histórico da conversa
if "client_profile" not in st.session_state:
    # Perfil padrão
    st.session_state.client_profile = """{
"nome_empresa": "Construtora Alfa Ltda",
"cnae_principal": "4120-4/00 (Construção de Edifícios)",
"regime_tributario": "Simples Nacional",
"faturamento_anual": "R$ 3.000.000,00"
}"""

# --- Constantes ---
NOME_DA_COLECAO = "leis_fiscais_v1"
MODELO_EMBEDDING = st.secrets['MODELO_EMBEDDING']
OPENAI_MODEL=st.secrets['OPENAI_MODEL']

# --- 3. CARREGAR OS SERVIÇOS (CÉREBRO E EXECUTOR) ---
# Usamos @st.cache_resource para conectar apenas uma vez.

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
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        print("✅ Executor (OpenAI LLM) conectado.")

        # 3. Verificar contagem
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
        st.session_state.db_count = -2 
        return None, None
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        st.error(f"Erro fatal ao conectar aos serviços: {e}")
        st.session_state.db_count = -3
        return None, None

# Carrega os serviços
qdrant_client, llm_client = carregar_cerebro_e_executor()

# --- 4. INTERFACE DA BARRA LATERAL (SIDEBAR) ---

with st.sidebar:
    st.image("https://raw.githubusercontent.com/ismaelcavalcante/agente-fiscal-app/refs/heads/main/assets/logo.png", width=80)
    st.title("Perfil do Cliente")
    st.markdown("O Agente usará este perfil para todas as consultas.")
    
    # Caixa de texto para o perfil, usando o valor da session_state
    perfil_texto = st.text_area(
        "Edite o JSON do Perfil:",
        value=st.session_state.client_profile,
        height=250
    )
    
    # Botão para salvar o perfil na memória do chat
    if st.button("Salvar Perfil"):
        st.session_state.client_profile = perfil_texto
        st.success("Perfil salvo para esta sessão!")
    
    st.markdown("---")
    st.subheader("Base de Conhecimento")
    st.markdown(f"**Fatias no Cérebro:** `{st.session_state.get('db_count', 0)}`")
    st.markdown("EC 132 e LC 214")

# --- 5. INTERFACE PRINCIPAL DO CHAT ---

st.title("🤖 Agente Fiscal v3.0 (Chat)")
st.markdown("Faça perguntas sobre a Reforma Tributária (IBS, CBS, IS).")

# 5.1. Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5.2. Recebe a nova pergunta do usuário (no input na base da página)
if prompt := st.chat_input("Eu terei direito ao crédito de IBS?"):
    
    # Adiciona a mensagem do usuário ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Verifica se os serviços estão prontos
    if llm_client and qdrant_client and st.session_state.get('db_count', 0) > 0:
        
        # Prepara para a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Analisando leis e consultando o Agente..."):
                try:
                    # --- 6. LÓGICA RAG (A MESMA DE ANTES) ---
                    
                    # Usamos o perfil salvo no st.session_state
                    perfil_cliente_salvo = st.session_state.client_profile
                    
                    print("Iniciando consulta RAG...")
                    query_text = f"Perfil: {perfil_cliente_salvo}\nPergunta: {prompt}"
                    
                    # 1. Criar vetor da pergunta
                    embedding_response = llm_client.embeddings.create(
                        input=query_text,
                        model=MODELO_EMBEDDING
                    )
                    query_vector = embedding_response.data[0].embedding
                    
                    # 2. Buscar no Qdrant
                    print("Buscando no Qdrant...")
                    resultados = qdrant_client.search(
                        collection_name=NOME_DA_COLECAO,
                        query_vector=query_vector,
                        limit=7 # Pedimos 7 fatias de lei
                    )
                    
                    contexto_juridico = "\n---\n".join(
                        [hit.payload['texto'] for hit in resultados]
                    )
                    
                    # 3. Montar o Prompt para o LLM
                    PROMPT_MESTRE = """
                    Você é o "IA Fiscal Advisor", um consultor tributário Sênior.
                    Responda a pergunta do cliente com base *exclusivamente* no Perfil do Cliente e no Contexto Jurídico (fatias das leis) fornecido.
                    Seja direto, claro e cite os artigos ou seções do contexto que fundamentam sua resposta.
                    """
                    
                    prompt_usuario = f"""
                    **Perfil do Cliente:**
                    {perfil_cliente_salvo}

                    **Pergunta do Cliente:**
                    "{prompt}"

                    **Contexto Jurídico Recuperado da Base (Use APENAS isso):**
                    ---
                    {contexto_juridico}
                    ---

                    **Sua Resposta (seja direto e fundamente no contexto):**
                    """
                    
                    # 4. Chamar o LLM
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
                    
                    # Exibe a resposta e a salva no histórico
                    st.markdown(resposta_final)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_final})

                except Exception as e:
                    st.error(f"Erro durante a execução: {e}")
                    print(f"Erro no RAG: {e}")
    else:
        # Mensagem de erro se os serviços falharem
        st.error("Erro de conexão. Verifique as 'Secrets' e se o Cérebro (Qdrant) está populado.")