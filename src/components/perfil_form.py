import json
import streamlit as st
from services.formatters import formatar_cnae, formatar_moeda
from services.cnae_api import buscar_cnae


def editar_perfil_form():

    with st.form("form_perfil"):
        nome = st.text_input("Nome da empresa")

        cnae_input = st.text_input("CNAE principal", help="Digite o CNAE, ex: 4120-4/00")
        cnae_formatado = formatar_cnae(cnae_input)

        if cnae_input:
            resultados = buscar_cnae(cnae_input)
            if resultados:
                st.write("Sugestões de CNAE:")
                for r in resultados:
                    st.write(f"- **{r['code']}** — {r['title']}")

        regime = st.selectbox(
            "Regime tributário",
            ["Simples Nacional", "Lucro Presumido", "Lucro Real"]
        )

        faturamento_input = st.text_input("Faturamento anual", help="Ex: R$ 1.234.567,89")
        faturamento_formatado = formatar_moeda(faturamento_input)

        submitted = st.form_submit_button("Salvar Perfil")

        if submitted:
            perfil = {
                "nome_empresa": nome,
                "cnae_principal": cnae_formatado,
                "regime_tributario": regime,
                "faturamento_anual": faturamento_formatado,
            }

            nome_perfil = nome or f"Perfil {len(st.session_state.perfis) + 1}"

            st.session_state.perfis[nome_perfil] = perfil
            st.session_state.perfil_ativo = nome_perfil

            st.success("Perfil salvo!")