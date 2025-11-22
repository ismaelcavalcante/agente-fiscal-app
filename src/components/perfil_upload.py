import json
import streamlit as st


def upload_perfil_json():

    arquivo = st.file_uploader("Enviar arquivo JSON", type="json")

    if arquivo:
        try:
            dados = json.loads(arquivo.read().decode("utf-8"))
            nome = dados.get("nome_empresa", f"Perfil {len(st.session_state.perfis)+1}")

            st.session_state.perfis[nome] = dados
            st.session_state.perfil_ativo = nome

            st.success(f"Perfil '{nome}' carregado com sucesso!")

        except Exception as e:
            st.error("Erro ao processar o arquivo JSON.")