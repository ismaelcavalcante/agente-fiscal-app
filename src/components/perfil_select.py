import streamlit as st


def selecionar_perfil():
    st.subheader("Selecionar Perfil")

    perfis = st.session_state.perfis

    if perfis:
        perfil_escolhido = st.selectbox(
            "Perfis cadastrados:",
            list(perfis.keys()),
            index=list(perfis.keys()).index(st.session_state.perfil_ativo)
            if st.session_state.perfil_ativo else 0
        )
        st.session_state.perfil_ativo = perfil_escolhido

        if st.button("🗑 Remover perfil selecionado"):
            del st.session_state.perfis[perfil_escolhido]
            st.session_state.perfil_ativo = None
            st.rerun()

    else:
        st.info("Nenhum perfil encontrado. Crie um novo perfil.")