import streamlit as st
from content.de_content import show_de_content
from content.ca_content import show_ca_content  

# 🌐 Selector d'idioma
lang = st.sidebar.selectbox("🌐 Deutsch / Català", ["de", "cat"], index=0)

# 🌍 Mostrar contingut segons idioma
if lang == "de":
    show_de_content()
elif lang == "cat":
    show_ca_content()

# Força redeplegament Streamlit
