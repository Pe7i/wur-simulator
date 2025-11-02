import streamlit as st

st.set_page_config(page_title="WUR WEB UI", layout="wide")

st.title("WUR APP")
st.caption("Přesměrovávám na stránku Predikce…")

try:
    st.switch_page("pages/01_Predikce.py")
except Exception:
    st.warning("Automatické přesměrování není k dispozici ve vaší verzi Streamlitu.")
    st.page_link("pages/01_Predikce.py", label="→ Pokračovat na Predikce")
