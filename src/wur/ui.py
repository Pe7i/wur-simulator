import os
import streamlit as st

def sidebar_logo(logo_candidates=None) -> None:
    with st.sidebar:
        st.markdown("---")
        paths = logo_candidates or ["src/FIS_logo.png", "FIS_logo.png"]
        logo = next((p for p in paths if os.path.exists(p)), None)
        if logo:
            st.image(logo, use_container_width=True)
        else:
            st.caption("Logo nebylo nalezeno (chybí zdroj `src/FIS_logo.png`).")
        st.markdown("---")
    
