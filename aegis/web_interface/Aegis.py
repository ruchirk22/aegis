# aegis/web_interface/Aegis.py
import streamlit as st

st.set_page_config(
    page_title="Aegis Home",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Welcome to Aegis")
st.markdown("### The Interactive LLM Red Teaming Framework")
st.sidebar.success("Select a tool from the sidebar to begin.")

st.info("Use the **Red Team Sandbox** to run live tests against various models.")