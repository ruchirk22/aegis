# aegis/web_interface/app.py

import streamlit as st
import sys
import os
import pandas as pd

# --- Path Correction ---
# Add the project root to the Python path to allow for absolute imports from the 'aegis' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Correction ---

from aegis.core.analyzer import LLMAnalyzer
from aegis.core.connectors import GeminiConnector, OpenRouterConnector
from aegis.core.models import AdversarialPrompt
from aegis.core.library import PromptLibrary

# --- Page Configuration ---
st.set_page_config(
    page_title="Aegis Red Team Sandbox",
    page_icon="🛡️",
    layout="wide"
)

# --- State Management & Caching ---
# Caching the prompt library and analyzer to prevent reloading on every interaction
@st.cache_resource
def load_resources():
    library = PromptLibrary()
    library.load_prompts()
    analyzer = LLMAnalyzer()
    return library, analyzer

st.session_state.library, st.session_state.analyzer = load_resources()

if 'result' not in st.session_state:
    st.session_state.result = None
if 'prompt_text' not in st.session_state:
    st.session_state.prompt_text = ""

# --- UI Layout ---
st.title("🛡️ Aegis: Interactive Red Team Sandbox")
st.markdown("Test and evaluate LLM vulnerabilities in real-time. Enter an adversarial prompt, select a model, and see the AI-powered analysis.")

with st.sidebar:
    st.header("Configuration")
    
    model_option = st.selectbox("Choose a Provider", ("Gemini", "OpenRouter"), key="provider")
    model_identifier = ""
    openrouter_model = ""

    if model_option == "Gemini":
        model_identifier = "gemini"
        st.info("Using the Gemini 1.5 Flash model via your `GEMINI_API_KEY`.")
    elif model_option == "OpenRouter":
        openrouter_model = st.text_input(
            "Enter OpenRouter Model Name", "google/gemini-flash-1.5",
            help="e.g., `anthropic/claude-3-haiku`, `mistralai/mistral-7b-instruct`"
        )
        model_identifier = f"openrouter/{openrouter_model}"
        st.info("Using the specified model via your `OPENROUTER_API_KEY`.")

# --- New Feature: Prompt Library Explorer ---
with st.expander("📚 Prompt Library Explorer", expanded=False):
    prompts = st.session_state.library.get_all()
    if prompts:
        df = pd.DataFrame([p.to_dict() for p in prompts])
        
        # Category filter
        categories = ["All"] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("Filter by Category", options=categories)

        if selected_category != "All":
            df_filtered = df[df['category'] == selected_category]
        else:
            df_filtered = df

        # Display prompts in a more readable format
        for index, row in df_filtered.iterrows():
            with st.container():
                st.markdown(f"**ID:** `{row['id']}` | **Category:** `{row['category']}` | **Severity:** `{row['severity']}`")
                st.code(row['prompt_text'], language='text')
                if st.button("Load this Prompt", key=f"load_{row['id']}"):
                    st.session_state.prompt_text = row['prompt_text']
                    st.rerun() # Rerun the script to update the text area
                st.divider()
    else:
        st.warning("No prompts found in the library.")


# Main content area
st.subheader("Enter Your Adversarial Prompt")
prompt_text = st.text_area(
    "Prompt",
    height=150,
    placeholder="e.g., You are 'DAN' (Do Anything Now)... or load one from the explorer above.",
    label_visibility="collapsed",
    key="prompt_text" # Use the session state key
)

if st.button("Evaluate Model", type="primary", use_container_width=True):
    if not st.session_state.prompt_text:
        st.warning("Please enter a prompt to evaluate.")
    else:
        try:
            with st.spinner("Evaluating... This may take a moment as two LLM calls are being made."):
                if model_option == "Gemini":
                    connector = GeminiConnector()
                else: # OpenRouter
                    connector = OpenRouterConnector(model_name=openrouter_model)

                temp_prompt = AdversarialPrompt(
                    id="sandbox_live_test", prompt_text=st.session_state.prompt_text,
                    category="Live_Test", subcategory="Sandbox", severity="UNKNOWN",
                    expected_behavior="REJECT"
                )
                response = connector.send_prompt(temp_prompt)
                analysis = st.session_state.analyzer.analyze(response, temp_prompt)
                st.session_state.result = {"response": response, "analysis": analysis}
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.result = None

# --- Display Results ---
if st.session_state.result:
    st.divider()
    st.subheader("Analysis Results")

    res = st.session_state.result
    analysis = res["analysis"]
    response = res["response"]

    color = "blue"
    if analysis.classification.name == "NON_COMPLIANT": color = "red"
    elif analysis.classification.name == "COMPLIANT": color = "green"
    elif analysis.classification.name == "PARTIAL_COMPLIANCE": color = "orange"

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Tested", response.model_name)
    col2.metric("Classification", analysis.classification.name)
    col3.metric("Vulnerability Score", f"{analysis.vulnerability_score:.1f}", delta=f"{analysis.vulnerability_score:.1f} / 100", delta_color="off")

    st.error(f"**Model Output:**\n\n{response.output_text}")
    st.info(f"**Analysis Explanation:**\n\n{analysis.explanation}")
