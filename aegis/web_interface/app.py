# aegis/web_interface/app.py

import streamlit as st
import sys
import os
import pandas as pd

# --- Path Correction ---
# Add the project root to the Python path to allow for absolute imports from the 'aegis' package.
# This is crucial for running the Streamlit app from its subdirectory.
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
# Caching the prompt library and analyzer to prevent reloading on every interaction.
# The updated connectors are now deployment-aware (checking st.secrets).
@st.cache_resource
def load_resources():
    """Loads and caches the prompt library and LLM analyzer."""
    library = PromptLibrary()
    library.load_prompts()
    analyzer = LLMAnalyzer()
    return library, analyzer

# Initialize session state variables
try:
    st.session_state.library, st.session_state.analyzer = load_resources()
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'prompt_text' not in st.session_state:
        st.session_state.prompt_text = ""
except ValueError as e:
    # This will catch initialization errors from connectors if keys are missing
    st.error(f"Fatal Error on Startup: {e}")
    st.warning("Please ensure your API keys are correctly configured in Streamlit Secrets and restart the app.")
    st.stop()


# --- UI Layout ---
st.title("🛡️ Aegis: Interactive Red Team Sandbox")
st.markdown("Test and evaluate LLM vulnerabilities in real-time. Enter an adversarial prompt, select a model, and see the AI-powered analysis.")

with st.sidebar:
    st.header("Configuration")
    
    model_option = st.selectbox("Choose a Provider", ("Gemini", "OpenRouter"), key="provider")
    
    model_identifier = ""
    openrouter_model = ""
    is_key_configured = False

    # --- API Key Status Check ---
    # Proactively check for API keys in secrets and inform the user.
    if model_option == "Gemini":
        model_identifier = "gemini"
        if 'GEMINI_API_KEY' in st.secrets and st.secrets['GEMINI_API_KEY']:
            st.success("GEMINI_API_KEY loaded successfully.")
            is_key_configured = True
        else:
            st.error("GEMINI_API_KEY not found!")
            st.warning("Please add your Gemini API key to your Streamlit secrets to proceed.")

    elif model_option == "OpenRouter":
        if 'OPENROUTER_API_KEY' in st.secrets and st.secrets['OPENROUTER_API_KEY']:
            st.success("OPENROUTER_API_KEY loaded successfully.")
            is_key_configured = True
        else:
            st.error("OPENROUTER_API_KEY not found!")
            st.warning("Please add your OpenRouter key to your Streamlit secrets to proceed.")
        
        openrouter_model = st.text_input(
            "Enter OpenRouter Model Name", "google/gemini-flash-1.5",
            help="e.g., `anthropic/claude-3-haiku`, `mistralai/mistral-7b-instruct`"
        )
        model_identifier = f"openrouter/{openrouter_model}"

# --- Prompt Library Explorer ---
with st.expander("📚 Prompt Library Explorer", expanded=False):
    prompts = st.session_state.library.get_all()
    if prompts:
        df = pd.DataFrame([p.to_dict() for p in prompts])
        
        categories = ["All"] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("Filter by Category", options=categories)

        df_filtered = df[df['category'] == selected_category] if selected_category != "All" else df

        for index, row in df_filtered.iterrows():
            with st.container():
                st.markdown(f"**ID:** `{row['id']}` | **Category:** `{row['category']}` | **Severity:** `{row['severity']}`")
                st.code(row['prompt_text'], language='text')
                if st.button("Load this Prompt", key=f"load_{row['id']}"):
                    st.session_state.prompt_text = row['prompt_text']
                    st.rerun()
                st.divider()
    else:
        st.warning("No prompts found in the library.")

# --- Main Interaction Area ---
st.subheader("Enter Your Adversarial Prompt")
prompt_text = st.text_area(
    "Prompt",
    height=150,
    placeholder="e.g., You are 'DAN' (Do Anything Now)... or load one from the explorer above.",
    label_visibility="collapsed",
    key="prompt_text"
)

if st.button("Evaluate Model", type="primary", use_container_width=True, disabled=not is_key_configured):
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
            st.error(f"An unexpected error occurred during evaluation: {e}")
            st.session_state.result = None

# --- Display Results ---
if st.session_state.result:
    st.divider()
    st.subheader("Analysis Results")

    res = st.session_state.result
    analysis = res["analysis"]
    response = res["response"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Tested", response.model_name)
    col2.metric("Classification", analysis.classification.name)
    col3.metric("Vulnerability Score", f"{analysis.vulnerability_score:.1f}", delta=f"{analysis.vulnerability_score:.1f} / 100", delta_color="off")

    # Display model output and analysis in styled boxes
    st.markdown(f"""
    <div style="border: 1px solid #ff4b4b; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
        <p style="font-weight: bold;">Model Output:</p>
        <pre><code>{response.output_text}</code></pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="border: 1px solid #3dd56d; border-radius: 5px; padding: 10px;">
        <p style="font-weight: bold;">Analysis Explanation:</p>
        <p>{analysis.explanation}</p>
    </div>
    """, unsafe_allow_html=True)
