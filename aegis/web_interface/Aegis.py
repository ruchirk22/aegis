# aegis/web_interface/Aegis.py

import streamlit as st
import sys
import os
import pandas as pd
import json
from io import BytesIO
from datetime import datetime
import uuid

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Correction ---

from aegis.core.analyzer import LLMAnalyzer
from aegis.core.connectors import OpenRouterConnector, CustomEndpointConnector, UserProvidedGeminiConnector
from aegis.core.models import AdversarialPrompt
from aegis.core.library import PromptLibrary
from aegis.core.reporting import generate_pdf_report
from aegis.core.database.manager import DatabaseManager
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Aegis Framework", page_icon="🛡️", layout="wide")

# --- State Management & Caching ---
@st.cache_resource
def load_resources():
    library = PromptLibrary()
    library.load_prompts()
    analyzer = LLMAnalyzer()
    db_manager = DatabaseManager()
    return library, analyzer, db_manager

st.session_state.library, st.session_state.analyzer, st.session_state.db_manager = load_resources()

# --- Session State Initializations ---
if 'active_tab' not in st.session_state: st.session_state.active_tab = "single"
if 'single_result' not in st.session_state: st.session_state.single_result = None
if 'batch_results_df' not in st.session_state: st.session_state.batch_results_df = pd.DataFrame()
if 'raw_batch_results' not in st.session_state: st.session_state.raw_batch_results = []
if 'prompt_text' not in st.session_state: st.session_state.prompt_text = ""
if 'user_api_key_openrouter' not in st.session_state: st.session_state.user_api_key_openrouter = ""
if 'user_api_key_gemini' not in st.session_state: st.session_state.user_api_key_gemini = ""
if 'explorer_expanded' not in st.session_state: st.session_state.explorer_expanded = True
if 'is_processing' not in st.session_state: st.session_state.is_processing = False

# --- Helper Functions ---
def convert_result_to_flat_dict(result_data, session_id):
    """Converts the nested result object to a flat dictionary for saving."""
    return {
        "session_id": session_id,
        "prompt_id": result_data["prompt"].id,
        "category": result_data["prompt"].category,
        "prompt_text": result_data["prompt"].prompt_text,
        "model_name": result_data["response"].model_name,
        "model_output": result_data["response"].output_text,
        "classification": result_data["analysis"].classification.name,
        "vulnerability_score": result_data["analysis"].vulnerability_score,
        "explanation": result_data["analysis"].explanation
    }

def convert_results_to_df(results_list, session_id):
    """Converts a list of result objects to a pandas DataFrame."""
    if not results_list: return pd.DataFrame()
    flat_list = [convert_result_to_flat_dict(res, session_id) for res in results_list]
    return pd.DataFrame(flat_list)

def display_analysis_results(result_data):
    """Renders the analysis results in a standardized format."""
    analysis, response = result_data["analysis"], result_data["response"]
    color = "blue"
    if analysis.classification.name == "NON_COMPLIANT": color = "red"
    elif analysis.classification.name == "COMPLIANT": color = "green"
    elif analysis.classification.name == "PARTIAL_COMPLIANCE": color = "orange"
    col1, col2, col3 = st.columns(3)
    col1.metric("Model/Endpoint Tested", response.model_name)
    col2.metric("Classification", analysis.classification.name)
    col3.metric("Vulnerability Score", f"{analysis.vulnerability_score:.1f}", delta=f"{analysis.vulnerability_score:.1f} / 100", delta_color="off")
    st.error(f"**Model Output:**\n\n{response.output_text}")
    st.info(f"**Analysis Explanation:**\n\n{analysis.explanation}")

# --- UI Layout (Sidebar) ---
st.sidebar.title("🛡️ Aegis Framework")
st.title("Red Team Sandbox")
with st.sidebar:
    st.header("Configuration")
    provider_option = st.selectbox("Choose a Provider", ("Gemini", "OpenRouter", "Custom Endpoint"))
    connector = None
    if provider_option == "Gemini":
        st.info("Select a vision-compatible model like 'gemini-1.5-flash-latest' for multi-modal tests.")
        st.text_input("Enter your Google Gemini API Key", type="password", key="user_api_key_gemini")
        selected_model = st.text_input("Enter a Gemini Model Name", "gemini-1.5-flash-latest")
        if st.session_state.user_api_key_gemini and selected_model:
            connector = UserProvidedGeminiConnector(model_name=selected_model, api_key=st.session_state.user_api_key_gemini)
    elif provider_option == "OpenRouter":
        st.warning("Multi-modal support for OpenRouter is not yet implemented.")
        st.text_input("Enter your OpenRouter API Key", type="password", key="user_api_key_openrouter")
        models = ["openai/gpt-4o-mini", "google/gemma-2-9b-it:free", "anthropic/claude-3.5-sonnet", "Enter a custom model name..."]
        selected_model = st.selectbox("Select an OpenRouter Model", options=models)
        if selected_model == "Enter a custom model name...":
            selected_model = st.text_input("Enter Custom Model Name", "anthropic/claude-3-opus")
        if st.session_state.user_api_key_openrouter and selected_model:
            connector = OpenRouterConnector(model_name=selected_model, api_key=st.session_state.user_api_key_openrouter)
    elif provider_option == "Custom Endpoint":
        st.warning("Multi-modal support for Custom Endpoints is not yet implemented.")
        endpoint_url = st.text_input("Enter Endpoint URL", "http://localhost:8000/generate")
        headers_str = st.text_area("Enter Headers (JSON)", '{"Authorization": "Bearer YOUR_KEY"}')
        try:
            headers = json.loads(headers_str) if headers_str else {}
            if endpoint_url: connector = CustomEndpointConnector(endpoint_url=endpoint_url, headers=headers)
        except json.JSONDecodeError: st.error("Invalid JSON for headers.")

# --- Main App Logic (Tabs) ---
tab1, tab2 = st.tabs(["🧪 Single Prompt Evaluation", "🚀 Batch Evaluation"])

with tab1:
    with st.expander("📚 Prompt Library Explorer", expanded=st.session_state.explorer_expanded):
        prompts = st.session_state.library.get_all()
        if prompts:
            df = pd.DataFrame([p.to_dict() for p in prompts])
            categories = ["All"] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", options=categories, key="single_cat_filter")
            df_filtered = df[df['category'] == selected_category] if selected_category != "All" else df
            for index, row in df_filtered.iterrows():
                with st.container():
                    st.markdown(f"**ID:** `{row['id']}` | **Category:** `{row['category']}` | **Severity:** `{row['severity']}`")
                    st.code(row['prompt_text'], language='text')
                    if st.button("Load this Prompt", key=f"load_{row['id']}"):
                        st.session_state.prompt_text = row['prompt_text']
                        st.session_state.explorer_expanded = False
                        st.rerun()
                    st.divider()

    st.subheader("Enter Your Adversarial Prompt")
    
    uploaded_image = st.file_uploader(
        "Upload an Image (Optional)", 
        type=["png", "jpg", "jpeg", "webp"],
        disabled=st.session_state.is_processing
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Image to be analyzed", width=200)
    
    st.text_area("Text Prompt", height=150, label_visibility="collapsed", key="prompt_text", disabled=st.session_state.is_processing)
    
    if st.button("Evaluate Single Prompt", type="primary", use_container_width=True, disabled=st.session_state.is_processing):
        if not st.session_state.prompt_text: 
            st.warning("Please enter a text prompt.")
        elif connector is None: 
            st.warning("Provider configuration is incomplete or invalid.")
        elif provider_option != "Gemini" and uploaded_image:
            st.error("Image uploads are currently only supported for the Gemini provider.")
        else:
            st.session_state.is_processing = True
            st.session_state.single_result = None
            st.session_state.active_tab = "single"
            st.rerun()

    results_placeholder = st.empty()
    if st.session_state.is_processing and st.session_state.active_tab == "single":
        with results_placeholder.container():
            with st.spinner("Evaluating..."):
                session_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                
                image_bytes = uploaded_image.getvalue() if uploaded_image else None
                
                temp_prompt = AdversarialPrompt(
                    id="sandbox_live_test", 
                    prompt_text=st.session_state.prompt_text, 
                    category="Live_Test", 
                    subcategory="MultiModal" if image_bytes else "Text", 
                    severity="UNKNOWN", 
                    expected_behavior="REJECT",
                    image_data=image_bytes
                )
                
                response = connector.send_prompt(temp_prompt)
                analysis = st.session_state.analyzer.analyze(response, temp_prompt)
                result_data = {"prompt": temp_prompt, "response": response, "analysis": analysis}
                
                flat_result = convert_result_to_flat_dict(result_data, session_id)
                st.session_state.db_manager.insert_result(flat_result)
                st.toast("✅ Result saved to database!")
                
                st.session_state.single_result = result_data
                st.session_state.is_processing = False
                st.rerun()

    if st.session_state.single_result and not st.session_state.is_processing:
        with results_placeholder.container():
            st.divider()
            st.subheader("Analysis Results")
            display_analysis_results(st.session_state.single_result)

with tab2:
    st.info("Multi-modal evaluation is not yet supported in Batch mode.")
    st.subheader("Run Bulk Attacks")
    batch_source = st.radio("Select Prompt Source", ("From Library", "Custom Prompts"), horizontal=True, disabled=st.session_state.is_processing)
    prompts_to_run = []
    if batch_source == "From Library":
        df = pd.DataFrame([p.to_dict() for p in st.session_state.library.get_all()])
        categories = sorted(df['category'].unique().tolist())
        selected_batch_category = st.selectbox("Select a Prompt Category", options=categories, disabled=st.session_state.is_processing)
        prompts_to_run = st.session_state.library.filter_by_category(selected_batch_category)
    elif batch_source == "Custom Prompts":
        custom_prompts_text = st.text_area("Enter prompts (one per line)", height=250, disabled=st.session_state.is_processing)
        if custom_prompts_text:
            lines = [line.strip() for line in custom_prompts_text.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                prompts_to_run.append(AdversarialPrompt(id=f"custom_{i+1}", prompt_text=line, category="Custom_Batch", subcategory="Custom", severity="UNKNOWN", expected_behavior="REJECT"))
    
    if st.button("Run Batch Evaluation", type="primary", use_container_width=True, key="run_batch_eval", disabled=st.session_state.is_processing):
        if not prompts_to_run: st.warning("No prompts to evaluate.")
        elif connector is None: st.warning("Provider configuration is incomplete or invalid.")
        else:
            st.session_state.is_processing = True
            st.session_state.batch_results_df = pd.DataFrame()
            st.session_state.active_tab = "batch"
            st.rerun()

    batch_results_placeholder = st.empty()
    if st.session_state.is_processing and st.session_state.active_tab == "batch":
        with batch_results_placeholder.container():
            with st.spinner("Running batch evaluation..."):
                session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                
                st.session_state.raw_batch_results = []
                progress_bar = st.progress(0, text="Starting...")
                total_prompts = len(prompts_to_run)
                for i, prompt in enumerate(prompts_to_run):
                    response = connector.send_prompt(prompt)
                    analysis = st.session_state.analyzer.analyze(response, prompt)
                    result_data = {"prompt": prompt, "response": response, "analysis": analysis}
                    st.session_state.raw_batch_results.append(result_data)
                    
                    flat_result = convert_result_to_flat_dict(result_data, session_id)
                    st.session_state.db_manager.insert_result(flat_result)
                    
                    progress_bar.progress((i + 1) / total_prompts, text=f"Evaluated & Saved {prompt.id}...")
                
                st.toast(f"✅ Batch evaluation complete! {total_prompts} results saved to database.")
                st.session_state.batch_results_df = convert_results_to_df(st.session_state.raw_batch_results, session_id)
                st.session_state.is_processing = False
                st.rerun()

    if not st.session_state.batch_results_df.empty and not st.session_state.is_processing:
        with batch_results_placeholder.container():
            st.divider()
            st.subheader("Batch Results")
            st.dataframe(st.session_state.batch_results_df)
