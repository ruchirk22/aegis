# vorak/web_interface/Home.py

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

from vorak.core.analyzer import LLMAnalyzer
from vorak.core.connectors import (
    OpenRouterConnector, CustomEndpointConnector, UserProvidedGeminiConnector,
    OpenAIConnector, AnthropicConnector
)
from vorak.core.models import AdversarialPrompt, EvaluationMode, Classification
from vorak.core.prompt_manager import PromptManager
from vorak.core.database.manager import DatabaseManager
from vorak.core.reporting import generate_pdf_report
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Vorak Red Team Sandbox", page_icon="🛡️", layout="wide")

# --- State Management & Caching ---
@st.cache_resource
def load_resources():
    """Loads resources that are shared across the session, running only once."""
    st.session_state.library = PromptManager()
    st.session_state.library.load_prompts()
    st.session_state.analyzer = LLMAnalyzer()
    st.session_state.db_manager = DatabaseManager()

# --- Session State Initializations ---
if 'library' not in st.session_state:
    load_resources()

if 'single_results' not in st.session_state: st.session_state.single_results = []
if 'batch_results' not in st.session_state: st.session_state.batch_results = []
if 'prompt_text' not in st.session_state: st.session_state.prompt_text = ""
if 'user_api_keys' not in st.session_state: st.session_state.user_api_keys = {}
if 'explorer_expanded' not in st.session_state: st.session_state.explorer_expanded = False
if 'is_processing' not in st.session_state: st.session_state.is_processing = False

# --- Helper Functions ---
def convert_result_to_flat_dict(result_data, session_id):
    """Converts a nested result dictionary into a flat dictionary for database insertion."""
    gov_data = result_data["analysis"].governance
    return {
        "session_id": session_id, "prompt_id": result_data["prompt"].id, "category": result_data["prompt"].category,
        "prompt_text": result_data["prompt"].prompt_text, "model_name": result_data["response"].model_name,
        "model_output": result_data["response"].output_text, "classification": result_data["analysis"].classification.name,
        "vulnerability_score": result_data["analysis"].vulnerability_score, "explanation": result_data["analysis"].explanation,
        "governance_nist": ", ".join(gov_data.nist_ai_rmf) if gov_data else "",
        "governance_eu": ", ".join(gov_data.eu_ai_act) if gov_data else "",
        "governance_iso": ", ".join(gov_data.iso_iec_23894) if gov_data else "",
    }

def convert_results_to_dataframe(results_list, session_id):
    """Converts a list of result dictionaries into a pandas DataFrame."""
    flat_list = [convert_result_to_flat_dict(res, session_id) for res in results_list]
    return pd.DataFrame(flat_list)

def generate_report_downloads(results, session_id, file_prefix):
    """Creates download buttons for CSV, JSON, and PDF reports."""
    df_results = convert_results_to_dataframe(results, session_id)
    
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", df_results.to_csv(index=False).encode('utf-8'), f"{file_prefix}_report.csv", "text/csv", use_container_width=True)
    c2.download_button("Download JSON", df_results.to_json(orient='records', indent=2).encode('utf-8'), f"{file_prefix}_report.json", "application/json", use_container_width=True)

    # PDF Generation
    try:
        classifications = [res["analysis"].classification.name for res in results]
        chart_image_buffer = BytesIO()
        classification_counts = pd.Series(classifications).value_counts()
        fig_bar = px.bar(
            classification_counts, x=classification_counts.index, y=classification_counts.values,
            labels={'x': 'Classification', 'y': 'Count'}, title="Classification Breakdown",
            color=classification_counts.index,
            color_discrete_map={
                'NON_COMPLIANT': 'red', 'COMPLIANT': 'green', 'PARTIAL_COMPLIANCE': 'orange',
                'AMBIGUOUS': 'grey', 'ERROR': 'black'
            }
        )
        fig_bar.write_image(chart_image_buffer, format='png', engine='kaleido')
        chart_image_buffer.seek(0)

        pdf_buffer = BytesIO()
        generate_pdf_report(results, pdf_buffer, chart_image_buffer)
        pdf_buffer.seek(0)
        
        c3.download_button("Download PDF", pdf_buffer, f"{file_prefix}_report.pdf", "application/pdf", use_container_width=True)
    except Exception as e:
        c3.warning(f"PDF generation failed. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")


def display_results(results, title_prefix=""):
    """Renders the analysis results in a professional format."""
    st.header("Evaluation Results")
    for i, result in enumerate(results):
        title = f"{title_prefix} (Turn {i+1})" if len(results) > 1 and "SCENARIO" in result["prompt"].id else title_prefix
        st.subheader(title)
        
        analysis, response = result["analysis"], result["response"]
        color_map = {"NON_COMPLIANT": "red", "COMPLIANT": "green", "PARTIAL_COMPLIANCE": "orange"}
        color = color_map.get(analysis.classification.name, "grey")
        
        st.markdown(f"**Prompt ID:** `{response.prompt_id}`")
        st.metric("Vulnerability Score", f"{analysis.vulnerability_score:.1f} / 100.0")
        st.markdown(f"**Classification:** <span style='color:{color}; font-weight:bold;'>{analysis.classification.name}</span>", unsafe_allow_html=True)
        
        with st.expander("Model Output", expanded=True):
            st.code(response.output_text, language=None)
        
        if analysis.governance:
            st.write("**Compliance Risks Identified:**")
            cols = st.columns(3)
            if analysis.governance.nist_ai_rmf: cols[0].expander("NIST AI RMF").write(analysis.governance.nist_ai_rmf)
            if analysis.governance.eu_ai_act: cols[1].expander("EU AI Act").write(analysis.governance.eu_ai_act)
            if analysis.governance.iso_iec_23894: cols[2].expander("ISO/IEC 23894").write(analysis.governance.iso_iec_23894)
        st.divider()

# --- UI Layout (Sidebar) ---
st.sidebar.title("Vorak Framework")
with st.sidebar:
    st.header("1. Configuration")
    provider_option = st.selectbox("Choose a Provider", ("Gemini", "OpenAI", "Claude (Anthropic)", "OpenRouter"))
    
    api_key = st.text_input(f"Enter {provider_option} API Key", type="password", key=f"api_key_input_{provider_option}",
                            value=st.session_state.user_api_keys.get(provider_option, ""))
    if api_key:
        st.session_state.user_api_keys[provider_option] = api_key

    connector = None
    model_name = ""
    if provider_option == "Gemini": model_name = st.text_input("Model Name", "gemini-1.5-flash-latest")
    elif provider_option == "OpenAI": model_name = st.text_input("Model Name", "gpt-4o-mini")
    elif provider_option == "Claude (Anthropic)": model_name = st.text_input("Model Name", "claude-3-5-sonnet-20240620")
    elif provider_option == "OpenRouter": model_name = st.text_input("Model Name", "google/gemma-2-9b-it:free")

    current_api_key = st.session_state.user_api_keys.get(provider_option)
    if current_api_key and model_name:
        try:
            if provider_option == "Gemini": connector = UserProvidedGeminiConnector(model_name=model_name, api_key=current_api_key)
            elif provider_option == "OpenAI": connector = OpenAIConnector(model_name=model_name, api_key=current_api_key)
            elif provider_option == "Claude (Anthropic)": connector = AnthropicConnector(model_name=model_name, api_key=current_api_key)
            elif provider_option == "OpenRouter": connector = OpenRouterConnector(model_name=model_name, api_key=current_api_key)
        except Exception as e:
            st.error(f"Failed to create connector: {e}")
            connector = None

    st.header("2. Evaluation Mode")
    st.session_state.mode = EvaluationMode(st.selectbox("Select Mode", [e.value for e in EvaluationMode if e not in [EvaluationMode.ATTACK_ONLY, EvaluationMode.ANALYSIS_ONLY]]))
    if st.session_state.mode == EvaluationMode.SCENARIO:
        st.session_state.turns = st.number_input("Number of Turns", min_value=1, max_value=10, value=3)

# --- Main App Logic ---
st.title("Red Team Sandbox")
tab1, tab2 = st.tabs(["Single Evaluation", "Batch Evaluation"])

with tab1:
    with st.expander("Prompt Library", expanded=st.session_state.explorer_expanded):
        df = pd.DataFrame([p.to_dict() for p in st.session_state.library.get_all()])
        for _, row in df.iterrows():
            col1, col2 = st.columns([4, 1])
            col1.code(row['prompt_text'], language=None)
            if col2.button("Load", key=f"load_{row['id']}", use_container_width=True):
                st.session_state.prompt_text = row['prompt_text']
                st.session_state.explorer_expanded = False
                st.rerun()

    st.text_area("Adversarial Prompt", height=150, key="prompt_text")
    if st.button("Run Evaluation", type="primary", use_container_width=True, disabled=st.session_state.is_processing or not connector):
        if not st.session_state.prompt_text: st.warning("Please enter a text prompt.")
        else:
            st.session_state.is_processing = True
            st.session_state.single_results = []
            st.rerun()

    if st.session_state.is_processing and not st.session_state.single_results:
        with st.spinner("Evaluating..."):
            session_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            initial_prompt = AdversarialPrompt(id="sandbox_live_test", prompt_text=st.session_state.prompt_text, category="Live_Test", subcategory="UI_Sandbox", severity="UNKNOWN", expected_behavior="REJECT")
            
            if st.session_state.mode == EvaluationMode.SCENARIO:
                history = []
                current_prompt = initial_prompt
                for i in range(st.session_state.turns):
                    response = connector.send_prompt(current_prompt, history)
                    analysis = st.session_state.analyzer.analyze(response, current_prompt)
                    st.session_state.single_results.append({"prompt": current_prompt, "response": response, "analysis": analysis})
                    if analysis.classification == Classification.NON_COMPLIANT: break
                    history.append({"role": "user", "content": current_prompt.prompt_text})
                    history.append({"role": "assistant", "content": response.output_text})
                    if i < st.session_state.turns - 1:
                        current_prompt = st.session_state.analyzer.generate_next_turn(history, initial_prompt, i + 2)
                        if not current_prompt: break
            else: # Standard, Adaptive, Governance
                response = connector.send_prompt(initial_prompt)
                analysis = st.session_state.analyzer.analyze(response, initial_prompt)
                st.session_state.single_results.append({"prompt": initial_prompt, "response": response, "analysis": analysis})
                if st.session_state.mode == EvaluationMode.ADAPTIVE and analysis.classification == Classification.COMPLIANT:
                    new_prompt = st.session_state.analyzer.run_adaptive_escalation(initial_prompt, response, 1)
                    if new_prompt:
                        response = connector.send_prompt(new_prompt)
                        analysis = st.session_state.analyzer.analyze(response, new_prompt)
                        st.session_state.single_results.append({"prompt": new_prompt, "response": response, "analysis": analysis})
            
            for res in st.session_state.single_results:
                st.session_state.db_manager.insert_result(convert_result_to_flat_dict(res, session_id))

            st.session_state.is_processing = False
            st.rerun()

    if st.session_state.single_results:
        display_results(st.session_state.single_results, "Single Evaluation")
        generate_report_downloads(st.session_state.single_results, "single_run", "single")

with tab2:
    st.subheader("Run Bulk Attacks")
    df = pd.DataFrame([p.to_dict() for p in st.session_state.library.get_all()])
    categories = sorted(df['category'].unique().tolist())
    selected_cat = st.selectbox("Select a Prompt Category", options=categories, key="batch_cat_select")
    prompts_to_run = st.session_state.library.filter_by_category(selected_cat)
    st.info(f"Selected category '{selected_cat}' contains {len(prompts_to_run)} prompts.")

    if st.button("Run Batch Evaluation", type="primary", use_container_width=True, disabled=st.session_state.is_processing or not connector):
        st.session_state.is_processing = True
        st.session_state.batch_results = []
        st.rerun()
    
    if st.session_state.is_processing and not st.session_state.batch_results:
        with st.spinner("Running batch evaluation..."):
            session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            progress_bar = st.progress(0, text="Starting...")
            for i, prompt in enumerate(prompts_to_run):
                response = connector.send_prompt(prompt)
                analysis = st.session_state.analyzer.analyze(response, prompt)
                result_data = {"prompt": prompt, "response": response, "analysis": analysis}
                st.session_state.batch_results.append(result_data)
                st.session_state.db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
                progress_bar.progress((i + 1) / len(prompts_to_run), text=f"Evaluated {prompt.id}")
        st.session_state.is_processing = False
        st.rerun()

    if st.session_state.batch_results:
        st.header("Batch Results")
        df_batch = convert_results_to_dataframe(st.session_state.batch_results, "batch_run")
        st.dataframe(df_batch)
        generate_report_downloads(st.session_state.batch_results, "batch_run", "batch")
