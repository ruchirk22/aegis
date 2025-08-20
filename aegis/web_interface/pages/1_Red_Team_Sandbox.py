# aegis/web_interface/app.py

import streamlit as st
import sys
import os
import pandas as pd
import json
import csv
import io
from datetime import datetime

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aegis.core.analyzer import LLMAnalyzer
from aegis.core.connectors import OpenRouterConnector
from aegis.core.models import AdversarialPrompt, AnalysisResult, ModelResponse
from aegis.core.library import PromptLibrary
from aegis.core.reporting import generate_pdf_report
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Aegis Red Team Sandbox",
    page_icon="🛡️",
    layout="wide"
)

# --- Model List for Dropdown ---
OPENROUTER_MODELS = [
    "openai/gpt-oss-20b:free", "z-ai/glm-4.5-air:free", "qwen/qwen3-coder:free",
    "moonshotai/kimi-k2:free", "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3n-e2b-it:free", "tencent/hunyuan-a13b-instruct:free",
    "tngtech/deepseek-r1t2-chimera:free", "mistralai/mistral-small-3.2-24b-instruct:free",
    "moonshotai/kimi-dev-72b:free", "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "deepseek/deepseek-r1-0528:free", "sarvamai/sarvam-m:free",
    "mistralai/devstral-small-2505:free", "google/gemma-3n-e4b-it:free",
    "qwen/qwen3-4b:free", "qwen/qwen3-30b-a3b:free", "qwen/qwen3-8b:free",
    "qwen/qwen3-14b:free", "qwen/qwen3-235b-a22b:free", "tngtech/deepseek-r1t-chimera:free",
    "microsoft/mai-ds-r1:free", "shisa-ai/shisa-v2-llama3.3-70b:free",
    "arliai/qwq-32b-arliai-rpr-v1:free", "agentica-org/deepcoder-14b-preview:free",
    "moonshotai/kimi-vl-a3b-thinking:free", "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "qwen/qwen2.5-vl-32b-instruct:free", "deepseek/deepseek-chat-v3-0324:free",
    "featherless/qwerky-72b:free", "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-4b-it:free", "google/gemma-3-12b-it:free", "rekaai/reka-flash-3:free",
    "google/gemma-3-27b-it:free", "qwen/qwq-32b:free",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "cognitivecomputations/dolphin3.0-mistral-24b:free", "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen-2.5-coder-32b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free", "mistralai/mistral-nemo:free",
    "meta-llama/llama-3.1-405b-instruct:free"
]

# --- Helper Functions for Report Generation ---
def prepare_report_data(prompt: AdversarialPrompt, response: ModelResponse, analysis: AnalysisResult):
    """Prepares a list containing a single result dictionary for reporting."""
    return [{
        "prompt": prompt,
        "response": response,
        "analysis": analysis
    }]

def to_json(report_data):
    """Converts report data to a JSON string."""
    export_list = [{
        "prompt_id": res["prompt"].id, "category": res["prompt"].category,
        "prompt_text": res["prompt"].prompt_text, "model_name": res["response"].model_name,
        "model_output": res["response"].output_text, "classification": res["analysis"].classification.name,
        "vulnerability_score": res["analysis"].vulnerability_score, "explanation": res["analysis"].explanation,
    } for res in report_data]
    return json.dumps(export_list, indent=2)

def to_csv(report_data):
    """Converts report data to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    headers = ["prompt_id", "category", "prompt_text", "model_name", "model_output", "classification", "vulnerability_score", "explanation"]
    writer.writerow(headers)
    for res in report_data:
        writer.writerow([
            res["prompt"].id, res["prompt"].category, res["prompt"].prompt_text,
            res["response"].model_name, res["response"].output_text,
            res["analysis"].classification.name, res["analysis"].vulnerability_score,
            res["analysis"].explanation
        ])
    return output.getvalue()

def to_pdf_bytes(report_data):
    """Generates a PDF report and returns it as bytes."""
    classifications = [res["analysis"].classification.name for res in report_data]
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
    img_bytes = fig_bar.to_image(format="png")
    
    # Use a temporary file path to save the image for fpdf
    temp_image_path = "temp_chart.png"
    with open(temp_image_path, "wb") as f:
        f.write(img_bytes)

    # Generate the PDF in memory
    pdf_buffer = io.BytesIO()
    generate_pdf_report(report_data, pdf_buffer, temp_image_path)
    os.remove(temp_image_path)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# --- State Management & Caching ---
@st.cache_resource
def load_resources():
    library = PromptLibrary()
    library.load_prompts()
    analyzer = LLMAnalyzer()
    return library, analyzer

try:
    st.session_state.library, st.session_state.analyzer = load_resources()
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'prompt_text' not in st.session_state:
        st.session_state.prompt_text = ""
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = None
except ValueError as e:
    st.error(f"Fatal Error on Startup: {e}")
    st.warning("Please ensure your API keys are correctly configured and restart the app.")
    st.stop()

# --- UI Layout ---
st.title("🛡️ Aegis: Interactive Red Team Sandbox")
st.markdown("Test and evaluate LLM vulnerabilities in real-time. Enter an adversarial prompt, select a model, and see the AI-powered analysis.")

with st.sidebar:
    st.header("Red Team Sandbox")
    st.subheader("Configuration")
    
    is_key_configured = False
    try:
        # This block will work on Streamlit Cloud where secrets are configured.
        if 'OPENROUTER_API_KEY' in st.secrets and st.secrets['OPENROUTER_API_KEY']:
            st.success("OPENROUTER_API_KEY loaded.")
            is_key_configured = True
        else:
            # This will show if deployed but the secret is missing.
            st.error("OPENROUTER_API_KEY not found!")
            st.warning("Please add your OpenRouter key to your Streamlit secrets to proceed.")
    except st.errors.StreamlitAPIException:
        # This block catches the error when secrets aren't configured at all (local dev).
        st.error("Local secrets file not found!")
        st.warning("""
            To run locally, please create a `.streamlit/secrets.toml` file in your project root and add your key:
            ```toml
            OPENROUTER_API_KEY = "your-key-here"
            ```
        """)

    openrouter_model = st.selectbox(
        "Select OpenRouter Model",
        options=OPENROUTER_MODELS,
        index=0,
        help="Select a model from the list to test against."
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
                connector = OpenRouterConnector(model_name=openrouter_model)
                temp_prompt = AdversarialPrompt(
                    id="sandbox_live_test", prompt_text=st.session_state.prompt_text,
                    category="Live_Test", subcategory="Sandbox", severity="UNKNOWN",
                    expected_behavior="REJECT"
                )
                response = connector.send_prompt(temp_prompt)
                analysis = st.session_state.analyzer.analyze(response, temp_prompt)
                st.session_state.result = {"response": response, "analysis": analysis}
                st.session_state.last_prompt = temp_prompt
        except Exception as e:
            st.error(f"An unexpected error occurred during evaluation: {e}")
            st.session_state.result = None

# --- Display Results & Export Options ---
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
    
    st.divider()
    st.subheader("Export Report")
    
    report_data = prepare_report_data(st.session_state.last_prompt, response, analysis)
    
    export_cols = st.columns(3)
    with export_cols[0]:
        st.download_button(
            label="Export as JSON",
            data=to_json(report_data),
            file_name=f"aegis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with export_cols[1]:
        st.download_button(
            label="Export as CSV",
            data=to_csv(report_data),
            file_name=f"aegis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with export_cols[2]:
        with st.spinner('Generating PDF...'):
            pdf_bytes = to_pdf_bytes(report_data)
            st.download_button(
                label="Export as PDF",
                data=pdf_bytes,
                file_name=f"aegis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
