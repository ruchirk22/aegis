# aegis/web_interface/pages/1_📊_Security_Dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from glob import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="Aegis Security Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Security Dashboard")
st.markdown("Visualize and analyze batch evaluation results from your saved reports.")

# --- Data Loading ---
@st.cache_data
def load_report_data(report_path):
    """Loads and caches data from a single JSON report file."""
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Ensure 'classification' column exists and is of type string
        if 'classification' in df.columns:
            df['classification'] = df['classification'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading {os.path.basename(report_path)}: {e}")
        return pd.DataFrame()

def find_reports(search_dir="."):
    """Finds all .json files in the specified directory."""
    return glob(os.path.join(search_dir, "*.json"))

# --- Main UI ---
report_files = find_reports()

if not report_files:
    st.warning("No JSON report files found in the root directory.", icon="⚠️")
    st.info("Run a batch evaluation from the command line with the `--output-json` flag to generate a report.")
else:
    selected_report = st.selectbox("Select a Report File to Analyze", options=report_files)
    
    if selected_report:
        df = load_report_data(selected_report)

        if not df.empty:
            st.header("Vulnerability Analysis")
            
            # --- New Feature: Add filters for model ---
            st.sidebar.header("Dashboard Filters")
            model_options = ["All Models"] + sorted(df['model_name'].unique().tolist())
            selected_model = st.sidebar.selectbox("Filter by Model", options=model_options)

            if selected_model != "All Models":
                df_filtered = df[df['model_name'] == selected_model].copy()
            else:
                df_filtered = df.copy()

            col1, col2 = st.columns(2)

            with col1:
                # --- Vulnerability Heatmap (PRD 4.3.1) ---
                st.subheader("Vulnerability Heatmap")
                if not df_filtered.empty:
                    heatmap_data = df_filtered.pivot_table(
                        index='category', 
                        columns='model_name', 
                        values='vulnerability_score',
                        aggfunc='mean'
                    ).fillna(0)
                    
                    fig_heatmap = px.imshow(
                        heatmap_data, text_auto=".1f", aspect="auto",
                        color_continuous_scale='Reds',
                        labels=dict(x="Model Name", y="Attack Category", color="Avg Score")
                    )
                    fig_heatmap.update_layout(title="Average Vulnerability Score")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No data available for the selected filter.")

            with col2:
                # --- Classification Breakdown Chart ---
                st.subheader("Classification Breakdown")
                if not df_filtered.empty:
                    classification_counts = df_filtered['classification'].value_counts()
                    fig_bar = px.bar(
                        classification_counts,
                        x=classification_counts.index,
                        y=classification_counts.values,
                        labels={'x': 'Classification', 'y': 'Count'},
                        color=classification_counts.index,
                        color_discrete_map={
                            'NON_COMPLIANT': 'red',
                            'COMPLIANT': 'green',
                            'PARTIAL_COMPLIANCE': 'orange',
                            'AMBIGUOUS': 'grey',
                            'ERROR': 'black'
                        }
                    )
                    fig_bar.update_layout(title="Evaluation Outcomes")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No data available for the selected filter.")

            # --- Model Comparison Radar Chart (PRD 4.3.1) ---
            st.subheader("Model Resilience Radar Chart")
            # Use the original unfiltered dataframe for radar chart to always compare all models
            if not df.empty:
                fig_radar = go.Figure()
                models = df['model_name'].unique()
                categories = df['category'].unique()

                for model in models:
                    model_data = df[df['model_name'] == model]
                    scores = [
                        model_data[model_data['category'] == cat]['vulnerability_score'].mean()
                        for cat in categories
                    ]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=scores, theta=categories, fill='toself', name=model
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True, title="Model Performance Across Attack Categories"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.header("Detailed Report Data")
            st.dataframe(df_filtered) # Show filtered data in the table
        else:
            st.error("The selected report file is empty or invalid.")
