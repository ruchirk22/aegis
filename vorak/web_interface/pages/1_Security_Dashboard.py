# vorak/web_interface/pages/1_Security_Dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Correction ---

from vorak.core.database.manager import DatabaseManager

# --- Page Configuration ---
st.set_page_config(page_title="Vorak Security Dashboard", page_icon="🛡️", layout="wide")

st.title("Security & Compliance Dashboard")
st.markdown("High-level overview of your AI model's security posture and compliance risks.")

@st.cache_resource
def get_db_manager():
    return DatabaseManager()

db_manager = get_db_manager()

@st.cache_data(ttl=60)
def load_data():
    df = db_manager.get_all_results_as_df()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure governance columns exist
        for col in ['governance_nist', 'governance_eu', 'governance_iso']:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].fillna('')
    return df

df = load_data()

if not df.empty:
    st.sidebar.header("Dashboard Filters")
    all_models = ["All Models"] + sorted(df['model_name'].unique().tolist())
    selected_model = st.sidebar.selectbox("Filter by Model", options=all_models)
    
    df_filtered = df[df['model_name'] == selected_model] if selected_model != "All Models" else df

    # --- KPIs ---
    st.header("Key Performance Indicators")
    kpi1, kpi2, kpi3 = st.columns(3)
    avg_score = df_filtered['vulnerability_score'].mean()
    non_compliant_rate = (df_filtered[df_filtered['classification'] == 'NON_COMPLIANT'].shape[0] / df_filtered.shape[0]) * 100 if not df_filtered.empty else 0
    
    kpi1.metric("Total Tests Run", df_filtered.shape[0])
    kpi2.metric("Average Vulnerability Score", f"{avg_score:.2f}")
    kpi3.metric("Non-Compliant Rate", f"{non_compliant_rate:.2f}%")

    st.divider()

    # --- Visualizations ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vulnerability by Attack Category")
        if not df_filtered.empty:
            # --- REPLACEMENT CHART: More insightful for CXOs ---
            category_vulns = df_filtered.groupby('category')['vulnerability_score'].mean().sort_values(ascending=True).reset_index()
            
            if not category_vulns.empty:
                fig_cat = px.bar(
                    category_vulns,
                    x='vulnerability_score',
                    y='category',
                    orientation='h',
                    title="Average Vulnerability Score per Category",
                    labels={'vulnerability_score': 'Average Score', 'category': 'Attack Category'},
                    text='vulnerability_score'
                )
                fig_cat.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Not enough data to display category vulnerabilities.")
        else:
            st.info("No data available for the selected filters.")

    with col2:
        st.subheader("Classification Breakdown")
        classification_counts = df_filtered['classification'].value_counts()
        fig_pie = px.pie(classification_counts, values=classification_counts.values, names=classification_counts.index,
                         title="Evaluation Outcomes", hole=.3,
                         color=classification_counts.index,
                         color_discrete_map={'NON_COMPLIANT': '#FF4B4B', 'COMPLIANT': '#2ECC71', 'PARTIAL_COMPLIANCE': '#FFA500'})
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    
    # --- Governance Hotspots ---
    st.header("Governance & Compliance Hotspots")
    gcol1, gcol2, gcol3 = st.columns(3)
    
    nist_risks = df_filtered[df_filtered['governance_nist'].str.strip() != '']['governance_nist'].str.split(', ').explode().value_counts()
    eu_risks = df_filtered[df_filtered['governance_eu'].str.strip() != '']['governance_eu'].str.split(', ').explode().value_counts()
    iso_risks = df_filtered[df_filtered['governance_iso'].str.strip() != '']['governance_iso'].str.split(', ').explode().value_counts()

    with gcol1:
        st.subheader("NIST AI RMF")
        if not nist_risks.empty: st.dataframe(nist_risks)
        else: st.info("No NIST risks recorded.")
    with gcol2:
        st.subheader("EU AI Act")
        if not eu_risks.empty: st.dataframe(eu_risks)
        else: st.info("No EU AI Act risks recorded.")
    with gcol3:
        st.subheader("ISO/IEC 23894")
        if not iso_risks.empty: st.dataframe(iso_risks)
        else: st.info("No ISO/IEC risks recorded.")

    st.divider()

    # --- Detailed Data ---
    with st.expander("Detailed Report Data"):
        st.dataframe(df_filtered)
else:
    st.info("No results found in the database. Run some evaluations in the Sandbox to populate the dashboard.")
