# aegis/web_interface/pages/1_📊_Security_Dashboard.py

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

from aegis.core.database.manager import DatabaseManager

# --- Page Configuration ---
st.set_page_config(page_title="Aegis Security Dashboard", page_icon="📊", layout="wide")

st.title("📊 Security Dashboard")
st.markdown("Visualize historical evaluation results from the central database.")

# --- Caching the Database Manager ---
@st.cache_resource
def get_db_manager():
    return DatabaseManager()

db_manager = get_db_manager()

# --- Data Loading ---
@st.cache_data(ttl=30) # Cache data for 30 seconds
def load_data_from_db():
    try:
        df = db_manager.get_all_results_as_df()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values(by="timestamp", ascending=False)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

df = load_data_from_db()

# --- Main UI ---
if not df.empty:
    st.divider()
    st.header("Filters")
    
    df_session_filtered = df

    # --- FIX: Check if session_id column exists before creating the filter ---
    if 'session_id' in df.columns:
        all_sessions = ["All Sessions"] + df['session_id'].unique().tolist()
        selected_session = st.selectbox("Filter by Testing Session", options=all_sessions)

        if selected_session != "All Sessions":
            df_session_filtered = df[df['session_id'] == selected_session]
    else:
        st.warning("Note: Your database is from an older version. The 'Filter by Session' feature is disabled.")


    # --- Other Interactive Filters ---
    col1, col2 = st.columns(2)
    with col1:
        all_models = df_session_filtered['model_name'].unique()
        # FIX: Corrected typo from multelect to multiselect
        selected_models = st.multiselect("Filter by Model", options=all_models, default=all_models)
    
    with col2:
        all_categories = df_session_filtered['category'].unique()
        selected_categories = st.multiselect("Filter by Category", options=all_categories, default=all_categories)

    # Apply all filters
    df_filtered = df_session_filtered[
        df_session_filtered['model_name'].isin(selected_models) & 
        df_session_filtered['category'].isin(selected_categories)
    ]

    if not df_filtered.empty:
        # (The rest of the visualization code remains the same)
        st.divider()
        st.header("Vulnerability Analysis")
        
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.subheader("Vulnerability Heatmap")
            try:
                heatmap_data = df_filtered.pivot_table(index='category', columns='model_name', values='vulnerability_score', aggfunc='mean').fillna(0)
                fig_heatmap = px.imshow(
                    heatmap_data, 
                    text_auto=".1f", 
                    aspect="auto", 
                    color_continuous_scale='Reds',
                    labels=dict(x="Model Name", y="Attack Category", color="Avg Score"),
                    title="Average Vulnerability Score by Model & Category"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception:
                st.warning("Could not generate heatmap. This chart requires data from multiple models and categories.")

        with gcol2:
            st.subheader("Classification Breakdown")
            classification_counts = df_filtered['classification'].value_counts()
            fig_bar = px.bar(
                classification_counts,
                x=classification_counts.index,
                y=classification_counts.values,
                color=classification_counts.index,
                labels={'x': 'Classification', 'y': 'Count'},
                title="Total Evaluation Outcomes",
                color_discrete_map={
                    'NON_COMPLIant': 'red', 'COMPLIANT': 'green', 'PARTIAL_COMPLIANCE': 'orange',
                    'AMBIGUOUS': 'grey', 'ERROR': 'black'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        st.header("Detailed Report Data")
        st.dataframe(df_filtered)
    else:
        st.warning("No data matches the current filter settings.")
else:
    st.info("No results found in the database. Run some evaluations in the 'Single Prompt' or 'Batch' tabs to populate the dashboard.")
