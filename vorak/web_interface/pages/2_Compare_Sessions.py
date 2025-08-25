# vorak/web_interface/pages/2_Compare_Sessions.py

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from io import BytesIO

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Correction ---

from vorak.core.database.manager import DatabaseManager
from vorak.core.comparison import ComparisonReport
from vorak.core.reporting import generate_comparison_pdf_report

# --- Page Configuration ---
st.set_page_config(page_title="Vorak Session Comparison", page_icon="📊", layout="wide")

st.title("Session Comparison")
st.markdown("Compare the performance of two different test sessions to track improvements and regressions.")

@st.cache_resource
def get_db_manager():
    return DatabaseManager()

@st.cache_data(ttl=60)
def load_session_ids():
    db_manager = get_db_manager()
    df = db_manager.get_all_results_as_df()
    if not df.empty and 'session_id' in df.columns:
        return sorted(df['session_id'].unique().tolist(), reverse=True)
    return []

session_ids = load_session_ids()

if not session_ids or len(session_ids) < 2:
    st.warning("You need at least two completed test sessions in your database to use the comparison feature.")
else:
    col1, col2 = st.columns(2)
    with col1:
        session_a = st.selectbox("Select Baseline Session (A)", options=session_ids, index=1)
    with col2:
        session_b = st.selectbox("Select Candidate Session (B)", options=session_ids, index=0)

    if st.button("Compare Sessions", type="primary", use_container_width=True):
        if session_a == session_b:
            st.error("Please select two different sessions to compare.")
        else:
            try:
                db_manager = get_db_manager()
                report = ComparisonReport(session_a, session_b, db_manager)
                summary = report.summary
                
                st.header("Comparison Summary")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Avg. Score (A)", f"{summary.avg_score_a:.2f}")
                kpi2.metric("Avg. Score (B)", f"{summary.avg_score_b:.2f}")
                kpi3.metric("Overall Delta", f"{summary.avg_score_delta:+.2f}", delta_color=("inverse" if summary.avg_score_delta < 0 else "normal"))

                st.subheader("Breakdown")
                chart_data = {
                    'Status': ['Improvements', 'Regressions', 'Unchanged'],
                    'Count': [summary.improvements, summary.regressions, summary.unchanged]
                }
                fig = px.bar(pd.DataFrame(chart_data), x='Status', y='Count', color='Status',
                             color_discrete_map={'Improvements': '#2ECC71', 'Regressions': '#FF4B4B', 'Unchanged': 'grey'},
                             title='Comparison of Outcomes')
                st.plotly_chart(fig, use_container_width=True)

                st.header("Detailed Comparison")
                
                # Create a DataFrame for display
                display_data = []
                for res in report.results:
                    display_data.append({
                        "Prompt ID": res.prompt_id,
                        "Score (A)": res.score_a,
                        "Score (B)": res.score_b,
                        "Delta": res.delta,
                        "Status": res.status,
                        "Classification Change": f"{res.classification_a} → {res.classification_b}" if res.classification_a != res.classification_b else "N/A"
                    })
                df_display = pd.DataFrame(display_data)
                st.dataframe(df_display)

                # --- PDF Download ---
                st.subheader("Download Report")
                chart_buffer = BytesIO()
                fig.write_image(chart_buffer, format='png', engine='kaleido')
                
                pdf_buffer = BytesIO()
                generate_comparison_pdf_report(report, pdf_buffer, chart_buffer)
                pdf_buffer.seek(0)
                
                st.download_button(
                    label="Download Comparison PDF",
                    data=pdf_buffer,
                    file_name=f"comparison_{session_a}_{session_b}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except ValueError as e:
                st.error(f"Error generating comparison: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
