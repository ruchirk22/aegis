# aegis/core/reporting.py

from fpdf import FPDF
import pandas as pd
import os
from datetime import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        # The shield emoji was removed from the title to prevent Unicode encoding errors
        # with the default PDF fonts.
        self.cell(0, 10, 'Aegis LLM Security Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(results: list, filepath: str, chart_image_path: str):
    """
    Generates a professional PDF report from the evaluation results.
    """
    df = pd.DataFrame([{
        "prompt_id": res["prompt"].id,
        "category": res["prompt"].category,
        "model_name": res["response"].model_name,
        "classification": res["analysis"].classification.name,
        "score": res["analysis"].vulnerability_score
    } for res in results])

    avg_score = df['score'].mean()
    non_compliant_count = df[df['classification'] == 'NON_COMPLIANT'].shape[0]
    total_prompts = len(df)

    pdf = PDF()
    pdf.add_page()

    # --- Title and Summary ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"Evaluation Summary for '{df['category'].iloc[0]}'", 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.cell(0, 8, f"Model Tested: {df['model_name'].iloc[0]}", 0, 1, 'L')
    pdf.ln(5)

    # --- Key Metrics ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Key Metrics", 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"- Total Prompts Evaluated: {total_prompts}", 0, 1, 'L')
    pdf.cell(0, 8, f"- Non-Compliant Responses: {non_compliant_count} ({ (non_compliant_count/total_prompts)*100 if total_prompts > 0 else 0:.1f}%)", 0, 1, 'L')
    pdf.cell(0, 8, f"- Average Vulnerability Score: {avg_score:.2f} / 100.0", 0, 1, 'L')
    pdf.ln(10)

    # --- Chart ---
    if os.path.exists(chart_image_path):
        pdf.image(chart_image_path, x=10, y=None, w=190)
        pdf.ln(10)

    # --- Detailed Results Table ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Detailed Results", 0, 1, 'L')
    pdf.set_font('Arial', '', 9)
    
    # Table Header
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(25, 7, 'Prompt ID', 1, 0, 'C', 1)
    pdf.cell(65, 7, 'Classification', 1, 0, 'C', 1)
    pdf.cell(20, 7, 'Score', 1, 1, 'C', 1)

    # Table Rows
    for index, row in df.iterrows():
        pdf.cell(25, 6, row['prompt_id'], 1, 0)
        pdf.cell(65, 6, row['classification'], 1, 0)
        pdf.cell(20, 6, f"{row['score']:.1f}", 1, 1)

    pdf.output(filepath)
