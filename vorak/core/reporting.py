# vorak/core/reporting.py

from fpdf import FPDF
import pandas as pd
from datetime import datetime
from io import BytesIO
from typing import List

# This import is safe because it's only used within a function
# that is called by the CLI or UI, which will have the class available.
from .comparison import ComparisonReport

# --- CONSTANTS FOR PROFESSIONAL STYLING ---
COLOR_PRIMARY = (34, 49, 63)      # Dark Blue-Gray
COLOR_SECONDARY = (75, 119, 190) # Royal Blue
COLOR_TEXT = (52, 73, 94)        # Muted Gray
COLOR_SUCCESS = (39, 174, 96)    # Green
COLOR_WARNING = (243, 156, 18)   # Orange
COLOR_DANGER = (231, 76, 60)     # Red
COLOR_LIGHT_GRAY = (236, 240, 241)

class PDF(FPDF):
    """
    A professionally redesigned PDF class for executive-level security reporting.
    Features a modern design, clear typography, and a focus on actionable insights.
    """
    
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.set_auto_page_break(auto=True, margin=20)
        self.set_left_margin(20)
        self.set_right_margin(20)
        self.alias_nb_pages()

    def safe_text(self, text: str) -> str:
        """Encodes text for PDF compatibility, replacing unknown characters."""
        return str(text).encode('latin-1', 'replace').decode('latin-1')

    def header(self):
        if self.page_no() == 1: return # No header on title page
        self.set_y(10)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(*COLOR_TEXT)
        self.cell(0, 10, 'Vorak LLM Security Assessment Report', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'R')

    def footer(self):
        if self.page_no() == 1: return # No footer on title page
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f" Confidential | Report Generated on {datetime.now().strftime('%B %d, %Y')}", 0, 0, 'C')

    def title_page(self, title, model_name):
        self.add_page()
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 297, 'F') # Full page background
        
        # --- VORAK LOGO ---
        self.set_y(40)
        self.set_font('Arial', 'B', 36)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, "VORAK", 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.set_text_color(180, 180, 180)
        self.cell(0, 10, "AI Security & Red Teaming", 0, 1, 'C')
        
        # --- REPORT TITLE BOX ---
        self.set_y(100)
        self.set_font('Arial', 'B', 22)
        self.set_text_color(255,255,255)
        self.multi_cell(0, 12, self.safe_text(title), 0, 'C')
        self.ln(15)
        
        self.set_font('Arial', '', 14)
        self.set_text_color(220, 220, 220)
        self.cell(0, 10, self.safe_text(f"Model Assessed: {model_name}"), 0, 1, 'C')
        self.ln(5)
        self.cell(0, 10, self.safe_text(f"Date of Assessment: {datetime.now().strftime('%B %d, %Y')}"), 0, 1, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, self.safe_text(title), 0, 1, 'L')
        self.set_draw_color(*COLOR_SECONDARY)
        self.line(self.get_x(), self.get_y(), self.get_x() + 170, self.get_y())
        self.ln(8)

    def subsection_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(*COLOR_SECONDARY)
        self.cell(0, 8, self.safe_text(title), 0, 1, 'L')
        self.ln(2)

    def body_text(self, text, style=''):
        self.set_font('Arial', style, 10)
        self.set_text_color(*COLOR_TEXT)
        self.multi_cell(0, 6, self.safe_text(text))
        self.ln(2)

    def key_value_pair(self, key, value, color=COLOR_TEXT):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(50, 6, self.safe_text(key))
        self.set_font('Arial', '', 10)
        self.set_text_color(*color)
        self.multi_cell(0, 6, self.safe_text(value))
        self.ln(2)

    def risk_box(self, classification, score):
        if classification == "NON_COMPLIANT":
            box_color = COLOR_DANGER
            text = "HIGH RISK"
        elif classification == "PARTIAL_COMPLIANCE":
            box_color = COLOR_WARNING
            text = "MEDIUM RISK"
        else:
            box_color = COLOR_SUCCESS
            text = "LOW RISK / COMPLIANT"

        self.set_font('Arial', 'B', 10)
        self.set_fill_color(*box_color)
        self.set_text_color(255,255,255)
        self.cell(40, 8, text, 0, 0, 'C', fill=True)
        self.set_font('Arial', '', 10)
        self.set_text_color(*COLOR_TEXT)
        self.cell(0, 8, f"  (Vulnerability Score: {score:.1f}/100.0)", 0, 1)
        self.ln(5)

def generate_pdf_report(results: list, output_buffer: BytesIO, chart_image_buffer: BytesIO):
    if not results: return
    
    df = pd.DataFrame([
        {
            "id": res["prompt"].id,
            "classification": res["analysis"].classification.name,
            "score": res["analysis"].vulnerability_score
        } for res in results
    ])
    avg_score = df['score'].mean()
    non_compliant_count = df[df['classification'] == 'NON_COMPLIANT'].shape[0]
    partial_count = df[df['classification'] == 'PARTIAL_COMPLIANCE'].shape[0]
    total_prompts = len(df)
    first_result = results[0]

    pdf = PDF()
    report_title = f"Security Assessment for '{first_result['prompt'].category}'"
    if any("_ADAPT_" in res["prompt"].id for res in results):
        report_title = f"Adaptive Threat Assessment for '{first_result['prompt'].id}'"
    
    pdf.title_page(report_title, first_result['response'].model_name)
    
    # --- EXECUTIVE SUMMARY PAGE ---
    pdf.add_page()
    pdf.section_title("Executive Summary")
    
    pdf.subsection_title("Overall Risk Posture")
    overall_class = "COMPLIANT"
    if non_compliant_count > 0: overall_class = "NON_COMPLIANT"
    elif partial_count > 0: overall_class = "PARTIAL_COMPLIANCE"
    pdf.risk_box(overall_class, avg_score)
    
    summary_text = (
        f"This report details the security assessment of the '{first_result['response'].model_name}' model. "
        f"A total of {total_prompts} adversarial prompts were executed. The model demonstrated an average "
        f"vulnerability score of {avg_score:.2f}, indicating its overall resilience against the tested threat vectors. "
        f"There were {non_compliant_count} instances of high-risk (non-compliant) responses and {partial_count} instances of medium-risk (partially-compliant) responses."
    )
    pdf.body_text(summary_text)

    pdf.subsection_title("Key Findings")
    pdf.key_value_pair("Total Prompts Evaluated:", str(total_prompts))
    pdf.key_value_pair("High-Risk Findings:", str(non_compliant_count), COLOR_DANGER if non_compliant_count > 0 else COLOR_SUCCESS)
    pdf.key_value_pair("Medium-Risk Findings:", str(partial_count), COLOR_WARNING if partial_count > 0 else COLOR_SUCCESS)
    
    pdf.subsection_title("Classification Breakdown")
    try:
        chart_image_buffer.seek(0)
        pdf.image(chart_image_buffer, x=pdf.get_x(), y=pdf.get_y(), w=170, type='PNG')
    except Exception as e:
        pdf.body_text(f"Chart could not be inserted: {e}")

    # --- DETAILED FINDINGS PAGES ---
    pdf.add_page()
    pdf.section_title("Detailed Findings")
    
    for i, res in enumerate(results):
        if pdf.get_y() > 220 and i < len(results) -1: pdf.add_page()
        prompt, response, analysis = res["prompt"], res["response"], res["analysis"]
        
        pdf.subsection_title(f"Finding #{i+1}: Prompt ID '{prompt.id}'")
        pdf.risk_box(analysis.classification.name, analysis.vulnerability_score)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(0, 6, "Adversarial Prompt", 0, 1)
        pdf.set_fill_color(*COLOR_LIGHT_GRAY)
        pdf.set_text_color(*COLOR_TEXT)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, pdf.safe_text(prompt.prompt_text), 1, 'L', fill=True)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(0, 6, "Model Response", 0, 1)
        pdf.set_fill_color(*COLOR_LIGHT_GRAY)
        pdf.set_text_color(*COLOR_TEXT)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, pdf.safe_text(response.output_text), 1, 'L', fill=True)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(0, 6, "Security Analysis", 0, 1)
        pdf.body_text(analysis.explanation)

        # --- Recommended Mitigation ---
        mitigation = "Monitor model behavior for similar inputs. Since the model was compliant, no immediate action is required."
        if analysis.classification.name == "NON_COMPLIANT":
            mitigation = "CRITICAL: This vulnerability requires immediate attention. Fine-tune the model with examples of this attack vector to enforce refusal. Implement stricter output filtering and review associated guardrails."
        elif analysis.classification.name == "PARTIAL_COMPLIANCE":
            mitigation = "Review and strengthen model guardrails related to this category. Consider additional fine-tuning to make the model's refusal more robust and explicit."
        
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(0, 6, "Recommended Mitigation", 0, 1)
        pdf.body_text(mitigation)

        if analysis.governance:
            pdf.subsection_title("Compliance & Governance Mapping")
            gov_text = []
            if analysis.governance.nist_ai_rmf: gov_text.append(f"NIST AI RMF: {', '.join(analysis.governance.nist_ai_rmf)}")
            if analysis.governance.eu_ai_act: gov_text.append(f"EU AI Act: {', '.join(analysis.governance.eu_ai_act)}")
            if analysis.governance.iso_iec_23894: gov_text.append(f"ISO/IEC 23894: {', '.join(analysis.governance.iso_iec_23894)}")
            if analysis.governance.mitre_atlas: gov_text.append(f"MITRE ATLAS: {', '.join(analysis.governance.mitre_atlas)}")
            pdf.body_text("\n".join(gov_text))
        
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
        pdf.ln(5)

    pdf.output(output_buffer)

def generate_comparison_pdf_report(report: ComparisonReport, output_buffer: BytesIO, chart_image_buffer: BytesIO):
    # This function can also be updated with the new styling if needed.
    # For now, keeping the focus on the main evaluation report.
    pdf = PDF()
    summary = report.summary
    pdf.title_page("Session Comparison Report", f"{report.results[0].model_name_a} vs. {report.results[0].model_name_b}")
    pdf.add_page()

    pdf.section_title("Overall Performance Delta")
    pdf.body_text(f"**Baseline Session (A):** {summary.session_a_id}\n"
                  f"**Candidate Session (B):** {summary.session_b_id}")
    
    delta_color = (0, 128, 0) if summary.avg_score_delta < 0 else (220, 50, 50)
    pdf.body_text(f"**Average Score (A):** {summary.avg_score_a:.2f}\n"
                  f"**Average Score (B):** {summary.avg_score_b:.2f}")
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.cell(50, 6, "Overall Delta:")
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(*delta_color)
    pdf.multi_cell(0, 6, f"{summary.avg_score_delta:+.2f}")
    pdf.ln(2)
    
    pdf.subsection_title("Comparison Breakdown")
    try:
        chart_image_buffer.seek(0)
        pdf.image(chart_image_buffer, x=pdf.get_x(), y=pdf.get_y(), w=170, type='PNG')
    except Exception as e:
        pdf.body_text(f"Chart could not be inserted: {e}")

    pdf.add_page()
    pdf.section_title("Detailed Prompt Comparison")
    
    # Table Header
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(*COLOR_PRIMARY)
    pdf.set_text_color(255,255,255)
    pdf.cell(30, 7, 'Prompt ID', 1, 0, 'C', fill=True)
    pdf.cell(25, 7, 'Score (A)', 1, 0, 'C', fill=True)
    pdf.cell(25, 7, 'Score (B)', 1, 0, 'C', fill=True)
    pdf.cell(20, 7, 'Delta', 1, 0, 'C', fill=True)
    pdf.cell(70, 7, 'Status / Classification Change', 1, 1, 'C', fill=True)

    # Table Rows
    pdf.set_font('Arial', '', 8)
    pdf.set_text_color(*COLOR_TEXT)
    for res in report.results:
        pdf.cell(30, 6, pdf.safe_text(res.prompt_id), 1)
        pdf.cell(25, 6, f"{res.score_a:.1f}", 1, 0, 'C')
        pdf.cell(25, 6, f"{res.score_b:.1f}", 1, 0, 'C')
        
        delta = res.delta
        if delta < 0: d_color = COLOR_SUCCESS
        elif delta > 0: d_color = COLOR_DANGER
        else: d_color = COLOR_TEXT
        
        pdf.set_text_color(*d_color)
        pdf.cell(20, 6, f"{delta:+.1f}", 1, 0, 'C')
        pdf.set_text_color(*COLOR_TEXT)
        
        status_text = res.status
        if res.classification_a != res.classification_b:
            status_text += f" ({res.classification_a} -> {res.classification_b})"
        pdf.multi_cell(70, 6, pdf.safe_text(status_text), 1, 'L')

    pdf.output(output_buffer)
