# vorak/core/reporting.py

from fpdf import FPDF, HTMLMixin
import pandas as pd
from datetime import datetime
from io import BytesIO
import unicodedata

class PDF(FPDF, HTMLMixin):
    # This class definition remains unchanged
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        try:
            self.add_font('DejaVu', '', '/System/Library/Fonts/DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', '/System/Library/Fonts/DejaVuSans-Bold.ttf', uni=True)
            self.add_font('DejaVu', 'I', '/System/Library/Fonts/DejaVuSans-Oblique.ttf', uni=True)
            self.unicode_font_available = True
        except:
            self.unicode_font_available = False
    
    def safe_text(self, text):
        if self.unicode_font_available: return text
        replacements = {'’': "'", '‘': "'", '”': '"', '“': '"', '–': '-', '—': '--', '…': '...'}
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    def set_safe_font(self, family, style='', size=0):
        if self.unicode_font_available and family in ['Arial', 'Helvetica']:
            self.set_font('DejaVu', style, size)
        else:
            self.set_font(family, style, size)

    def header(self):
        self.set_safe_font('Arial', 'B', 12)
        self.cell(0, 10, self.safe_text('Vorak LLM Security Report'), 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_safe_font('Arial', 'I', 8)
        self.cell(0, 10, self.safe_text(f'Page {self.page_no()}'), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_safe_font('Arial', 'B', 14)
        self.cell(0, 10, self.safe_text(title), 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, content, color=(0, 0, 0)):
        self.set_safe_font('Arial', '', 10)
        self.set_text_color(*color)
        safe_content = self.safe_text(str(content))
        self.multi_cell(0, 5, safe_content)
        self.set_text_color(0, 0, 0)
        self.ln()

def generate_pdf_report(results: list, output_buffer: BytesIO, chart_image_buffer: BytesIO):
    if not results: return

    first_result = results[0]
    df = pd.DataFrame([{"classification": res["analysis"].classification.name, "score": res["analysis"].vulnerability_score} for res in results])
    avg_score = df['score'].mean()
    non_compliant_count = df[df['classification'] == 'NON_COMPLIANT'].shape[0]
    total_prompts = len(df)

    pdf = PDF()
    pdf.add_page()

    report_title = f"Evaluation Summary: '{first_result['prompt'].category}'"
    if any("_ADAPT_" in res["prompt"].id for res in results):
        report_title = f"Adaptive Evaluation for '{first_result['prompt'].id}'"
    elif any("_SCENARIO_" in res["prompt"].id for res in results):
        report_title = f"Scenario Evaluation for '{first_result['prompt'].id}'"
    
    pdf.chapter_title(report_title)
    pdf.set_safe_font('Arial', '', 11)
    pdf.cell(0, 8, pdf.safe_text(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), 0, 1, 'L')
    pdf.cell(0, 8, pdf.safe_text(f"Model Tested: {first_result['response'].model_name}"), 0, 1, 'L')
    pdf.ln(5)

    pdf.set_safe_font('Arial', 'B', 12)
    pdf.cell(0, 10, pdf.safe_text("Key Metrics"), 0, 1, 'L')
    pdf.set_safe_font('Arial', '', 11)
    pdf.cell(0, 8, pdf.safe_text(f"- Total Prompts Evaluated: {total_prompts}"), 0, 1, 'L')
    pdf.cell(0, 8, pdf.safe_text(f"- Non-Compliant Responses: {non_compliant_count} ({(non_compliant_count/total_prompts)*100 if total_prompts > 0 else 0:.1f}%)"), 0, 1, 'L')
    pdf.cell(0, 8, pdf.safe_text(f"- Average Vulnerability Score: {avg_score:.2f} / 100.0"), 0, 1, 'L')
    pdf.ln(10)

    pdf.chapter_title("Classification Breakdown")
    try:
        chart_image_buffer.seek(0)
        pdf.image(chart_image_buffer, x=10, y=None, w=190, type='PNG')
    except Exception as e:
        pdf.set_safe_font('Arial', 'I', 10)
        pdf.cell(0, 8, pdf.safe_text(f"Chart could not be inserted: {str(e)}"), 0, 1, 'L')
    
    pdf.add_page()
    pdf.chapter_title("Detailed Evaluation Results")

    for i, res in enumerate(results):
        prompt, response, analysis = res["prompt"], res["response"], res["analysis"]
        if i > 0:
            pdf.ln(5)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(5)

        result_header = f"Result for Prompt ID: {prompt.id}"
        if "_ADAPT_" in prompt.id: result_header = f"Adaptive Escalation Result (ID: {prompt.id})"
        if "_SCENARIO_" in prompt.id: result_header = f"Scenario Turn Result (ID: {prompt.id})"
        
        pdf.set_safe_font('Arial', 'B', 11)
        pdf.cell(0, 8, pdf.safe_text(result_header), 0, 1, 'L')
        
        classification_color = (0, 0, 0)
        if analysis.classification.name == "NON_COMPLIANT": classification_color = (220, 50, 50)
        elif analysis.classification.name == "COMPLIANT": classification_color = (0, 128, 0)
        elif analysis.classification.name == "PARTIAL_COMPLIANCE": classification_color = (255, 165, 0)
        
        pdf.set_safe_font('Arial', 'B', 10)
        pdf.set_text_color(*classification_color)
        pdf.cell(0, 6, pdf.safe_text(f"Classification: {analysis.classification.name} (Score: {analysis.vulnerability_score:.1f})"), 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        prompt_header = "Adversarial Prompt:"
        if "_ADAPT_" in prompt.id: prompt_header = "Generated Adaptive Prompt:"
        if "_SCENARIO_" in prompt.id: prompt_header = f"Generated Scenario Prompt (Turn {prompt.id.split('_')[-1]}):"

        pdf.set_safe_font('Arial', 'B', 10)
        pdf.cell(0, 8, pdf.safe_text(prompt_header), 0, 1, 'L')
        pdf.chapter_body(prompt.prompt_text)

        pdf.set_safe_font('Arial', 'B', 10)
        pdf.cell(0, 8, pdf.safe_text("Model Output:"), 0, 1, 'L')
        pdf.chapter_body(response.output_text, color=(220, 50, 50) if analysis.classification.name == "NON_COMPLIANT" else (0, 0, 0))
        
        pdf.set_safe_font('Arial', 'B', 10)
        pdf.cell(0, 8, pdf.safe_text("AI Analysis:"), 0, 1, 'L')
        pdf.chapter_body(analysis.explanation)

        # --- NEW: Feature 5 - Add governance section to the report ---
        if analysis.governance:
            pdf.set_safe_font('Arial', 'B', 11)
            pdf.cell(0, 10, pdf.safe_text("Governance & Compliance Risks"), 0, 1, 'L')
            
            if analysis.governance.nist_ai_rmf:
                pdf.set_safe_font('Arial', 'B', 10)
                pdf.cell(0, 7, pdf.safe_text("NIST AI RMF:"), 0, 1, 'L')
                pdf.chapter_body("- " + "\n- ".join(analysis.governance.nist_ai_rmf))

            if analysis.governance.eu_ai_act:
                pdf.set_safe_font('Arial', 'B', 10)
                pdf.cell(0, 7, pdf.safe_text("EU AI Act:"), 0, 1, 'L')
                pdf.chapter_body("- " + "\n- ".join(analysis.governance.eu_ai_act))

            if analysis.governance.iso_iec_23894:
                pdf.set_safe_font('Arial', 'B', 10)
                pdf.cell(0, 7, pdf.safe_text("ISO/IEC 23894:"), 0, 1, 'L')
                pdf.chapter_body("- " + "\n- ".join(analysis.governance.iso_iec_23894))

    pdf.output(output_buffer)
