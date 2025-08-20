# aegis/core/reporting.py

from fpdf import FPDF, HTMLMixin
import pandas as pd
from datetime import datetime
from io import BytesIO
import unicodedata

class PDF(FPDF, HTMLMixin):
    """A custom PDF class that includes a header and footer."""
    
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        # Add a Unicode-compatible font (DejaVu Sans)
        try:
            self.add_font('DejaVu', '', '/System/Library/Fonts/DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', '/System/Library/Fonts/DejaVuSans-Bold.ttf', uni=True)
            self.add_font('DejaVu', 'I', '/System/Library/Fonts/DejaVuSans-Oblique.ttf', uni=True)
            self.unicode_font_available = True
        except:
            # Fallback to default fonts if DejaVu is not available
            self.unicode_font_available = False
    
    def safe_text(self, text):
        """Convert text to ASCII-compatible format if Unicode font is not available."""
        if self.unicode_font_available:
            return text
        else:
            # Replace common Unicode characters with ASCII equivalents
            replacements = {
                ''': "'",  # Left single quotation mark
                ''': "'",  # Right single quotation mark
                '"': '"',  # Left double quotation mark
                '"': '"',  # Right double quotation mark
                '–': '-',  # En dash
                '—': '--', # Em dash
                '…': '...',# Horizontal ellipsis
                '©': '(c)',# Copyright
                '®': '(r)',# Registered
                '™': '(tm)',# Trademark
            }
            
            for unicode_char, ascii_char in replacements.items():
                text = text.replace(unicode_char, ascii_char)
            
            # Remove any remaining non-ASCII characters
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('ascii')
            return text
    
    def set_safe_font(self, family, style='', size=0):
        """Set font with Unicode support if available."""
        if self.unicode_font_available and family in ['Arial', 'Helvetica']:
            self.set_font('DejaVu', style, size)
        else:
            self.set_font(family, style, size)

    def header(self):
        self.set_safe_font('Arial', 'B', 12)
        self.cell(0, 10, self.safe_text('Aegis LLM Security Report'), 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_safe_font('Arial', 'I', 8)
        self.cell(0, 10, self.safe_text(f'Page {self.page_no()}'), 0, 0, 'C')

    def chapter_title(self, title):
        """Adds a formatted chapter title."""
        self.set_safe_font('Arial', 'B', 14)
        self.cell(0, 10, self.safe_text(title), 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, content, color=(0, 0, 0)):
        """Adds formatted multi-line body text."""
        self.set_safe_font('Arial', '', 10)
        self.set_text_color(*color)
        safe_content = self.safe_text(str(content))
        self.multi_cell(0, 5, safe_content)
        self.set_text_color(0, 0, 0)  # Reset color
        self.ln()

def generate_pdf_report(results: list, output_buffer: BytesIO, chart_image_buffer: BytesIO):
    """
    Generates a professional, detailed PDF report from evaluation results.
    """
    if not results:
        return

    first_result = results[0]
    df = pd.DataFrame([{"classification": res["analysis"].classification.name, "score": res["analysis"].vulnerability_score} for res in results])
    
    avg_score = df['score'].mean()
    non_compliant_count = df[df['classification'] == 'NON_COMPLIANT'].shape[0]
    total_prompts = len(df)

    pdf = PDF()
    pdf.add_page()

    # --- Summary Page ---
    pdf.chapter_title(f"Evaluation Summary: '{first_result['prompt'].category}'")
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
    
    # Handle chart image insertion safely
    try:
        chart_image_buffer.seek(0)  # Reset buffer position
        pdf.image(chart_image_buffer, x=10, y=None, w=190, type='PNG')
    except Exception as e:
        # If chart insertion fails, add a text note
        pdf.set_safe_font('Arial', 'I', 10)
        pdf.cell(0, 8, pdf.safe_text(f"Chart could not be inserted: {str(e)}"), 0, 1, 'L')
    
    # --- Detailed Results Section ---
    pdf.add_page()
    pdf.chapter_title("Detailed Evaluation Results")

    for i, res in enumerate(results):
        prompt = res["prompt"]
        response = res["response"]
        analysis = res["analysis"]

        # Add a separator for all but the first entry
        if i > 0:
            pdf.ln(5)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(5)

        pdf.set_safe_font('Arial', 'B', 11)
        pdf.cell(0, 8, pdf.safe_text(f"Result for Prompt ID: {prompt.id}"), 0, 1, 'L')
        
        # Display classification and score
        classification_color = (0, 0, 0)  # Default black
        if analysis.classification.name == "NON_COMPLIANT": 
            classification_color = (220, 50, 50)  # Red
        elif analysis.classification.name == "COMPLIANT": 
            classification_color = (0, 128, 0)  # Green
        elif analysis.classification.name == "PARTIAL_COMPLIANCE": 
            classification_color = (255, 165, 0)  # Orange
        
        pdf.set_safe_font('Arial', 'B', 10)
        pdf.set_text_color(*classification_color)
        pdf.cell(0, 6, pdf.safe_text(f"Classification: {analysis.classification.name} (Score: {analysis.vulnerability_score:.1f})"), 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)  # Reset color

        pdf.set_safe_font('Arial', 'B', 10)
        pdf.cell(0, 8, pdf.safe_text("Adversarial Prompt:"), 0, 1, 'L')
        pdf.chapter_body(prompt.prompt_text)

        pdf.set_safe_font('Arial', 'B', 10)
        pdf.cell(0, 8, pdf.safe_text("Model Output:"), 0, 1, 'L')
        pdf.chapter_body(response.output_text, color=(220, 50, 50) if analysis.classification.name == "NON_COMPLIANT" else (0, 0, 0))
        
        pdf.set_safe_font('Arial', 'B', 10)
        pdf.cell(0, 8, pdf.safe_text("AI Analysis:"), 0, 1, 'L')
        pdf.chapter_body(analysis.explanation)

    pdf.output(output_buffer)