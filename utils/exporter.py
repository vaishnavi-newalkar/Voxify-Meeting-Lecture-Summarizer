from datetime import datetime


def export_to_txt(transcript: str, summary: str, action_items: list[str]) -> bytes:
    """
    Create a formatted plain-text report.

    Returns bytes ready for st.download_button.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        "         AI MEETING SUMMARIZER — REPORT",
        "=" * 60,
        f"Generated : {timestamp}",
        "",
        "─" * 60,
        "SUMMARY",
        "─" * 60,
        summary if summary else "No summary available.",
        "",
    ]

    if action_items:
        lines += [
            "─" * 60,
            "ACTION ITEMS",
            "─" * 60,
        ]
        for i, item in enumerate(action_items, 1):
            lines.append(f"  {i}. {item}")
        lines.append("")

    lines += [
        "─" * 60,
        "FULL TRANSCRIPT",
        "─" * 60,
        transcript if transcript else "No transcript available.",
        "",
        "=" * 60,
        "End of Report",
        "=" * 60,
    ]

    content = "\n".join(lines)
    return content.encode("utf-8")


def export_to_pdf(transcript: str, summary: str, action_items: list[str]) -> bytes:
    """
    Create a PDF report using fpdf2.

    Requires: pip install fpdf2
    Returns bytes ready for st.download_button.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is not installed. Run: pip install fpdf2")

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(180, 83, 9)   # Indigo
            self.cell(0, 10, "Voxify — Meeting Intelligence", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.set_font("Helvetica", "", 9)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cell(0, 6, f"Generated: {ts}", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
            self.set_draw_color(180, 83, 9)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    def section_title(title: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(255, 243, 220)
        pdf.set_text_color(120, 53, 15)
        pdf.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def body_text(text: str):
        pdf.set_font("Helvetica", "", 10)
        # Encode to latin-1 safely
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_text)
        pdf.ln(3)

    # Summary
    section_title("📋 SUMMARY")
    body_text(summary if summary else "No summary available.")

    # Action Items
    if action_items:
        section_title("✅ ACTION ITEMS")
        for i, item in enumerate(action_items, 1):
            safe = item.encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(8, 6, f"{i}.")
            pdf.multi_cell(0, 6, safe)
        pdf.ln(3)

    # Full Transcript
    section_title("📄 FULL TRANSCRIPT")
    body_text(transcript if transcript else "No transcript available.")

    return bytes(pdf.output())
