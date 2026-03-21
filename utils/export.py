"""
utils/export.py — PDF and Markdown export for research reports

Provides download-ready exports of the generated research reports:
    - export_markdown(report)  → Save as .md file, return path
    - export_pdf(report)       → Convert markdown to formatted PDF via fpdf2, return path

PDF rendering handles:
    - # and ## headers (different font sizes)
    - Bullet points (- items)
    - Markdown links [text](url) → cleaned for PDF
    - Normal paragraph text with word wrapping
"""

import re
import os
import tempfile
from fpdf import FPDF


def export_markdown(report: str, filename: str = "research_report.md") -> str:
    """
    Save the research report as a Markdown file.

    Args:
        report: The markdown-formatted report text
        filename: Output filename

    Returns:
        Absolute path to the saved .md file
    """
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def export_pdf(report: str, filename: str = "research_report.pdf") -> str:
    """
    Convert a markdown research report to a clean, formatted PDF.
    Handles headers, bullet points, links, and paragraph text.

    Args:
        report: The markdown-formatted report text
        filename: Output filename

    Returns:
        Absolute path to the saved .pdf file
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    lines = report.split("\n")

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            # ── Section Header (##) ──
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(6)
            header_text = stripped.replace("## ", "")
            # Clean any markdown formatting
            header_text = _clean_markdown(header_text)
            pdf.cell(0, 10, header_text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        elif stripped.startswith("# "):
            # ── Main Title (#) ──
            pdf.set_font("Helvetica", "B", 18)
            title_text = stripped.replace("# ", "")
            title_text = _clean_markdown(title_text)
            pdf.cell(0, 12, title_text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        elif stripped.startswith("- "):
            # ── Bullet Point ──
            pdf.set_font("Helvetica", "", 11)
            bullet_text = _clean_markdown(stripped[2:])
            pdf.multi_cell(0, 7, f"  -  {bullet_text}")
            pdf.ln(1)

        elif stripped == "":
            # ── Blank Line ──
            pdf.ln(3)

        else:
            # ── Normal Paragraph ──
            pdf.set_font("Helvetica", "", 11)
            clean_text = _clean_markdown(stripped)
            pdf.multi_cell(0, 7, clean_text)
            pdf.ln(1)

    path = os.path.join(tempfile.gettempdir(), filename)
    pdf.output(path)
    return path


def _clean_markdown(text: str) -> str:
    """
    Remove markdown formatting for PDF output.
    Converts [text](url) → text (url) and strips bold/italic markers.
    """
    # Convert markdown links to plain text
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    # Remove bold markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Remove italic markers
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    return text
