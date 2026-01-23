"""
PURPOSE:
- Extract text page-by-page from PDFs

DESIGN NOTE:
- Page numbers MUST be preserved for citation
- Do NOT merge pages here

FUTURE:
- OCR fallback for scanned PDFs
"""

import pdfplumber

def extract_text_from_pdf(path: str) -> list[dict]:
    pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append({
                "page": i,
                "text": page.extract_text() or ""
            })

    return pages
