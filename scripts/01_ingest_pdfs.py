import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.utils import safe_doc_id
from src.ingest.pdf_loader import extract_text_from_pdf
from src.ingest.cleaner import clean_page_text
from src.ingest.chunker import chunk_pages

load_dotenv()

RAW_DIR = Path("data/raw_pdfs")
OUT_DIR = Path("data/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    all_chunks = []
    pdfs = list(RAW_DIR.glob("*.pdf"))
    print(f"Processing {len(pdfs)} PDFs...")
    for i, pdf in enumerate(pdfs, 1):
        doc_id = safe_doc_id(str(pdf))
        print(f"[{i}/{len(pdfs)}] Processing {pdf.name}...")
        pages = extract_text_from_pdf(str(pdf))
        print(f"  Extracted {len(pages)} pages")
        for p in pages:
            p["text"] = clean_page_text(p["text"])

        chunks = chunk_pages(doc_id=doc_id, pages=pages)
        print(f"  Created {len(chunks)} chunks")
        all_chunks.extend([c.model_dump() for c in chunks])

    out_path = OUT_DIR / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Saved chunks: {len(all_chunks)} -> {out_path}")

if __name__ == "__main__":
    main()
