import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.models import Chunk
from src.core.utils import safe_doc_id
from src.ingest.cleaner import clean_page_text

load_dotenv()

CSV_PATH = Path("data/train.csv")
OUT_DIR = Path("data/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    all_chunks = []
    df = pd.read_csv(CSV_PATH)
    print(f"Processing {len(df)} rows from CSV...")

    for idx, row in df.iterrows():
        doc_id = f"csv_row_{idx}"
        question = row['Question']
        answer = row['Answer']
        qtype = row['qtype']
        text = f"Question: {question}\nAnswer: {answer}"
        cleaned_text = clean_page_text(text)

        chunk = Chunk(
            chunk_id=f"{doc_id}_chunk",
            doc_id=doc_id,
            page=1,
            text=cleaned_text,
            section=qtype,
            meta=row.to_dict()
        )
        all_chunks.append(chunk.model_dump())

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} rows...")

    out_path = OUT_DIR / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Saved chunks: {len(all_chunks)} -> {out_path}")

if __name__ == "__main__":
    main()
