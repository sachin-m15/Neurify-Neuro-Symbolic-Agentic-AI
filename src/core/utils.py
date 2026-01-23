import hashlib
from pathlib import Path

def file_hash(path: str) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]

def safe_doc_id(pdf_path: str) -> str:
    p = Path(pdf_path)
    h = file_hash(pdf_path)
    return f"{p.stem}_{h}"
