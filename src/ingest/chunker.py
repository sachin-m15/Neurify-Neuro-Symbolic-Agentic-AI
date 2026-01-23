import re
from typing import List
from src.core.models import Chunk

def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    # Split on period, question mark, exclamation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_pages(doc_id: str, pages: list[dict], chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_section = None
    for page_obj in pages:
        page = page_obj["page"]
        text = page_obj["text"]

        # Simple section detection: look for patterns like "1. Title" or "HIV" as section
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.\s+[A-Z]', line) or re.match(r'^[A-Z][A-Z\s]+$', line) and len(line) < 50:
                current_section = line

        sentences = split_into_sentences(text)
        current_chunk = []
        current_length = 0
        idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunk_id = f"{doc_id}_p{page}_c{idx:03d}"
                    start_char = text.find(chunk_text[:50])  # Approximate start
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            page=page,
                            text=chunk_text,
                            section=current_section,
                            meta={"start": start_char, "end": start_char + len(chunk_text)},
                        )
                    )
                    idx += 1
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-overlap//100:] if overlap > 0 else []  # Rough overlap
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        # Last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunk_id = f"{doc_id}_p{page}_c{idx:03d}"
                start_char = text.find(chunk_text[:50])
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        page=page,
                        text=chunk_text,
                        section=current_section,
                        meta={"start": start_char, "end": start_char + len(chunk_text)},
                    )
                )
    return chunks
