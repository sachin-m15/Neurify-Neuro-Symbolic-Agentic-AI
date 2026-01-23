SYSTEM_CLAIM_GENERATION = """
You are a strict assistant for Verified RAG over PDF documents.

You MUST:
- Answer ONLY using the provided evidence chunks or from the FAISS index present
- Produce atomic claims
- Each claim MUST include citations using ONLY chunk_ids from evidence
- If evidence is insufficient, set short_answer to "NOT_ENOUGH_EVIDENCE"
- Do not guess. Do not hallucinate.
Return ONLY valid JSON.
"""

def build_user_prompt(question: str, evidence_chunks: list) -> str:
    evidence_text = "\n\n".join(
        [f"[{c.chunk_id} | page={c.page}] {c.text}" for c in evidence_chunks]
    )
    return f"""
Question:
{question}

Evidence chunks:
{evidence_text}

Return JSON with this schema:
{{
  "short_answer": "string",
  "claims": [
    {{
      "claim_id": "C1",
      "text": "string",
      "citations": ["chunk_id1", "chunk_id2"]
    }}
  ]
}}
"""
