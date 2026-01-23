"""
PURPOSE:
- Central data models shared across the system
- Enforces structure for claims, chunks, and verification

WHY IMPORTANT:
- Verification depends on strict, predictable schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Chunk(BaseModel):
    chunk_id: str  # Unique identifier used for citations
    doc_id: str # Source document identifier
    page: int # Page number (critical for citation trust)
    text: str # Actual chunk text
    section: Optional[str] = None # Optional section heading (future enhancement)
    meta: Dict[str, Any] = {} # Any extra metadata (char spans, etc.)

class Claim(BaseModel):
    claim_id: str # Claim identifier (C1, C2, ...)
    text: str # Atomic factual statement
    citations: List[str]  # List of chunk_ids used as evidence

class GeneratedAnswer(BaseModel):
    short_answer: str # Short natural-language answer
    claims: List[Claim] # Decomposed atomic claims

class ClaimVerification(BaseModel):
    claim_id: str # Which claim was checked
    status: str # SUPPORTED / UNSUPPORTED / RETRY
    evidence_chunks: List[str] # Evidence chunks actually used
    score: float # Entailment confidence score
    reasons: List[str] # Reasons for failure (if any)

class VerificationReport(BaseModel):
    overall_status: str # PASS / RETRY / REFUSE
    claim_reports: List[ClaimVerification] # Per-claim verification results
