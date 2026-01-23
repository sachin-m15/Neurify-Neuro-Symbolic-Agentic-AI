from src.core.models import Claim

def check_citation_present(claim: Claim) -> tuple[bool, str]:
    if not claim.citations:
        return False, "No citation provided"
    return True, "OK"

def check_citation_exists(claim: Claim, retrieved_chunk_ids: set[str]) -> tuple[bool, str]:
    for cid in claim.citations:
        if cid not in retrieved_chunk_ids:
            return False, f"Invalid citation: {cid} not in retrieved chunks"
    return True, "OK"
