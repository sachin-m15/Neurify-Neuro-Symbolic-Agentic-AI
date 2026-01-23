def build_retry_query(original_question: str, failed_claim_texts: list[str]) -> str:
    extra = "\n".join([f"- {c}" for c in failed_claim_texts])
    return f"{original_question}\n\nFind evidence for these missing claims:\n{extra}"
