import re

def extract_numbers(text: str) -> list[str]:
    # Remove commas to handle formatting differences (e.g. "1,000" vs "1000")
    text_clean = text.replace(",", "")
    return re.findall(r"\d+(?:\.\d+)?", text_clean)

def numeric_match(claim_text: str, evidence_text: str) -> tuple[bool, str]:
    claim_nums = set(extract_numbers(claim_text))
    if not claim_nums:
        return True, "No numeric constraint"

    ev_nums = set(extract_numbers(evidence_text))
    missing = claim_nums - ev_nums
    if missing:
        return False, f"Missing numbers in evidence: {sorted(list(missing))}"
    return True, "OK"
