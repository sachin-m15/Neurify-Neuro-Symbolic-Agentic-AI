def supported_claim_rate(verification_report: dict) -> float:
    claims = verification_report.get("claim_reports", [])
    if not claims:
        return 0.0
    ok = sum(1 for c in claims if c.get("status") == "SUPPORTED")
    return ok / len(claims)
