from src.core.models import GeneratedAnswer, VerificationReport, ClaimVerification
from src.verification.rules import check_citation_present, check_citation_exists
from src.verification.numeric_check import numeric_match

class Verifier:
    def __init__(self, entailment_scorer, support_threshold: float = 0.75):
        self.entailment = entailment_scorer
        self.support_threshold = support_threshold

    def verify(self, generated: GeneratedAnswer, retrieved_chunks: list) -> VerificationReport:
        chunk_map = {c.chunk_id: c for c in retrieved_chunks}
        retrieved_ids = set(chunk_map.keys())

        claim_reports = []
        all_supported = True

        for claim in generated.claims:
            reasons = []
            evidence_used = []
            best_score = 0.0

            ok, msg = check_citation_present(claim)
            if not ok:
                all_supported = False
                claim_reports.append(ClaimVerification(
                    claim_id=claim.claim_id,
                    status="UNSUPPORTED",
                    evidence_chunks=[],
                    score=0.0,
                    reasons=[msg]
                ))
                continue

            ok, msg = check_citation_exists(claim, retrieved_ids)
            if not ok:
                all_supported = False
                claim_reports.append(ClaimVerification(
                    claim_id=claim.claim_id,
                    status="UNSUPPORTED",
                    evidence_chunks=[],
                    score=0.0,
                    reasons=[msg]
                ))
                continue

            best_chunk_id = None
            for cid in claim.citations:
                ev_text = chunk_map[cid].text

                num_ok, num_msg = numeric_match(claim.text, ev_text)
                if not num_ok:
                    reasons.append(f"[numeric] {num_msg}")
                    continue

                s = self.entailment.score(claim.text, ev_text)
                if s > best_score:
                    best_score = s
                    best_chunk_id = cid

            if best_chunk_id is not None:
                evidence_used = [best_chunk_id]

            if best_score >= self.support_threshold:
                status = "SUPPORTED"
                reasons = []  # Clear previous failure reasons if ultimately supported
            else:
                status = "UNSUPPORTED"
                all_supported = False
                reasons.append(f"Entailment score too low: {best_score:.2f}")

            claim_reports.append(ClaimVerification(
                claim_id=claim.claim_id,
                status=status,
                evidence_chunks=evidence_used,
                score=float(best_score),
                reasons=reasons
            ))

        overall = "PASS" if all_supported else "RETRY"
        return VerificationReport(overall_status=overall, claim_reports=claim_reports)
