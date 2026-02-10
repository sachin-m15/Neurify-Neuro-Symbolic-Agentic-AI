from src.core.models import GeneratedAnswer, VerificationReport, ClaimVerification
from src.verification.rules import check_citation_present, check_citation_exists
from src.verification.numeric_check import numeric_match
import logging

logger = logging.getLogger(__name__)

class Verifier:
    def __init__(self, entailment_scorer, support_threshold: float = 0.80):
        self.entailment = entailment_scorer
        self.support_threshold = support_threshold

    def verify(self, generated: GeneratedAnswer, retrieved_chunks: list) -> VerificationReport:
        logger.debug(f"[DEBUG] Starting verification with threshold: {self.support_threshold}")
        chunk_map = {c.chunk_id: c for c in retrieved_chunks}
        retrieved_ids = set(chunk_map.keys())
        logger.debug(f"[DEBUG] Retrieved chunk IDs: {retrieved_ids}")

        claim_reports = []
        all_supported = True

        for claim in generated.claims:
            logger.debug(f"[DEBUG] Processing claim {claim.claim_id}: {claim.text[:100]}...")
            logger.debug(f"[DEBUG] Claim citations: {claim.citations}")
            reasons = []
            evidence_used = []
            best_score = 0.0

            ok, msg = check_citation_present(claim)
            logger.debug(f"[DEBUG] Citation present check: {ok}, {msg}")
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
            logger.debug(f"[DEBUG] Citation exists check: {ok}, {msg}")
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
            citations_scored = 0
            for cid in claim.citations:
                ev_text = chunk_map[cid].text

                num_ok, num_msg = numeric_match(claim.text, ev_text)
                logger.debug(f"[DEBUG] Numeric check for chunk {cid}: {num_ok}, {num_msg}")
                if not num_ok:
                    reasons.append(f"[numeric] {num_msg}")
                    continue

                s = self.entailment.score(claim.text, ev_text)
                citations_scored += 1
                logger.debug(f"[DEBUG] Entailment score for chunk {cid}: {s:.4f}")
                if s > best_score:
                    best_score = s
                    best_chunk_id = cid

            logger.debug(f"[DEBUG] Citations scored: {citations_scored}, best_score: {best_score:.4f}, best_chunk_id: {best_chunk_id}")

            if best_chunk_id is not None:
                evidence_used = [best_chunk_id]

            if best_score >= self.support_threshold:
                status = "SUPPORTED"
                # Keep numeric check failures as warnings, don't clear them
            else:
                status = "UNSUPPORTED"
                all_supported = False
                reasons.append(f"Entailment score too low: {best_score:.2f} (threshold: {self.support_threshold})")

            logger.debug(f"[DEBUG] Claim {claim.claim_id} final status: {status}, score: {best_score:.4f}")

            claim_reports.append(ClaimVerification(
                claim_id=claim.claim_id,
                status=status,
                evidence_chunks=evidence_used,
                score=float(best_score),
                reasons=reasons
            ))

        overall = "PASS" if all_supported else "RETRY"
        logger.debug(f"[DEBUG] Final overall_status: {overall}, all_supported: {all_supported}")
        return VerificationReport(overall_status=overall, claim_reports=claim_reports)
