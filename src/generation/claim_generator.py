from src.core.models import GeneratedAnswer, Claim
from src.agent.prompts import SYSTEM_CLAIM_GENERATION, build_user_prompt

class ClaimGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question: str, evidence_chunks: list) -> GeneratedAnswer:
        payload = self.llm.json(
            system=SYSTEM_CLAIM_GENERATION,
            user=build_user_prompt(question, evidence_chunks),
        )

        claims = []
        for c in payload.get("claims", []):
            claims.append(
                Claim(
                    claim_id=c.get("claim_id", ""),
                    text=c.get("text", ""),
                    citations=c.get("citations", []),
                )
            )

        return GeneratedAnswer(
            short_answer=payload.get("short_answer", ""),
            claims=claims
        )
