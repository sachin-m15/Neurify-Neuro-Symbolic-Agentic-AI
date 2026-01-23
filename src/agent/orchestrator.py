from src.verification.repair import build_retry_query

class VerifiedRAGAgent:
    def __init__(self, retriever, claim_generator, verifier, max_retries: int = 2):
        self.retriever = retriever
        self.gen = claim_generator
        self.verifier = verifier
        self.max_retries = max_retries

    def run(self, question: str) -> dict:
        query = question
        last_generated = None
        last_report = None

        for _ in range(self.max_retries + 1):
            retrieved_with_scores = self.retriever.retrieve(query=query, k=10)
            retrieved = [chunk for chunk, score in retrieved_with_scores]

            generated = self.gen.generate(question=question, evidence_chunks=retrieved)
            report = self.verifier.verify(generated=generated, retrieved_chunks=retrieved)

            last_generated = generated
            last_report = report

            if report.overall_status == "PASS":
                return {
                    "status": "PASS",
                    "answer": generated.short_answer,
                    "claims": [c.model_dump() for c in generated.claims],
                    "verification": report.model_dump(),
                    "retrieved_chunks": [{"chunk": c.model_dump(), "score": s} for c, s in retrieved_with_scores],
                }

            failed_texts = []
            for cr in report.claim_reports:
                if cr.status != "SUPPORTED":
                    for c in generated.claims:
                        if c.claim_id == cr.claim_id:
                            failed_texts.append(c.text)

            if not failed_texts:
                break

            query = build_retry_query(question, failed_texts)

        return {
            "status": "REFUSE",
            "answer": "Not enough verified evidence in the uploaded PDF(s) to answer reliably.",
            "claims": [c.model_dump() for c in last_generated.claims] if last_generated else [],
            "verification": last_report.model_dump() if last_report else {},
        }
