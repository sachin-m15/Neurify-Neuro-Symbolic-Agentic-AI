class BaselineRAGAgent:
    def __init__(self, retriever, claim_generator):
        self.retriever = retriever
        self.gen = claim_generator

    def run(self, question: str) -> dict:
        retrieved_with_scores = self.retriever.retrieve(query=question, k=10)
        retrieved = [chunk for chunk, score in retrieved_with_scores]
        generated = self.gen.generate(question=question, evidence_chunks=retrieved)

        return {
            "status": "GENERATED",
            "answer": generated.short_answer,
            "claims": [c.model_dump() for c in generated.claims],
            "retrieved_chunks": [{"chunk": c.model_dump(), "score": s} for c, s in retrieved_with_scores],
        }