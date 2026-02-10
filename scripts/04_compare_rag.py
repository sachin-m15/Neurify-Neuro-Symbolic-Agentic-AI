"""
PURPOSE:
- Compare baseline RAG vs Verified RAG on the same questions
"""

import pickle
from dotenv import load_dotenv
import sys
import json
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm import ChatLLM
from src.generation.claim_generator import ClaimGenerator
from src.retrieval.retriever import Retriever
from src.verification.entailment import EntailmentScorer
from src.verification.verifier import Verifier
from src.agent.orchestrator import VerifiedRAGAgent
from src.agent.baseline_orchestrator import BaselineRAGAgent
from src.evaluation.testset import sample_questions

load_dotenv()

def main():
    # Load FAISS index
    with open("data/indexes/store.pkl", "rb") as f:
        store = pickle.load(f)

    # Build components
    llm = ChatLLM()
    baseline_retriever = Retriever(store)  # No query expansion for baseline
    verified_retriever = Retriever(store, llm=llm)  # With query expansion
    generator = ClaimGenerator(llm)
    entailment = EntailmentScorer()
    verifier = Verifier(entailment, support_threshold=0.80)

    baseline_agent = BaselineRAGAgent(
        retriever=baseline_retriever,
        claim_generator=generator
    )

    verified_agent = VerifiedRAGAgent(
        retriever=verified_retriever,
        claim_generator=generator,
        verifier=verifier,
        max_retries=2
    )

    questions = sample_questions()[:1]  # Use only first question for quick test

    print("\n=== Comparison: Baseline RAG vs Verified RAG ===\n")

    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}\n")

        # Run baseline
        baseline_result = baseline_agent.run(question)
        print("BASELINE RAG:")
        print(f"  Status: {baseline_result['status']}")
        print(f"  Answer: {baseline_result['answer']}")
        print(f"  Claims: {len(baseline_result['claims'])}")

        # Run verified
        verified_result = verified_agent.run(question)
        print("\nVERIFIED RAG:")
        print(f"  Status: {verified_result['status']}")
        print(f"  Answer: {verified_result['answer']}")
        print(f"  Claims: {len(verified_result['claims'])}")

        # Show verification scores
        if "verification" in verified_result:
            verification = verified_result["verification"]
            print(f"  Verification Status: {verification['overall_status']}")
            supported = sum(1 for cr in verification["claim_reports"] if cr["status"] == "SUPPORTED")
            total = len(verification["claim_reports"])
            print(f"  Supported Claims: {supported}/{total}")
            print("  Claim Scores:")
            for cr in verification["claim_reports"]:
                status_icon = "✅" if cr['status'] == "SUPPORTED" else "❌"
                print(f"    {status_icon} {cr['claim_id']}: {cr['status']} (Score: {cr['score']:.3f})")

        # Show differences
        print("\nDIFFERENCES:")
        if baseline_result['answer'] != verified_result['answer']:
            print("  - Answers differ")
            print(f"    Baseline: {baseline_result['answer']}")
            print(f"    Verified: {verified_result['answer']}")
        else:
            print("  - Answers are the same")

        if verified_result['status'] == 'REFUSE':
            print("  - Verified RAG refused to answer due to insufficient verified evidence")
        elif verified_result['status'] == 'PASS':
            print("  - Verified RAG passed verification")

        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()