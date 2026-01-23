"""
PURPOSE:
- Run Verified RAG queries directly from terminal
- Useful for debugging, evaluation, and interviews
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

load_dotenv()

def main():
    # Load FAISS index
    with open("data/indexes/store.pkl", "rb") as f:
        store = pickle.load(f)

    # Build agent components
    llm = ChatLLM()
    baseline_retriever = Retriever(store)  # No enhancements for baseline
    verified_retriever = Retriever(store, llm=llm)  # With query expansion and reranking
    generator = ClaimGenerator(llm)
    entailment = EntailmentScorer()
    verifier = Verifier(entailment, support_threshold=0.75)

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

    print("\n=== Verified PDF RAG (CLI Mode) ===")
    print("Type a question. Press ENTER on empty line to exit.\n")

    while True:
        question = input("Ask> ").strip()

        if not question:
            print("Exiting.")
            break

        # Run both baseline and verified agents
        baseline_result = baseline_agent.run(question)
        verified_result = verified_agent.run(question)

        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        # Show Baseline RAG
        print("\n🔍 BASELINE RAG (Standard Retrieval + Generation):")
        print(f"   Status: {baseline_result['status']}")
        print(f"   Answer: {baseline_result['answer']}")
        print(f"   Claims Generated: {len(baseline_result['claims'])}")

        # Show Verified RAG
        print("\n🛡️ VERIFIED RAG (Enhanced Retrieval + Neuro-Symbolic Verification):")
        print(f"   Status: {verified_result['status']}")
        print(f"   Answer: {verified_result['answer']}")
        print(f"   Claims Generated: {len(verified_result['claims'])}")

        # Show verification details with scores
        if "verification" in verified_result:
            verification = verified_result["verification"]
            print(f"\n🔍 VERIFICATION RESULTS:")
            print(f"   Overall Status: {verification['overall_status']}")
            supported = sum(1 for cr in verification["claim_reports"] if cr["status"] == "SUPPORTED")
            total = len(verification["claim_reports"])
            print(f"   Claims Supported: {supported}/{total}")

            print("   Claim Details:")
            for cr in verification["claim_reports"]:
                status_icon = "✅" if cr['status'] == "SUPPORTED" else "❌"
                print(f"     {status_icon} {cr['claim_id']}: {cr['status']} (Score: {cr['score']:.3f})")
                if cr.get("reasons"):
                    print(f"       Reasons: {', '.join(cr['reasons'])}")

        # Compare answers
        print(f"\n⚖️ COMPARISON:")
        if baseline_result['answer'] != verified_result['answer']:
            print("   ❌ Answers differ - potential hallucination detected!")
            print(f"      Baseline: {baseline_result['answer']}")
            print(f"      Verified: {verified_result['answer']}")
        else:
            print("   ✅ Answers match")

        if verified_result['status'] == 'REFUSE':
            print("   🛑 Verified RAG refused to answer - insufficient evidence")
        elif verified_result['status'] == 'PASS':
            print("   ✅ Verified RAG passed verification - answer is trustworthy")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
