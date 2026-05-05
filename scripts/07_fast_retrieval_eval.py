"""
Fast RAG Retrieval Evaluation Script
Evaluates 50 queries with Recall@k and MRR - no LLM calls for speed
"""

import pickle
import json
import numpy as np
import random
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm import ChatLLM
from src.retrieval.retriever import Retriever

def load_chunks():
    """Load chunks from JSON file"""
    with open("data/chunks/chunks.json", "r") as f:
        return json.load(f)

def load_store():
    """Load FAISS store"""
    with open("data/indexes/store.pkl", "rb") as f:
        return pickle.load(f)

def select_diverse_queries(chunks, num_queries=50):
    """Select diverse queries from different sections"""
    # Group chunks by section (qtype)
    sections = defaultdict(list)
    for chunk in chunks:
        section = chunk.get('section', 'unknown')
        sections[section].append(chunk)

    print(f"Available sections: {list(sections.keys())}")
    for section, items in sections.items():
        print(f"  {section}: {len(items)} chunks")

    # Select queries from different sections
    selected_queries = []
    queries_per_section = max(1, num_queries // len(sections))

    for section, section_chunks in sections.items():
        # Shuffle to get random selection
        random.shuffle(section_chunks)

        # Take queries from this section
        section_queries = min(queries_per_section, len(section_chunks))
        for i in range(section_queries):
            chunk = section_chunks[i]
            query_data = {
                "question": chunk["meta"]["Question"],
                "ground_truth_chunk_id": chunk["chunk_id"],
                "ground_truth_answer": chunk["meta"]["Answer"],
                "section": section
            }
            selected_queries.append(query_data)

    print(f"Selected {len(selected_queries)} diverse queries from {len(sections)} sections")
    return selected_queries[:num_queries]

def evaluate_retrieval_fast():
    """Fast retrieval evaluation without LLM calls"""
    # Load data
    chunks = load_chunks()
    store = load_store()

    # Select diverse test queries
    test_queries = select_diverse_queries(chunks, num_queries=50)

    # Initialize components (without LLM for speed)
    retriever = Retriever(store)  # No LLM for faster retrieval

    # Evaluation metrics
    results = []
    section_stats = defaultdict(list)

    k_values = [1, 3, 5]

    for i, query_data in enumerate(test_queries, 1):
        question = query_data["question"]
        ground_truth_chunk_id = query_data["ground_truth_chunk_id"]
        section = query_data["section"]

        print(f"Evaluating [{i}/{len(test_queries)}] {question[:50]}... (Section: {section})")

        # Retrieve top 5 chunks
        retrieved_with_scores = retriever.retrieve(question, k=5)
        retrieved_chunks = [{"chunk": chunk.model_dump(), "score": score} for chunk, score in retrieved_with_scores]

        # Extract chunk_ids from retrieved
        retrieved_chunk_ids = [chunk["chunk"]["chunk_id"] for chunk in retrieved_chunks]

        # Find rank of ground truth
        if ground_truth_chunk_id in retrieved_chunk_ids:
            rank = retrieved_chunk_ids.index(ground_truth_chunk_id) + 1
            rr = 1.0 / rank  # Reciprocal rank
        else:
            rank = None
            rr = 0.0

        # Recall@k
        recall_scores = {}
        for k in k_values:
            recall_scores[f"Recall@{k}"] = 1.0 if ground_truth_chunk_id in retrieved_chunk_ids[:k] else 0.0

        result = {
            "question": question,
            "section": section,
            "rank": rank,
            "rr": rr,
            **recall_scores
        }

        results.append(result)
        section_stats[section].append(result)

        print(f"  Rank: {rank}, RR: {rr:.3f}, Recall@1: {recall_scores['Recall@1']}, Recall@3: {recall_scores['Recall@3']}")

    # Aggregate results
    avg_rr = np.mean([r["rr"] for r in results])
    avg_recall = {k: np.mean([r[k] for r in results]) for k in recall_scores.keys()}

    print("\n=== COMPREHENSIVE RESULTS ===")
    print(f"Total Queries Evaluated: {len(results)}")
    print(f"MRR: {avg_rr:.3f}")
    for k, v in avg_recall.items():
        print(f"{k}: {v:.3f}")

    # Section-wise performance
    print("\n=== SECTION-WISE PERFORMANCE ===")
    for section, section_results in section_stats.items():
        if section_results:
            section_mrr = np.mean([r["rr"] for r in section_results])
            section_recall1 = np.mean([r["Recall@1"] for r in section_results])
            section_recall3 = np.mean([r["Recall@3"] for r in section_results])
            print(f"{section}: {len(section_results)} queries, MRR: {section_mrr:.3f}, Recall@1: {section_recall1:.3f}, Recall@3: {section_recall3:.3f}")

    # Save detailed results
    output_file = "data/retrieval_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_queries": len(results),
                "mrr": avg_rr,
                "recall_scores": avg_recall,
                "sections_covered": list(section_stats.keys())
            },
            "section_stats": {section: {
                "count": len(results),
                "mrr": np.mean([r["rr"] for r in results]),
                "recall@1": np.mean([r["Recall@1"] for r in results]),
                "recall@3": np.mean([r["Recall@3"] for r in results])
            } for section, results in section_stats.items()},
            "detailed_results": results
        }, f, indent=2)

    print(f"\nDetailed results saved to {output_file}")
    return results

if __name__ == "__main__":
    evaluate_retrieval_fast()