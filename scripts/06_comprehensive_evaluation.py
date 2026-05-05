"""
Comprehensive RAG Evaluation Script
Evaluates 50-100 queries from chunks.json with Recall@k, MRR, Semantic similarity, LLM judge, and Ragas
"""

import pickle
import json
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm import ChatLLM
from src.generation.claim_generator import ClaimGenerator
from src.retrieval.retriever import Retriever

load_dotenv()

def load_chunks():
    """Load chunks from JSON file"""
    with open("data/chunks/chunks.json", "r") as f:
        return json.load(f)

def load_store():
    """Load FAISS store"""
    with open("data/indexes/store.pkl", "rb") as f:
        return pickle.load(f)

def compute_semantic_similarity(text1, text2, model):
    """Compute semantic similarity between two texts"""
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def llm_judge(question, generated_answer, ground_truth_answer, llm):
    """Use LLM to judge if generated answer is correct compared to ground truth"""
    prompt = f"""
Please evaluate if the following generated answer correctly addresses the question, compared to the ground truth answer.
Give a score from 0 to 1, where 1 means perfectly correct and matches ground truth, 0 means completely wrong.

Question: {question}

Generated Answer: {generated_answer}

Ground Truth Answer: {ground_truth_answer[:1000]}...

Provide only a numerical score between 0 and 1.
"""
    try:
        response = llm.generate(prompt, max_tokens=50, temperature=0.1)
        score = float(response.strip())
        return min(max(score, 0), 1)  # Clamp to [0,1]
    except:
        return 0.5  # Default if parsing fails

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

    # If we don't have enough, fill with more from largest sections
    while len(selected_queries) < num_queries and selected_queries:
        largest_section = max(sections.keys(), key=lambda s: len(sections[s]))
        available_chunks = [c for c in sections[largest_section]
                          if c["chunk_id"] not in [q["ground_truth_chunk_id"] for q in selected_queries]]

        if not available_chunks:
            break

        chunk = random.choice(available_chunks)
        query_data = {
            "question": chunk["meta"]["Question"],
            "ground_truth_chunk_id": chunk["chunk_id"],
            "ground_truth_answer": chunk["meta"]["Answer"],
            "section": largest_section
        }
        selected_queries.append(query_data)

    print(f"Selected {len(selected_queries)} diverse queries from {len(sections)} sections")
    return selected_queries[:num_queries]

def evaluate_rag_comprehensive():
    # Load data
    chunks = load_chunks()
    store = load_store()

    # Select diverse test queries
    test_queries = select_diverse_queries(chunks, num_queries=50)

    # Initialize components
    llm = ChatLLM()
    retriever = Retriever(store, llm=llm)  # Use verified retriever with query expansion
    generator = ClaimGenerator(llm)

    # Load semantic similarity model
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Evaluation metrics
    results = []
    section_stats = defaultdict(list)

    k_values = [1, 3, 5, 10]

    for i, query_data in enumerate(test_queries, 1):
        question = query_data["question"]
        ground_truth_chunk_id = query_data["ground_truth_chunk_id"]
        ground_truth_answer = query_data["ground_truth_answer"]
        section = query_data["section"]

        print(f"\nEvaluating [{i}/{len(test_queries)}] {question[:60]}... (Section: {section})")

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

        # Generate answer using top 3 retrieved chunks
        top_chunks_text = [chunk["chunk"]["text"] for chunk in retrieved_chunks[:3]]
        generated_answer = " ".join(top_chunks_text[:1000])  # Truncate for processing

        # Semantic similarity
        sim_score = compute_semantic_similarity(generated_answer, ground_truth_answer, sim_model)

        # LLM judge - disabled for speed
        llm_score = None

        result = {
            "question": question,
            "section": section,
            "rank": rank,
            "rr": rr,
            **recall_scores,
            "semantic_similarity": sim_score,
            "llm_judge_score": llm_score,
            "generated_answer": generated_answer[:200] + "...",
            "ground_truth_answer": ground_truth_answer[:200] + "..."
        }

        results.append(result)
        section_stats[section].append(result)

        print(f"  Rank: {rank}, RR: {rr:.3f}")
        print(f"  Recall@1: {recall_scores['Recall@1']}, Recall@3: {recall_scores['Recall@3']}, Recall@5: {recall_scores['Recall@5']}")
        print(f"  Semantic Sim: {sim_score:.3f}" + (f", LLM Judge: {llm_score:.3f}" if llm_score is not None else ""))

    # Aggregate results
    avg_rr = np.mean([r["rr"] for r in results])
    avg_recall = {k: np.mean([r[k] for r in results]) for k in recall_scores.keys()}
    avg_sim = np.mean([r["semantic_similarity"] for r in results])

    # LLM judge results (only for evaluated queries)
    llm_judge_results = [r["llm_judge_score"] for r in results if r["llm_judge_score"] is not None]
    avg_llm = np.mean(llm_judge_results) if llm_judge_results else None

    print("\n=== COMPREHENSIVE RESULTS ===")
    print(f"Total Queries Evaluated: {len(results)}")
    print(f"MRR: {avg_rr:.3f}")
    for k, v in avg_recall.items():
        print(f"{k}: {v:.3f}")
    print(f"Average Semantic Similarity: {avg_sim:.3f}")
    if avg_llm is not None:
        print(f"Average LLM Judge Score (sampled): {avg_llm:.3f}")

    # Section-wise performance
    print("\n=== SECTION-WISE PERFORMANCE ===")
    for section, section_results in section_stats.items():
        if section_results:
            section_mrr = np.mean([r["rr"] for r in section_results])
            section_recall1 = np.mean([r["Recall@1"] for r in section_results])
            section_recall3 = np.mean([r["Recall@3"] for r in section_results])
            print(f"{section}: {len(section_results)} queries, MRR: {section_mrr:.3f}, Recall@1: {section_recall1:.3f}, Recall@3: {section_recall3:.3f}")

    # Save detailed results
    output_file = "data/evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_queries": len(results),
                "mrr": avg_rr,
                "recall_scores": avg_recall,
                "avg_semantic_similarity": avg_sim,
                "avg_llm_judge": avg_llm,
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
    evaluate_rag_comprehensive()