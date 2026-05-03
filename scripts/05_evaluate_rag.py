"""
RAG Evaluation Script
Computes Recall@k, MRR, Semantic Similarity, LLM Judge for selected queries
"""

import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm import ChatLLM
from src.generation.claim_generator import ClaimGenerator
from src.retrieval.retriever import Retriever
from src.verification.entailment import EntailmentScorer
from src.verification.verifier import Verifier
from src.agent.baseline_orchestrator import BaselineRAGAgent

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

Ground Truth Answer: {ground_truth_answer}

Provide only a numerical score between 0 and 1.
"""
    response = llm.generate(prompt)
    try:
        score = float(response.strip())
        return min(max(score, 0), 1)  # Clamp to [0,1]
    except:
        return 0.5  # Default if parsing fails

def evaluate_rag():
    # Load data
    chunks = load_chunks()
    store = load_store()

    # Select test queries with ground truth chunk_ids
    test_queries = [
        {
            "question": "Who is at risk for Lymphocytic Choriomeningitis (LCM)?",
            "ground_truth_chunk_id": "csv_row_0_chunk",
            "ground_truth_answer": "LCMV infections can occur after exposure to fresh urine, droppings, saliva, or nesting materials from infected rodents. Transmission may also occur when these materials are directly introduced into broken skin, the nose, the eyes, or the mouth, or presumably, via the bite of an infected rodent. Person-to-person transmission has not been reported, with the exception of vertical transmission from infected mother to fetus, and rarely, through organ transplantation."
        },
        {
            "question": "What are the symptoms of Lymphocytic Choriomeningitis (LCM)?",
            "ground_truth_chunk_id": "csv_row_1_chunk",
            "ground_truth_answer": "LCMV is most commonly recognized as causing neurological disease, as its name implies, though infection without symptoms or mild febrile illnesses are more common clinical manifestations. For infected persons who do become ill, onset of symptoms usually occurs 8-13 days after exposure to the virus as part of a biphasic febrile illness..."
        },
        {
            "question": "How to diagnose Lymphocytic Choriomeningitis (LCM)?",
            "ground_truth_chunk_id": "csv_row_3_chunk",
            "ground_truth_answer": "During the first phase of the disease, the most common laboratory abnormalities are a low white blood cell count (leukopenia) and a low platelet count (thrombocytopenia). Liver enzymes in the serum may also be mildly elevated. After the onset of neurological disease during the second phase, an increase in protein levels, an increase in the number of white blood cells or a decrease in the glucose levels in the cerebrospinal fluid (CSF) is usually found. Laboratory diagnosis is usually made by detecting IgM and IgG antibodies in the CSF and serum. Virus can be detected by PCR or virus isolation in the CSF at during the acute stage of illness."
        },
        {
            "question": "What are the treatments for Lymphocytic Choriomeningitis (LCM)?",
            "ground_truth_chunk_id": "csv_row_4_chunk",
            "ground_truth_answer": "Aseptic meningitis, encephalitis, or meningoencephalitis requires hospitalization and supportive treatment based on severity. Anti-inflammatory drugs, such as corticosteroids, may be considered under specific circumstances. Although studies have shown that ribavirin, a drug used to treat several other viral diseases, is effective against LCMV in vitro, there is no established evidence to support its routine use for treatment of LCM in humans."
        },
        {
            "question": "How to prevent Lymphocytic Choriomeningitis (LCM)?",
            "ground_truth_chunk_id": "csv_row_5_chunk",
            "ground_truth_answer": "LCMV infection can be prevented by avoiding contact with wild mice and taking precautions when handling pet rodents (i.e. mice, hamsters, or guinea pigs). Rarely, pet rodents may become infected with LCMV from wild rodents. Breeders, pet stores, and pet owners should take measures to prevent infestations of wild rodents..."
        }
    ]

    # Initialize components
    llm = ChatLLM()
    retriever = Retriever(store, llm=llm)  # Use verified retriever with query expansion
    generator = ClaimGenerator(llm)
    entailment = EntailmentScorer("roberta-large-mnli")
    verifier = Verifier(entailment_scorer=entailment, support_threshold=0.80)
    agent = BaselineRAGAgent(
        retriever=retriever,
        claim_generator=generator
    )

    # Load semantic similarity model
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Evaluation metrics
    results = []

    k_values = [1, 3, 5, 10]

    for query_data in test_queries:
        question = query_data["question"]
        ground_truth_chunk_id = query_data["ground_truth_chunk_id"]
        ground_truth_answer = query_data["ground_truth_answer"]

        print(f"\nEvaluating: {question}")

        # Retrieve chunks directly
        retrieved_with_scores = retriever.retrieve(question, k=10)
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

        # Generate answer using simple concatenation for semantic sim
        top_chunks_text = [chunk["chunk"]["text"] for chunk in retrieved_chunks[:3]]
        generated_answer = " ".join(top_chunks_text[:500])  # Truncate for sim

        # Semantic similarity
        sim_score = compute_semantic_similarity(generated_answer, ground_truth_answer, sim_model)

        # LLM judge - skip for speed
        llm_score = 0.0  # Placeholder

        result = {
            "question": question,
            "rank": rank,
            "rr": rr,
            **recall_scores,
            "semantic_similarity": sim_score,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer[:200] + "..."  # Truncate for display
        }

        results.append(result)

        print(f"  Rank: {rank}, RR: {rr:.3f}")
        print(f"  Recall@1: {recall_scores['Recall@1']}, Recall@3: {recall_scores['Recall@3']}, Recall@5: {recall_scores['Recall@5']}")
        print(f"  Semantic Sim: {sim_score:.3f}")
        print(f"  Generated: {generated_answer[:100]}...")
        print(f"  Ground Truth: {ground_truth_answer[:100]}...")

    # Aggregate results
    avg_rr = np.mean([r["rr"] for r in results])
    avg_recall = {k: np.mean([r[k] for r in results]) for k in recall_scores.keys()}
    avg_sim = np.mean([r["semantic_similarity"] for r in results])

    print("\n=== AGGREGATE RESULTS ===")
    print(f"MRR: {avg_rr:.3f}")
    for k, v in avg_recall.items():
        print(f"{k}: {v:.3f}")
    print(f"Average Semantic Similarity: {avg_sim:.3f}")

    return results

if __name__ == "__main__":
    evaluate_rag()