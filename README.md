# Neurofy — Verification-First Agentic RAG

   A neuro-symbolic verification engine that governs when Retrieval-Augmented Generation (RAG) systems are allowed to answer, retry, or refuse — based on provable evidence from documents.

---

## 📌 Overview

Neurofy is a Verified Agentic RAG system designed to reduce hallucinations in document-based AI systems.
Instead of always generating answers, Neurofy verifies every claim using a combination of:

- **Neural reasoning** (transformer-based entailment models)
- **Symbolic reasoning** (rule-based constraints and checks)

If claims cannot be verified, the system retries retrieval or refuses to answer.

This makes Neurofy suitable for safety-critical domains such as:

- WHO guidelines
- healthcare policies
- legal and regulatory documents
- technical standards

---

## 🎯 Key Idea (What Makes This Different)
**Answers are not generated freely. They are permitted only after verification.**

Without verification, this system reduces to a basic RAG pipeline.
With verification, it becomes a decision-making, agentic AI system.

---

## 🧠 Core Capabilities

- ✅ Claim-level verification (not paragraph-level)
- ✅ Neuro-symbolic reasoning (rules + neural models)
- ✅ Evidence-backed refusal when information is insufficient
- ✅ Agentic retry loop (retrieve → verify → retry → decide)
- ✅ Numeric and citation consistency checks
- ✅ Designed for factual, policy, and guideline documents
 
---

## 🏗️ Architecture
```
User Query
   ↓
Semantic Retrieval (FAISS)
   ↓
Claim Generation (LLM, structured JSON)
   ↓
Neuro-Symbolic Verification
   ├── Symbolic rules (citations, numbers)
   └── Neural entailment (NLI models)
   ↓
Agentic Decision
   ├── PASS → Answer returned
   ├── RETRY → Focused re-retrieval
   └── REFUSE → Safe refusal
```
---

## 🔍 Neuro-Symbolic Verification Explained

Neurofy combines two reasoning paradigms:

### 🧠 Neural Reasoning

- Transformer-based Natural Language Inference (NLI)
- Measures whether evidence entails a claim
- Provides semantic support scores

### 📐 Symbolic Reasoning

- Hard logical constraints:
  - Every claim must have citations
  - Citations must exist in retrieved evidence
  - Numeric values must match exactly
- Enforces refusal when rules are violated

A claim is accepted only if it passes both.

---

## 📁Project Structure
```
verified-rag-pdf/
│
├── app.py
├── api.py                           
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   ├── raw_pdfs/
│   ├── extracted_text/
│   ├── chunks/
│   └── indexes/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── __init__.py
│
│   ├── core/
│   │   ├── llm.py                   # LLM wrapper
│   │   ├── embeddings.py            # embedding wrapper
│   │   ├── models.py                # Pydantic models (Claim, Chunk, Report)
│   │   ├── utils.py                 # helpers (hashing, text cleanup)
│   │   └── logger.py
│
│   ├── ingest/
│   │   ├── pdf_loader.py            # text extraction + OCR fallback
│   │   ├── cleaner.py               # cleanup rules
│   │   └── chunker.py               # chunking with metadata/page mapping
│
│   ├── retrieval/
│   │   ├── vector_store.py          # FAISS/Chroma wrapper
│   │   ├── retriever.py             # hybrid retrieve + rerank
│   │   └── reranker.py              # cross-encoder reranker (optional)
│
│   ├── generation/
│   │   ├── claim_generator.py        # LLM -> claims JSON
│   │   └── answer_builder.py         # final answer formatting
│
│   ├── verification/
│   │   ├── rules.py                  # neuro-symbolic hard checks
│   │   ├── entailment.py             # transformer NLI / cross-encoder
│   │   ├── numeric_check.py          # numbers + units checks
│   │   ├── verifier.py               # orchestrates all verification
│   │   └── repair.py                 # failed claims -> focused re-retrieve
│
│   ├── agent/
│   │   ├── orchestrator.py           # full agent loop (retrieve->generate->verify->retry)
│   │   └── prompts.py                # prompts for answer+claims, judge, rewriting
│
│   └── evaluation/
│       ├── testset.py                # create small QA set
│       └── metrics.py                # hallucination rate, support rate
│
└── scripts/
    ├── 01_ingest_pdfs.py
    ├── 02_build_index.py
    └── 03_run_cli.py
```
---

## 🧩 Key Neuro-Symbolic Components
To inspect the neuro-symbolic AI components, focus on these key files in the `src/verification/` and `src/agent/` directories:

## Core Neuro-Symbolic AI Files

### [`src/verification/verifier.py`](src/verification/verifier.py)
- **Main verification orchestrator** that combines neural and symbolic reasoning
- Implements the overall verification logic with entailment scoring and rule checking
- Contains the `Verifier` class that produces the confidence scores you see

### [`src/verification/entailment.py`](src/verification/entailment.py)
- **Neural component**: Uses transformer models (like RoBERTa) for textual entailment
- Computes the entailment scores between claims and evidence chunks
- The `EntailmentScorer` class handles the neural network inference

### [`src/verification/rules.py`](src/verification/rules.py)
- **Symbolic component**: Rule-based checks for citation validity
- Functions like `check_citation_present()` and `check_citation_exists()`
- Implements logical rules for claim validation

### [`src/verification/numeric_check.py`](src/verification/numeric_check.py)
- **Symbolic numeric validation**: Checks if numbers in claims match evidence
- Prevents hallucinations with quantitative data
- Uses regex and comparison logic

### [`src/agent/orchestrator.py`](src/agent/orchestrator.py)
- **High-level orchestration**: Coordinates the entire verification pipeline
- Shows how neural (entailment) and symbolic (rules) components work together
- Handles retry logic based on verification results

## How Neuro-Symbolic AI Works

The system combines:
- **Neural**: Transformer-based entailment scoring (probability that evidence supports claim)
- **Symbolic**: Rule-based validation (citation checks, numeric verification, logical consistency)

The final scores you see are primarily from the neural entailment model, filtered through symbolic rules. The `support_threshold` (default 0.75) determines the pass/fail boundary.

Start with `verifier.py` to understand the overall flow, then dive into `entailment.py` for the neural scoring details.

---
## 🤖 Why This Is an Agentic AI System

Neurofy is agentic by behavior, not by framework.

The system:

- Makes autonomous decisions (answer / retry / refuse)
- Observes verification outcomes
- Adjusts retrieval strategy
- Stops when proof is insufficient

---

## 🚫 When Does the System Refuse?

- Neurofy refuses to answer when:
- Evidence does not exist in the PDFs
- Claims lack valid citations
- Entailment scores fall below threshold
- Numeric facts do not match evidence
- Conflicting documents cannot be safely resolved
- Retry budget is exhausted
- The query requests unsafe or personalized medical advice

This refusal behavior is intentional and desirable.

---

## 📊 Evaluation Philosophy

Neurofy prioritizes trust over coverage.

Metrics include:
- Supported claim rate
- Hallucination reduction vs baseline RAG
- Refusal accuracy
- Entailment confidence distribution

---

## 🧪 Example Queries (WHO Guidelines)
- "Is TB preventive treatment (TPT) recommended for pregnant women living with HIV, and if so, which medicines are considered safe?"
- "What is the role of co-trimoxazole prophylaxis in people living with HIV who have active TB disease?"
- "What are the WHO-recommended four symptom screen for TB among adults and adolescents living with HIV?"
- "What is the recommended duration of TB preventive treatment for people living with HIV?"
- "What is the best time to start antiretroviral therapy for someone newly diagnosed with HIV?"
- "When should antiretroviral therapy be initiated in all HIV-positive individuals according to current WHO guidelines?"
- "What is the primary prevention strategy for mother-to-child transmission of HIV?"
- "In which HIV-positive patients should LF-LAM be used for TB diagnosis in inpatient settings?"
- "What is the best CRP cut-off value for TB screening in HIV-positive patients in rural settings?"
- "When should co-trimoxazole prophylaxis be given to HIV-positive patients with active TB?"
- "When should antiretroviral therapy be initiated in HIV-positive patients diagnosed with TB?"
- "How long is the recommended TB treatment duration for drug-sensitive TB in HIV-positive patients?"
- "What are the four WHO-recommended symptoms for TB screening in HIV-positive adults?"

If answers cannot be verified from the documents, Neurofy will refuse.

--- 

## 🚀 How to Run
1️⃣ Install dependencies
```
pip install -r requirements.txt
```
2️⃣ Add PDFs
```
data/raw_pdfs/
```
3️⃣ Ingest and index
```
python scripts/01_ingest_pdfs.py
python scripts/02_build_index.py
```
4️⃣ Run CLI (recommended for debugging)
```
python scripts/03_run_cli.py
```
5️⃣ Run Streamlit UI
```
streamlit run app.py
```
---
## 🧠 Why This Project Matters

Most RAG systems optimize for fluency.
Neurofy optimizes for correctness and safety.

 This project demonstrates how adding neuro-symbolic constraints transforms RAG from a text generator into a trust-aware decision system.

## 📌 Future Work

- OCR fallback for scanned PDFs
- Version-aware document prioritization
- Contradiction visualization
- Cross-encoder reranking
- Formal evaluation dashboard
- LangGraph-based orchestration (optional)
--- 

## 🏁 Final Note

Neurofy is not a chatbot.
It is a **Neuro-symbolic, agentic verification engine that governs when AI systems are allowed to answer.**
That distinction is the core contribution of this project.
