# Verified Agentic RAG

A verified Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents with agentic verification.


```
verified-rag-pdf/
│
├── app.py
├── api.py                           # optional (FastAPI)
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