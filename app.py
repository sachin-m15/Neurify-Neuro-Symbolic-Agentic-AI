import pickle
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.core.llm import ChatLLM
from src.generation.claim_generator import ClaimGenerator
from src.retrieval.retriever import Retriever
from src.verification.entailment import EntailmentScorer
from src.verification.verifier import Verifier
from src.agent.orchestrator import VerifiedRAGAgent
from src.agent.baseline_orchestrator import BaselineRAGAgent

load_dotenv()

@st.cache_resource
def load_store():
    with open("data/indexes/store.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_entail():
    from src.verification.entailment import EntailmentScorer
    return EntailmentScorer("roberta-large-mnli")

@st.cache_resource
def load_reranker():
    from src.retrieval.reranker import CrossEncoderReranker
    return CrossEncoderReranker()

@st.cache_resource
def load_agents():
    with open("data/indexes/store.pkl", "rb") as f:
        store = pickle.load(f)

    llm = ChatLLM()
    baseline_retriever = Retriever(store=store)  # No enhancements
    verified_retriever = Retriever(store=store, llm=llm)  # With enhancements
    gen = ClaimGenerator(llm=llm)

    entail = EntailmentScorer("roberta-large-mnli")
    verifier = Verifier(entailment_scorer=entail, support_threshold=support_threshold)

    baseline_agent = BaselineRAGAgent(
        retriever=baseline_retriever,
        claim_generator=gen
    )

    verified_agent = VerifiedRAGAgent(
        retriever=verified_retriever,
        claim_generator=gen,
        verifier=verifier,
        max_retries=2
    )

    return baseline_agent, verified_agent

st.set_page_config(page_title="Neurify", layout="wide", page_icon="🧠")
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5em;
        color: #1f77b4;
        text-align: left;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #666;
        text-align: left;
        margin-bottom: 30px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        text-align: center;
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .success-text { color: #28a745; }
    .error-text { color: #dc3545; }
    .warning-text { color: #ffc107; }
</style>
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.title("🛠️ Settings")
    st.markdown("---")
    support_threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.75, 0.05)
    st.markdown("Adjust the minimum entailment score required for claim verification.")
    st.markdown("---")
    st.markdown("### About Neurify")
    st.write("Neurify uses neuro-symbolic AI to verify RAG responses against document evidence, preventing hallucinations.")
    st.markdown("---")
    st.markdown("**Features:**")
    st.write("• Baseline vs Verified RAG comparison")
    st.write("• Real-time verification")
    st.write("• Detailed claim analysis")
st.title("🧠 Neurify")
st.markdown('<p class="main-header">Hallucination-Proof</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Neuro-symbolic verification engine for trustworthy Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
st.markdown("---")
st.write("💡 Ask questions from your indexed PDFs. Get answers only when claims are verified against evidence.")
st.header("💬 Enter your question here...")
question = st.text_area("", height=80, placeholder="Ask anything from your indexed documents...")

baseline_agent, verified_agent = load_agents()
verified_agent.verifier.support_threshold = support_threshold

if st.button("🚀 Run Analysis", type="primary") and question:
    with st.spinner("🧠 Processing your question with neuro-symbolic verification..."):
        
        # Run both agents
        baseline_result = baseline_agent.run(question)
        verified_result = verified_agent.run(question)

    st.header("⚡ Analysis Results: Baseline vs Verified RAG")
    st.markdown("---")

    # Display question prominently
    st.subheader(f"❓ Question: {question}")
    st.markdown("---")

    # Side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Baseline RAG")
        st.caption("(Standard Retrieval + Generation)")

        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Status", baseline_result["status"])
        with metric_col2:
            st.metric("Answer", "Generated" if baseline_result["answer"] else "None")
        with metric_col3:
            st.metric("Claims", len(baseline_result["claims"]))

        st.write("**Answer:**", baseline_result["answer"])

    with col2:
        st.subheader("🛡️ Verified RAG")
        st.caption("(Enhanced Retrieval + Neuro-Symbolic Verification)")

        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Status", verified_result["status"])
        with metric_col2:
            if verified_result["answer"] == "NOT_ENOUGH_EVIDENCE":
                st.error("❌ No verified info")
            else:
                st.success("✅ Verified")
        with metric_col3:
            st.metric("Claims", len(verified_result["claims"]))

        if verified_result["answer"] == "NOT_ENOUGH_EVIDENCE":
            st.error("❌ No verified information found for this query in the indexed documents.")
        else:
            st.write("**Answer:**", verified_result["answer"])

    st.markdown("---")
    
    st.markdown("---")    
    # Verification Report (full width)
    st.subheader("🔍 Neuro-Symbolic Verification Results")
    with st.expander("🔍 Detailed Neuro-Symbolic Verification", expanded=True):
        verification = verified_result.get("verification", {})
        if verification:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Status", verification.get('overall_status'))
            with col2:
                supported = sum(1 for cr in verification.get("claim_reports", []) if cr["status"] == "SUPPORTED")
                total = len(verification.get("claim_reports", []))
                st.metric("Claims Supported", f"{supported}/{total}")

            st.write("**Claim Verification Details:**")
            for claim in verification.get("claim_reports", []):
                status_icon = "✅" if claim["status"] == "SUPPORTED" else "❌"
                st.write(f"{status_icon} **Claim {claim['claim_id']}:** {claim['status']} (Entailment Score: {claim['score']:.3f})")
                if claim.get("reasons"):
                    st.write("   *Reasons:*", ", ".join(claim["reasons"]))
        else:
            st.json(verification)

    # Comparison Analysis
    st.subheader("⚖️ Comparison Analysis")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.write("**Answer Comparison:**")
        if baseline_result['answer'] != verified_result['answer']:
            st.error("❌ **Answers Differ** - Potential hallucination detected!")
            st.write("**Baseline:**", baseline_result['answer'])
            st.write("**Verified:**", verified_result['answer'])
        else:
            st.success("✅ **Answers Match**")

    with comp_col2:
        st.write("**Trustworthiness:**")
        if verified_result['status'] == 'REFUSE':
            st.warning("🛑 **Verified RAG refused to answer** - insufficient evidence")
        elif verified_result['status'] == 'PASS':
            st.success("✅ **Verified RAG passed verification** - answer is trustworthy")
        else:
            st.info("🔄 **Verification in progress**")

    # Results Comparison Table
    st.subheader("📋 Results Comparison Table")
    results_data = {
        "Metric": ["Status", "Answer Provided", "Claims Generated", "Verification Applied", "Hallucination Risk"],
        "Baseline RAG": [
            baseline_result["status"],
            "Yes" if baseline_result["answer"] else "No",
            str(len(baseline_result["claims"])),
            "No",
            "High"
        ],
        "Verified RAG": [
            verified_result["status"],
            "Yes" if verified_result["answer"] != "NOT_ENOUGH_EVIDENCE" else "No",
            str(len(verified_result["claims"])),
            "Yes" if verified_result.get("verification") else "No",
            "Low" if verified_result.get("verification", {}).get("overall_status") == "PASS" else "Medium"
        ]
    }

    st.table(results_data)
    
    st.subheader("📊 Verification Metrics")
    verification = verified_result.get("verification", {})
    supported = sum(1 for cr in verification.get("claim_reports", []) if cr["status"] == "SUPPORTED") if verification else 0
    chart_data = pd.DataFrame({
        'Metric': ['Generated Claims', 'Supported Claims'],
        'Baseline RAG': [len(baseline_result["claims"]), 0],
        'Verified RAG': [len(verified_result["claims"]), supported]
    })
    st.bar_chart(chart_data.set_index('Metric'))
    
    # Additional details in expanders
    if verified_result.get("status") == "PASS" and verified_result["answer"] != "NOT_ENOUGH_EVIDENCE":
        with st.expander("📋 Claims Breakdown", expanded=False):
            st.json(verified_result.get("claims", []))

        with st.expander("📄 Retrieved Evidence Chunks", expanded=False):
            chunks = verified_result.get("retrieved_chunks", [])
            if chunks:
                for i, item in enumerate(chunks):
                    chunk = item["chunk"]
                    score = item["score"]
                    st.write(f"**Chunk {i+1}:** Retrieval Score: {score:.3f}")
                    st.write(f"**Text:** {chunk['text'][:300]}...")
                    st.write(f"**Source:** {chunk['doc_id']} p.{chunk['page']}")
                    st.divider()
            else:
                st.json(chunks)
st.caption("Made by Sachin")
