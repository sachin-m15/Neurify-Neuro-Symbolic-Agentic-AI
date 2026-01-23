from typing import List
from src.core.models import Chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.core.llm import ChatLLM

class Retriever:
    def __init__(self, store: VectorStore, reranker: CrossEncoderReranker = None, llm: ChatLLM = None):
        self.store = store
        self.reranker = reranker or CrossEncoderReranker()
        self.llm = llm

    def _expand_query(self, query: str) -> List[str]:
        if not self.llm:
            return [query]
        prompt = f"Generate 2-3 related search queries for: {query}"
        response = self.llm.generate(prompt, max_tokens=100)
        expanded = [q.strip() for q in response.split('\n') if q.strip()]
        return [query] + expanded[:3]  # Include original + up to 3 expanded

    def retrieve(self, query: str, k: int = 10, rerank: bool = True, expand: bool = True, filters: dict = None) -> List[tuple[Chunk, float]]:
        queries = self._expand_query(query) if expand else [query]
        all_chunk_scores = []
        for q in queries:
            # Retrieve more chunks initially for reranking
            initial_k = k * 2 if rerank else k
            chunk_scores = self.store.search(query=q, k=initial_k)
            all_chunk_scores.extend(chunk_scores)
        # Remove duplicates, keep highest score
        seen = {}
        for c, s in all_chunk_scores:
            if c.chunk_id not in seen or s > seen[c.chunk_id][1]:
                seen[c.chunk_id] = (c, s)
        unique_chunk_scores = list(seen.values())
        # Apply filters
        if filters:
            unique_chunk_scores = [(c, s) for c, s in unique_chunk_scores if self._matches_filter(c, filters)]
        if rerank and self.reranker:
            # Reranker returns chunks, assume scores are updated or use original
            reranked_chunks = self.reranker.rerank(query, [c for c, _ in unique_chunk_scores])
            # For simplicity, assign high scores to reranked order
            unique_chunk_scores = [(c, 1.0 - i * 0.01) for i, c in enumerate(reranked_chunks)]
        # Sort by score descending
        unique_chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return unique_chunk_scores[:k]

    def _matches_filter(self, chunk: Chunk, filters: dict) -> bool:
        if 'doc_id' in filters and chunk.doc_id != filters['doc_id']:
            return False
        if 'section' in filters and chunk.section != filters['section']:
            return False
        if 'page' in filters and chunk.page != filters['page']:
            return False
        return True

    def _filter_chunks(self, chunks: List[Chunk], filters: dict) -> List[Chunk]:
        filtered = []
        for c in chunks:
            match = True
            if 'doc_id' in filters and c.doc_id != filters['doc_id']:
                match = False
            if 'section' in filters and c.section != filters['section']:
                match = False
            if 'page' in filters and c.page != filters['page']:
                match = False
            if match:
                filtered.append(c)
        return filtered
