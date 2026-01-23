from typing import List
import numpy as np
import faiss

from src.core.models import Chunk

class VectorStore:
    def __init__(self, embedder, dim: int = 3072):
        """
        dim must match embedding model output dimension.
        For text-embedding-3-large -> typically 3072.
        """
        self.embedder = embedder
        self.dim = dim

        # Use IVF for approximate search, better for larger datasets
        nlist = min(100, max(4, len(self.chunks) // 39)) if hasattr(self, 'chunks') and self.chunks else 4  # Rough heuristic
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.chunk_ids: list[str] = []
        self.chunks: dict[str, Chunk] = {}

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def add_chunks(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        mat = np.array(embeddings, dtype=np.float32)
        mat = self._normalize(mat)

        # Train the index if it's IVF
        if not self.index.is_trained:
            self.index.train(mat)

        self.index.add(mat)

        for c in chunks:
            self.chunk_ids.append(c.chunk_id)
            self.chunks[c.chunk_id] = c

    def search(self, query: str, k: int = 8) -> List[tuple[Chunk, float]]:
        q = np.array([self.embedder.embed_query(query)], dtype=np.float32)
        q = self._normalize(q)

        # Set nprobe for IVF search
        self.index.nprobe = min(10, self.index.nlist)

        scores, ids = self.index.search(q, k)
        results: List[tuple[Chunk, float]] = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            chunk_id = self.chunk_ids[i]
            results.append((self.chunks[chunk_id], float(score)))
        return results
