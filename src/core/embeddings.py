import os
import time
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np

class GeminiEmbedder:
    def __init__(self):
        # Use local SentenceTransformer model to avoid API limits
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # SentenceTransformer embeddings have 384 dimensions
        self.dim = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384



