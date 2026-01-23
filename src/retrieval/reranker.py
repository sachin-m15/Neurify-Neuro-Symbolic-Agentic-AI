from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.core.models import Chunk

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks

        pairs = [[query, chunk.text] for chunk in chunks]
        inputs = self.tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze().cpu().numpy()

        if scores.ndim == 0:
            scores = [scores]

        # Sort chunks by scores descending
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks]
