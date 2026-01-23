import os
import time
import tiktoken
from openai import OpenAI

class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        # Use tiktoken for token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")  # For text-embedding-3 models

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Batch to avoid token limit (max 300k tokens per request)
        max_tokens_per_batch = 200000  # Leave more buffer
        embeddings = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = len(self.encoding.encode(text))

            # If single text exceeds limit, process it alone
            if text_tokens > max_tokens_per_batch:
                if current_batch:
                    # Process current batch first
                    resp = self.client.embeddings.create(
                        model=self.model,
                        input=current_batch
                    )
                    embeddings.extend([item.embedding for item in resp.data])
                    current_batch = []
                    current_tokens = 0
                    time.sleep(1)
                # Process large text alone
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                embeddings.extend([item.embedding for item in resp.data])
                time.sleep(1)
                continue

            # Check if adding this text would exceed limit
            if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
                # Process current batch
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=current_batch
                )
                embeddings.extend([item.embedding for item in resp.data])
                current_batch = []
                current_tokens = 0
                time.sleep(1)  # Rate limit: 1 second between batches

            current_batch.append(text)
            current_tokens += text_tokens

        # Process remaining batch
        if current_batch:
            resp = self.client.embeddings.create(
                model=self.model,
                input=current_batch
            )
            embeddings.extend([item.embedding for item in resp.data])

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def __getstate__(self):
        return {'model': self.model}

    def __setstate__(self, state):
        self.model = state['model']
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.get_encoding("cl100k_base")
