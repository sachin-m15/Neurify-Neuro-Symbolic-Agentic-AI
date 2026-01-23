import os
from openai import OpenAI

class ChatLLM:
    def __init__(self, model: str | None = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text response from a single prompt.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def json(self, system: str, user: str) -> dict:
        """
        Returns JSON dict strictly.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return resp.choices[0].message.content and __import__("json").loads(resp.choices[0].message.content)
