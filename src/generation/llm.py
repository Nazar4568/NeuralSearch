import os
from typing import List
from huggingface_hub import AsyncInferenceClient


class LLMGenerator:
    """Sends the retrieved context and user query to the LLM via official HF SDK."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("HF_API_KEY")

        self.client = AsyncInferenceClient(
            model="openai/gpt-oss-120b",
            token=self.api_key,
            timeout=30.0
        )

    async def generate_answer(
            self,
            query: str,
            context_documents: List[str]
    ) -> str:

        if not self.api_key:
            return "Generated answer is unavailable: HF_API_KEY is not set."

        context_str = "\n\n".join(
            f"Snippet {i + 1}:\n{doc}"
            for i, doc in enumerate(context_documents)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. "
                    "Answer the user's question based ONLY on the provided context. "
                    "If the answer cannot be found in the context, say that "
                    "the provided context does not contain enough information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_str}\n\n"
                    f"Question: {query}"
                )
            }
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"LLM API Error: {str(e)}"