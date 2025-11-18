from typing import List
from portkey_ai import Portkey
from app.config import get_settings
import numpy as np

settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings and chat completions using Portkey."""

    def __init__(self):
        """Initialize Portkey client."""
        self.client = Portkey(
            api_key=settings.portkey_api_key,
            virtual_key=settings.portkey_virtual_key
        )
        self.embedding_model = settings.embedding_model
        self.chat_model = settings.chat_model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [item.embedding for item in response.data]

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response based on query and retrieved context.

        Args:
            query: The user's query
            context: The retrieved context from documents

        Returns:
            Generated response string
        """
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use the context to answer the question accurately. If the context doesn't contain enough information
to answer the question, say so clearly."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
