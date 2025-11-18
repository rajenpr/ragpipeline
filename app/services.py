from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings and chat completions using LangChain with Portkey routing."""

    def __init__(self):
        """Initialize LangChain LLM and embeddings with Portkey configuration."""
        # Initialize Chat LLM with Portkey routing
        self.llm = ChatOpenAI(
            api_key=settings.effective_openai_api_key,
            base_url=settings.portkey_base_url,
            default_headers=settings.portkey_chat_headers,
            model=settings.chat_model,
            temperature=0.7,
            max_tokens=500
        )

        # Initialize Embeddings with Portkey routing
        self.embeddings_model = OpenAIEmbeddings(
            api_key=settings.effective_openai_api_key,
            base_url=settings.portkey_base_url,
            default_headers=settings.portkey_embedding_headers,
            model=settings.embedding_model
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
        return self.embeddings_model.embed_query(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings_model.embed_documents(texts)

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response based on query and retrieved context.

        Args:
            query: The user's query
            context: The retrieved context from documents

        Returns:
            Generated response string
        """
        # Create chat prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the context to answer the question accurately. If the context doesn't contain enough information
to answer the question, say so clearly."""),
            ("human", """Context:
{context}

Question: {query}

Answer:""")
        ])

        # Format the prompt with context and query
        prompt = prompt_template.format_messages(context=context, query=query)

        # Invoke the LLM
        response = self.llm.invoke(prompt)

        return response.content


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
