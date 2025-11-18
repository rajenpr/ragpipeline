"""
Example client for RAG Pipeline API

This module demonstrates how to interact with the RAG Pipeline API
from your Python applications.
"""

import requests
from typing import List, Dict, Optional


class RAGClient:
    """Client for interacting with RAG Pipeline API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the RAG client.

        Args:
            base_url: Base URL of the RAG API
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict:
        """
        Check if the API is healthy.

        Returns:
            Health status response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def load_document(self, content: str, metadata: Optional[str] = None) -> Dict:
        """
        Load a single document.

        Args:
            content: Document text content
            metadata: Optional metadata as JSON string

        Returns:
            Load response with document ID
        """
        return self.load_documents([{"content": content, "metadata": metadata}])

    def load_documents(self, documents: List[Dict[str, str]]) -> Dict:
        """
        Load multiple documents.

        Args:
            documents: List of documents with 'content' and optional 'metadata'

        Returns:
            Load response with document IDs
        """
        payload = {"documents": documents}
        response = requests.post(f"{self.base_url}/load", json=payload)
        response.raise_for_status()
        return response.json()

    def query(
        self,
        query: str,
        top_k: int = 5,
        generate_response: bool = True
    ) -> Dict:
        """
        Query the document store.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            generate_response: Whether to generate AI response

        Returns:
            Query response with retrieved documents and optional AI response
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "generate_response": generate_response
        }
        response = requests.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> Dict:
        """
        Get statistics about stored documents.

        Returns:
            Statistics response
        """
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the RAG client."""

    # Initialize client
    client = RAGClient("http://localhost:8000")

    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"✅ API is {health['status']}")

    # Load some documents
    print("\nLoading documents...")
    load_result = client.load_documents([
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": '{"category": "programming", "language": "python"}'
        },
        {
            "content": "Machine learning is a subset of AI that enables systems to learn from data.",
            "metadata": '{"category": "AI", "topic": "machine learning"}'
        },
        {
            "content": "Neural networks are computing systems inspired by biological neural networks.",
            "metadata": '{"category": "AI", "topic": "neural networks"}'
        }
    ])
    print(f"✅ Loaded {load_result['count']} documents")
    print(f"   Document IDs: {load_result['document_ids']}")

    # Get stats
    stats = client.get_stats()
    print(f"\n📊 Total documents in database: {stats['total_documents']}")

    # Query the documents
    print("\nQuerying: 'What is Python?'")
    result = client.query(
        query="What is Python?",
        top_k=3,
        generate_response=True
    )

    print(f"\n🔍 Retrieved {len(result['retrieved_documents'])} documents:")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"\n  {i}. (Similarity: {doc['similarity_score']:.4f})")
        print(f"     {doc['content'][:80]}...")

    if result['generated_response']:
        print(f"\n🤖 AI Response:")
        print(f"   {result['generated_response']}")


if __name__ == "__main__":
    main()
