#!/usr/bin/env python3
"""
Test script for RAG Pipeline API

This script demonstrates how to use the RAG Pipeline API to:
1. Load sample documents
2. Query the documents
3. Get AI-generated responses
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_health():
    """Test the health endpoint."""
    print_section("Testing Health Endpoint")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def load_documents():
    """Load sample documents into the system."""
    print_section("Loading Sample Documents")

    sample_docs = {
        "documents": [
            {
                "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
                "metadata": json.dumps({"source": "documentation", "topic": "FastAPI"})
            },
            {
                "content": "PgVector is a PostgreSQL extension for vector similarity search. It supports exact and approximate nearest neighbor search, L2 distance, inner product, and cosine distance.",
                "metadata": json.dumps({"source": "documentation", "topic": "PgVector"})
            },
            {
                "content": "Portkey is an AI gateway that provides a unified API to interact with multiple LLM providers. It offers features like load balancing, fallbacks, and observability.",
                "metadata": json.dumps({"source": "documentation", "topic": "Portkey"})
            },
            {
                "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from a knowledge base before generating responses.",
                "metadata": json.dumps({"source": "documentation", "topic": "RAG"})
            },
            {
                "content": "Vector embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings in vector space.",
                "metadata": json.dumps({"source": "documentation", "topic": "Embeddings"})
            }
        ]
    }

    response = requests.post(f"{BASE_URL}/load", json=sample_docs)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def query_documents(query_text, generate_response=True, top_k=3):
    """Query the documents."""
    print_section(f"Querying: '{query_text}'")

    query_data = {
        "query": query_text,
        "top_k": top_k,
        "generate_response": generate_response
    }

    response = requests.post(f"{BASE_URL}/query", json=query_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nRetrieved {len(result['retrieved_documents'])} documents:")

        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"\n  Document {i} (Similarity: {doc['similarity_score']:.4f}):")
            print(f"  ID: {doc['id']}")
            print(f"  Content: {doc['content'][:100]}...")
            if doc['metadata']:
                print(f"  Metadata: {doc['metadata']}")

        if result.get('generated_response'):
            print(f"\n🤖 AI-Generated Response:")
            print(f"  {result['generated_response']}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def get_stats():
    """Get statistics about stored documents."""
    print_section("Database Statistics")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def main():
    """Run all tests."""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║         RAG Pipeline API Test Script                  ║")
    print("╚════════════════════════════════════════════════════════╝")

    try:
        # Test health
        if not test_health():
            print("\n❌ Health check failed! Make sure the API is running.")
            return

        # Load documents
        if not load_documents():
            print("\n❌ Failed to load documents!")
            return

        # Wait a moment for processing
        time.sleep(1)

        # Get stats
        get_stats()

        # Query documents - Example 1
        query_documents("What is FastAPI?", generate_response=True, top_k=3)

        # Query documents - Example 2
        query_documents("How does vector similarity search work?", generate_response=True, top_k=3)

        # Query documents - Example 3
        query_documents("Explain RAG", generate_response=True, top_k=3)

        print_section("✅ All Tests Completed Successfully!")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API.")
        print("Make sure the API is running at", BASE_URL)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
