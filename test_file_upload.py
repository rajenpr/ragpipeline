#!/usr/bin/env python3
"""
Test script for file upload functionality in RAG Pipeline API

This script demonstrates how to upload various document types.
"""

import requests
import json
import tempfile
import os


BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def create_sample_text_file():
    """Create a sample text file for testing."""
    content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

1. Supervised Learning
Supervised learning uses labeled datasets to train algorithms. The algorithm learns to map inputs to outputs based on example input-output pairs.

2. Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled responses. Common techniques include clustering and dimensionality reduction.

3. Reinforcement Learning
Reinforcement learning is about taking suitable actions to maximize reward in a particular situation. It employs trial and error to learn optimal behaviors.

## Applications

Machine learning has numerous applications including:
- Image and speech recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles

## Conclusion

Machine learning continues to evolve and shape the future of technology, enabling computers to perform tasks that previously required human intelligence.
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


def create_sample_markdown_file():
    """Create a sample markdown file for testing."""
    content = """# FastAPI Framework Guide

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Increase development speed by about 200% to 300%
- **Fewer bugs**: Reduce human-induced errors by about 40%
- **Intuitive**: Great editor support with auto-completion
- **Easy**: Designed to be easy to use and learn
- **Short**: Minimize code duplication

## Example Code

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

## Type Hints

FastAPI uses Python type hints for:
- Data validation
- Auto-generated documentation
- Editor support and auto-completion
- Reduced bugs

This makes development faster and more reliable.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        return f.name


def test_upload_text_file():
    """Test uploading a text file."""
    print_section("Test 1: Upload Text File")

    filepath = create_sample_text_file()

    try:
        with open(filepath, 'rb') as f:
            files = {'file': ('machine_learning.txt', f, 'text/plain')}
            data = {
                'chunk_size': 500,
                'chunk_overlap': 100,
                'metadata': json.dumps({"source": "test", "category": "AI", "topic": "machine learning"})
            }

            response = requests.post(f"{BASE_URL}/upload", files=files, data=data)

            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"   Filename: {result['filename']}")
                print(f"   File Type: {result['file_type']}")
                print(f"   Chunks Created: {result['chunks_created']}")
                print(f"   Total Characters: {result['total_characters']}")
                print(f"   Document IDs: {result['document_ids']}")
                return result['document_ids']
            else:
                print(f"❌ Error: {response.text}")
                return []
    finally:
        # Clean up temporary file
        os.unlink(filepath)


def test_upload_markdown_file():
    """Test uploading a markdown file."""
    print_section("Test 2: Upload Markdown File")

    filepath = create_sample_markdown_file()

    try:
        with open(filepath, 'rb') as f:
            files = {'file': ('fastapi_guide.md', f, 'text/markdown')}
            data = {
                'chunk_size': 800,
                'chunk_overlap': 150,
                'metadata': json.dumps({"source": "test", "category": "frameworks", "topic": "FastAPI"})
            }

            response = requests.post(f"{BASE_URL}/upload", files=files, data=data)

            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"   Filename: {result['filename']}")
                print(f"   File Type: {result['file_type']}")
                print(f"   Chunks Created: {result['chunks_created']}")
                print(f"   Total Characters: {result['total_characters']}")
                print(f"   Document IDs: {result['document_ids']}")
                return result['document_ids']
            else:
                print(f"❌ Error: {response.text}")
                return []
    finally:
        # Clean up temporary file
        os.unlink(filepath)


def test_query_uploaded_content():
    """Test querying the uploaded content."""
    print_section("Test 3: Query Uploaded Content")

    queries = [
        "What is machine learning?",
        "Explain supervised learning",
        "What are the key features of FastAPI?"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)

        response = requests.post(
            f"{BASE_URL}/query",
            json={
                "query": query,
                "top_k": 3,
                "generate_response": True
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {len(result['retrieved_documents'])} documents:")

            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"\n  Document {i} (Similarity: {doc['similarity_score']:.4f}):")
                print(f"  Content preview: {doc['content'][:100]}...")

                # Parse and display metadata
                if doc['metadata']:
                    metadata = json.loads(doc['metadata'])
                    if 'filename' in metadata:
                        print(f"  Source file: {metadata['filename']}")
                    if 'chunk_index' in metadata:
                        print(f"  Chunk: {metadata['chunk_index'] + 1}/{metadata.get('chunk_total', '?')}")

            if result.get('generated_response'):
                print(f"\n  🤖 AI Response:")
                print(f"  {result['generated_response']}")
        else:
            print(f"❌ Error: {response.text}")


def test_upload_with_default_params():
    """Test uploading with default parameters."""
    print_section("Test 4: Upload with Default Parameters")

    filepath = create_sample_text_file()

    try:
        with open(filepath, 'rb') as f:
            files = {'file': ('default_params.txt', f, 'text/plain')}
            # No additional parameters - using defaults

            response = requests.post(f"{BASE_URL}/upload", files=files)

            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success with default parameters!")
                print(f"   Chunks Created: {result['chunks_created']}")
                print(f"   (Using default chunk_size=1000, chunk_overlap=200)")
            else:
                print(f"❌ Error: {response.text}")
    finally:
        os.unlink(filepath)


def test_unsupported_file_type():
    """Test uploading an unsupported file type."""
    print_section("Test 5: Unsupported File Type (Expected to Fail)")

    # Create a fake .exe file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
        f.write("fake executable content")
        filepath = f.name

    try:
        with open(filepath, 'rb') as f:
            files = {'file': ('program.exe', f, 'application/x-msdownload')}

            response = requests.post(f"{BASE_URL}/upload", files=files)

            print(f"Status Code: {response.status_code}")
            if response.status_code == 400:
                print(f"✅ Correctly rejected unsupported file type")
                print(f"   Error message: {response.json()['detail']}")
            else:
                print(f"⚠️  Unexpected response: {response.text}")
    finally:
        os.unlink(filepath)


def main():
    """Run all tests."""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║      RAG Pipeline File Upload Test Script             ║")
    print("╚════════════════════════════════════════════════════════╝")

    try:
        # Check if API is running
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ API is not healthy!")
            return
    except requests.exceptions.RequestException:
        print("\n❌ Could not connect to API at", BASE_URL)
        print("Make sure the API is running: uvicorn app.main:app --reload")
        return

    # Run tests
    test_upload_text_file()
    test_upload_markdown_file()
    test_upload_with_default_params()
    test_unsupported_file_type()
    test_query_uploaded_content()

    print_section("✅ All Tests Completed!")
    print("\nNote: To test .docx, .pdf, or .doc files, place them in the")
    print("current directory and modify this script to upload them.")


if __name__ == "__main__":
    main()
