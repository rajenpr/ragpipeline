# RAG Pipeline with FastAPI, PgVector, and Portkey

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, PostgreSQL with PgVector extension, and Portkey for AI operations.

## Features

- **Document Loading**: Store documents with automatic vector embeddings
- **Semantic Search**: Query documents using vector similarity search
- **AI-Powered Responses**: Generate context-aware responses using LLMs via Portkey
- **PgVector Integration**: Efficient vector storage and retrieval in PostgreSQL
- **Batch Processing**: Load multiple documents efficiently
- **RESTful API**: Clean and well-documented API endpoints

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       v
┌─────────────────────────────────┐
│      FastAPI Application        │
│  ┌──────────┐    ┌───────────┐ │
│  │  /load   │    │  /query   │ │
│  └──────────┘    └───────────┘ │
└────────┬──────────────┬─────────┘
         │              │
         v              v
    ┌────────────────────────┐
    │   Portkey AI Service   │
    │  - Embeddings          │
    │  - Chat Completions    │
    └────────────────────────┘
         │
         v
    ┌──────────────────┐
    │   PostgreSQL     │
    │   + PgVector     │
    └──────────────────┘
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Portkey API account ([Get one here](https://portkey.ai))

## Setup

### 1. Clone and Navigate

```bash
cd ragpipeline
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Portkey credentials:

```env
PORTKEY_API_KEY=your_portkey_api_key_here
PORTKEY_VIRTUAL_KEY=your_virtual_key_here
```

### 3. Start PostgreSQL with PgVector

```bash
docker-compose up -d
```

This will start a PostgreSQL database with PgVector extension on port 5432.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Load Documents

**Endpoint**: `POST /load`

Load documents and store their vector embeddings.

**Request Body**:
```json
{
  "documents": [
    {
      "content": "FastAPI is a modern web framework for building APIs with Python.",
      "metadata": "{\"source\": \"docs\", \"category\": \"frameworks\"}"
    },
    {
      "content": "PgVector is a PostgreSQL extension for vector similarity search.",
      "metadata": "{\"source\": \"docs\", \"category\": \"databases\"}"
    }
  ]
}
```

**Response**:
```json
{
  "message": "Documents loaded successfully",
  "count": 2,
  "document_ids": [1, 2]
}
```

### 2. Query Documents

**Endpoint**: `POST /query`

Query documents using semantic search and get an AI-generated response.

**Request Body**:
```json
{
  "query": "What is FastAPI?",
  "top_k": 5,
  "generate_response": true
}
```

**Response**:
```json
{
  "query": "What is FastAPI?",
  "retrieved_documents": [
    {
      "id": 1,
      "content": "FastAPI is a modern web framework for building APIs with Python.",
      "metadata": "{\"source\": \"docs\", \"category\": \"frameworks\"}",
      "similarity_score": 0.89
    }
  ],
  "generated_response": "FastAPI is a modern, high-performance web framework for building APIs with Python. It's designed to be easy to use while providing excellent performance and automatic API documentation."
}
```

### 3. Health Check

**Endpoint**: `GET /health`

Check if the service is healthy and database is connected.

### 4. Statistics

**Endpoint**: `GET /stats`

Get statistics about stored documents.

## Usage Examples

### Using cURL

**Load documents**:
```bash
curl -X POST "http://localhost:8000/load" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "The Python programming language was created by Guido van Rossum.",
        "metadata": "{\"topic\": \"programming\"}"
      }
    ]
  }'
```

**Query documents**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who created Python?",
    "top_k": 3,
    "generate_response": true
  }'
```

### Using Python

```python
import requests

# Load documents
response = requests.post(
    "http://localhost:8000/load",
    json={
        "documents": [
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": '{"category": "AI"}'
            }
        ]
    }
)
print(response.json())

# Query documents
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is machine learning?",
        "top_k": 5,
        "generate_response": True
    }
)
print(response.json())
```

## Project Structure

```
ragpipeline/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoints
│   ├── config.py        # Configuration and settings
│   ├── database.py      # Database setup and session management
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas for request/response
│   └── services.py      # Embedding and AI services (Portkey)
├── docker-compose.yml   # PostgreSQL with PgVector
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore
└── README.md
```

## Configuration

All configuration is managed through environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/ragdb` |
| `PORTKEY_API_KEY` | Your Portkey API key | Required |
| `PORTKEY_VIRTUAL_KEY` | Your Portkey virtual key | Required |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-small` |
| `CHAT_MODEL` | Model for chat completions | `gpt-4-turbo-preview` |
| `EMBEDDING_DIMENSION` | Dimension of embedding vectors | `1536` |

## API Documentation

Once the application is running, visit:

- **Interactive API docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API docs (ReDoc)**: http://localhost:8000/redoc

## How It Works

### Document Loading Flow

1. Client sends documents via `/load` endpoint
2. Documents are sent to Portkey to generate embeddings
3. Documents and their embeddings are stored in PostgreSQL with PgVector
4. Document IDs are returned to the client

### Query Flow

1. Client sends query via `/query` endpoint
2. Query text is converted to embedding via Portkey
3. PgVector performs cosine similarity search to find relevant documents
4. Top-K most similar documents are retrieved
5. (Optional) Retrieved documents are used as context for LLM to generate response
6. Results and generated response are returned to client

## Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload
```

### Database Management

Reset database:
```bash
docker-compose down -v
docker-compose up -d
```

Access PostgreSQL:
```bash
docker exec -it ragpipeline_postgres psql -U postgres -d ragdb
```

### Useful SQL Queries

Check stored documents:
```sql
SELECT id, content, created_at FROM documents;
```

Check vector dimensions:
```sql
SELECT id, array_length(embedding::float[], 1) as dimension FROM documents LIMIT 1;
```

## Troubleshooting

### Database connection errors
- Ensure PostgreSQL is running: `docker-compose ps`
- Check database logs: `docker-compose logs postgres`

### Embedding errors
- Verify Portkey API keys in `.env`
- Check Portkey dashboard for API usage and errors

### Performance optimization
- Adjust `top_k` parameter for query endpoint
- Create indexes on frequently queried fields
- Use batch loading for large document sets

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
