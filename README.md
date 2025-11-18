# RAG Pipeline with FastAPI, PgVector, and Portkey

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, PostgreSQL with PgVector extension, and Portkey for AI operations.

## Features

- **Document Loading**: Store documents with automatic vector embeddings
- **File Upload Support**: Upload and process .doc, .docx, .pdf, .txt, and .md files
- **Automatic Text Chunking**: Smart splitting of large documents into manageable chunks
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
┌──────────────────────────────────────┐
│      FastAPI Application             │
│  ┌──────────┐    ┌───────────┐      │
│  │  /load   │    │  /query   │      │
│  │  /upload │    │           │      │
│  └──────────┘    └───────────┘      │
└────────┬──────────────┬──────────────┘
         │              │
         v              v
    ┌────────────────────────────┐
    │   LangChain Integration    │
    │  - ChatOpenAI              │
    │  - OpenAIEmbeddings        │
    └────────┬───────────────────┘
             │
             v
    ┌────────────────────────────┐
    │   Portkey AI Gateway       │
    │  - Routes to OpenAI        │
    │  - Observability           │
    │  - Caching & Fallbacks     │
    └────────┬───────────────────┘
             │
             v
    ┌────────────────────────────┐
    │   OpenAI API               │
    │  - text-embedding-3-small  │
    │  - gpt-4-turbo-preview     │
    └────────────────────────────┘

         (Embeddings stored in)
             │
             v
    ┌──────────────────┐
    │   PostgreSQL     │
    │   + PgVector     │
    └──────────────────┘
```

## Prerequisites

- Python 3.10+
- PostgreSQL with PgVector extension (local via Docker or external server)
- Portkey API account ([Get one here](https://portkey.ai))
- **One of the following**:
  - Azure OpenAI access via Portkey (Option A - Recommended)
  - Portkey Virtual Keys (Option B)
  - OpenAI API account ([Get one here](https://platform.openai.com/)) (Option C)

## Setup

### 1. Clone and Navigate

```bash
cd ragpipeline
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and choose **ONE** of the following authentication setups:

#### Option A: Provider Routing with Azure OpenAI (Recommended)

```env
# Database (update with your PostgreSQL details)
DATABASE_URL=postgresql://n8n_user:redhat@123@10.121.210.176:5432/rag_system

# Portkey with Azure OpenAI provider routing
PORTKEY_API_KEY=your_portkey_api_key_here
PORTKEY_CHAT_PROVIDER=@azure-openai-eus-global/gpt-4.1-dzs
PORTKEY_EMBEDDING_PROVIDER=@azure-openai-eus-global/text-embedding-3-large-std
PORTKEY_BASE_URL=https://api.portkey.ai/v1

# Azure OpenAI models
EMBEDDING_MODEL=text-embedding-3-large-std
CHAT_MODEL=gpt-4.1-dzs
EMBEDDING_DIMENSION=3072
```

**Benefits:**
- Route to specific Azure OpenAI deployments
- No need to manage OpenAI API keys in your application
- Different providers for chat and embeddings
- Enterprise-grade Azure infrastructure

#### Option B: Using Portkey Virtual Keys

```env
# Portkey manages your provider API keys
PORTKEY_API_KEY=your_portkey_api_key_here
PORTKEY_VIRTUAL_KEY=your_portkey_virtual_key_here
PORTKEY_BASE_URL=https://api.portkey.ai/v1

# Model selection
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-turbo-preview
EMBEDDING_DIMENSION=1536
```

**Setup Steps for Virtual Keys:**
1. Go to [Portkey Dashboard](https://app.portkey.ai/)
2. Navigate to "Virtual Keys" section
3. Create a virtual key linked to your provider (OpenAI, Azure, etc.)
4. Copy the virtual key to your `.env` file

**Benefits:**
- Centralized key management in Portkey
- Easy provider switching without code changes
- Works with OpenAI, Azure, Anthropic, and more

#### Option C: Using Your Own OpenAI API Key

```env
# Your OpenAI API key (routed through Portkey for observability)
OPENAI_API_KEY=sk-your-openai-api-key-here
PORTKEY_API_KEY=your_portkey_api_key_here
PORTKEY_BASE_URL=https://api.portkey.ai/v1

# Model selection
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-turbo-preview
EMBEDDING_DIMENSION=1536
```

**Benefits:**
- Direct control over your OpenAI key
- Portkey provides observability, caching, and fallbacks
- Full transparency of API usage

**How Portkey Works:**
- All AI API calls are routed through Portkey's gateway
- Portkey provides observability, caching, load balancing, and fallbacks
- Supports provider routing, virtual keys, or direct API keys
- Works with OpenAI, Azure OpenAI, Anthropic, and other providers

### 3. Start PostgreSQL with PgVector (Optional)

**If using external PostgreSQL** (like in Option A above with custom DATABASE_URL):
- Skip this step if your PostgreSQL server is already running
- Ensure PgVector extension is installed on your PostgreSQL server:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

**If using local Docker PostgreSQL**:
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

### 3. Upload Document File

**Endpoint**: `POST /upload`

Upload document files (.doc, .docx, .pdf, .txt, .md) and automatically extract text, chunk it, and store embeddings.

**Form Data**:
- `file`: The document file (required)
- `chunk_size`: Maximum characters per chunk (optional, default: 1000)
- `chunk_overlap`: Characters to overlap between chunks (optional, default: 200)
- `metadata`: Additional metadata as JSON string (optional)

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200" \
  -F 'metadata={"source": "research", "category": "AI"}'
```

**Response**:
```json
{
  "message": "File uploaded and processed successfully",
  "filename": "document.pdf",
  "file_type": "application/pdf",
  "chunks_created": 5,
  "document_ids": [3, 4, 5, 6, 7],
  "total_characters": 4523
}
```

**Supported File Formats**:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents
- `.docx` - Microsoft Word (modern format)
- `.doc` - Microsoft Word (legacy format, requires `pandoc` installed)

**Note for .doc files**: Processing legacy .doc files requires `pandoc` to be installed on your system:
```bash
# Ubuntu/Debian
sudo apt-get install pandoc

# macOS
brew install pandoc

# Windows
# Download from https://pandoc.org/installing.html
```

### 4. Health Check

**Endpoint**: `GET /health`

Check if the service is healthy and database is connected.

### 5. Statistics

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

**Upload a file**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@mydocument.pdf" \
  -F "chunk_size=1000" \
  -F 'metadata={"source": "upload", "category": "docs"}'
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

# Upload a file
with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    data = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "metadata": '{"source": "upload", "category": "research"}'
    }
    response = requests.post("http://localhost:8000/upload", files=files, data=data)
    print(response.json())
```

## Project Structure

```
ragpipeline/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoints
│   ├── config.py        # Configuration and settings
│   ├── database.py      # PostgreSQL and PgVector setup
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas for request/response
│   ├── services.py      # Embedding and AI services (Portkey)
│   └── document_parser.py  # Document parsing and text chunking utilities
├── docker-compose.yml   # PostgreSQL with PgVector
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore
├── README.md
├── run.sh              # Quick startup script
├── test_api.py         # API testing script
└── example_client.py   # Python client library example
```

## Configuration

All configuration is managed through environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://n8n_user:redhat@123@10.121.210.176:5432/rag_system` |
| `OPENAI_API_KEY` | Your OpenAI API key | Optional* |
| `PORTKEY_API_KEY` | Your Portkey API key | Required |
| `PORTKEY_VIRTUAL_KEY` | Your Portkey virtual key | Optional* |
| `PORTKEY_CHAT_PROVIDER` | Portkey provider for chat (e.g., Azure OpenAI) | Optional* |
| `PORTKEY_EMBEDDING_PROVIDER` | Portkey provider for embeddings | Optional* |
| `PORTKEY_BASE_URL` | Portkey gateway base URL | `https://api.portkey.ai/v1` |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-large-std` |
| `CHAT_MODEL` | Model for chat completions | `gpt-4.1-dzs` |
| `EMBEDDING_DIMENSION` | Dimension of embedding vectors | `3072` |

**Authentication Requirement:** You must provide ONE of:
- `PORTKEY_CHAT_PROVIDER` + `PORTKEY_EMBEDDING_PROVIDER` (Option A: Provider routing for Azure OpenAI), OR
- `PORTKEY_VIRTUAL_KEY` (Option B: Virtual keys), OR
- `OPENAI_API_KEY` (Option C: Direct OpenAI key)

**Note:** This application uses LangChain with AI models (OpenAI/Azure OpenAI) routed through Portkey's gateway for enhanced observability, caching, and reliability.

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
