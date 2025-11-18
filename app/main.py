from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import json

from app.database import get_db, init_db
from app.models import Document
from app.schemas import (
    DocumentInput,
    DocumentBatchInput,
    LoadResponse,
    QueryInput,
    QueryResponse,
    RetrievedDocument,
)
from app.services import get_embedding_service, EmbeddingService

app = FastAPI(
    title="RAG Pipeline API",
    description="A RAG pipeline with FastAPI, PgVector, and Portkey",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    print("RAG Pipeline API started successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Pipeline API",
        "endpoints": {
            "load": "/load - Load documents and store embeddings",
            "query": "/query - Query documents and get AI response",
            "docs": "/docs - API documentation",
        },
    }


@app.post("/load", response_model=LoadResponse)
async def load_documents(
    batch: DocumentBatchInput,
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Load documents and store their embeddings in the database.

    Args:
        batch: Batch of documents to load
        db: Database session
        embedding_service: Embedding service instance

    Returns:
        LoadResponse with status and document IDs
    """
    try:
        # Extract text content from all documents
        texts = [doc.content for doc in batch.documents]

        # Generate embeddings for all documents in batch
        embeddings = embedding_service.get_embeddings_batch(texts)

        # Store documents in database
        document_ids = []
        for doc_input, embedding in zip(batch.documents, embeddings):
            document = Document(
                content=doc_input.content,
                metadata=doc_input.metadata,
                embedding=embedding,
            )
            db.add(document)
            db.flush()  # Get the ID without committing
            document_ids.append(document.id)

        db.commit()

        return LoadResponse(
            message="Documents loaded successfully",
            count=len(document_ids),
            document_ids=document_ids,
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error loading documents: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    query_input: QueryInput,
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Query documents using semantic search and optionally generate AI response.

    Args:
        query_input: Query parameters
        db: Database session
        embedding_service: Embedding service instance

    Returns:
        QueryResponse with retrieved documents and optional generated response
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_service.get_embedding(query_input.query)

        # Perform vector similarity search using pgvector
        # Using cosine distance (1 - cosine_similarity)
        query_sql = text(
            """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> :query_embedding) as similarity
            FROM documents
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
            """
        )

        result = db.execute(
            query_sql,
            {
                "query_embedding": str(query_embedding),
                "top_k": query_input.top_k,
            },
        )

        # Process results
        retrieved_docs = []
        for row in result:
            retrieved_docs.append(
                RetrievedDocument(
                    id=row.id,
                    content=row.content,
                    metadata=row.metadata,
                    similarity_score=float(row.similarity),
                )
            )

        # Generate response if requested
        generated_response = None
        if query_input.generate_response and retrieved_docs:
            # Combine retrieved documents into context
            context = "\n\n".join(
                [
                    f"[Document {i+1}]\n{doc.content}"
                    for i, doc in enumerate(retrieved_docs)
                ]
            )

            # Generate response using LLM
            generated_response = embedding_service.generate_response(
                query_input.query, context
            )

        return QueryResponse(
            query=query_input.query,
            retrieved_documents=retrieved_docs,
            generated_response=generated_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database connection
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get statistics about stored documents."""
    try:
        count = db.query(Document).count()
        return {"total_documents": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
