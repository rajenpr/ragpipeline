from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import json
import logging

from app.database import get_db, init_db
from app.models import Document
from app.schemas import (
    DocumentInput,
    DocumentBatchInput,
    LoadResponse,
    QueryInput,
    QueryResponse,
    RetrievedDocument,
    FileUploadResponse,
)
from app.services import get_embedding_service, EmbeddingService
from app.document_parser import DocumentParser, TextChunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            "upload": "/upload - Upload and process document files (.doc, .docx, .pdf, .txt)",
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
                doc_metadata=doc_input.metadata,
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


@app.post("/upload", response_model=FileUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Upload a document file and store it in chunks with embeddings.

    Supported formats: .txt, .doc, .docx, .pdf, .md

    Args:
        file: The uploaded file
        chunk_size: Maximum size of each text chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        metadata: Optional metadata as JSON string
        db: Database session
        embedding_service: Embedding service instance

    Returns:
        FileUploadResponse with upload status and chunk information
    """
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
    MAX_CHUNKS = 100  # Limit number of chunks to process

    try:
        logger.info(f"Starting upload for file: {file.filename}")

        # Read file content
        content = await file.read()
        file_size = len(content)

        logger.info(f"File size: {file_size} bytes")

        # Check file size limit
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024} MB"
            )

        # Parse document based on file type
        logger.info(f"Parsing document: {file.filename}")
        try:
            text = DocumentParser.parse_document(content, file.filename)
        except ValueError as e:
            logger.error(f"Document parsing error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(f"Extracted {len(text)} characters from document")

        # Split text into chunks
        logger.info("Splitting text into chunks...")
        chunks = TextChunker.chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the document"
            )

        # Limit number of chunks
        if len(chunks) > MAX_CHUNKS:
            logger.warning(f"Too many chunks ({len(chunks)}), limiting to {MAX_CHUNKS}")
            chunks = chunks[:MAX_CHUNKS]

        logger.info(f"Created {len(chunks)} chunks")

        # Prepare metadata
        file_metadata = {
            "filename": file.filename,
            "file_type": file.content_type,
            "total_chunks": len(chunks),
        }

        # Merge with user-provided metadata
        if metadata:
            try:
                user_metadata = json.loads(metadata)
                file_metadata.update(user_metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata field")

        # Generate embeddings for chunks in smaller batches
        logger.info("Generating embeddings...")
        BATCH_SIZE = 20  # Process 20 chunks at a time
        all_embeddings = []

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.info(f"Processing embedding batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1}")
            try:
                batch_embeddings = embedding_service.get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating embeddings: {str(e)}"
                )

        embeddings = all_embeddings
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Store chunks in database
        logger.info("Storing chunks in database...")
        document_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Add chunk information to metadata
            chunk_metadata = file_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_total"] = len(chunks)

            document = Document(
                content=chunk,
                doc_metadata=json.dumps(chunk_metadata),
                embedding=embedding,
            )
            db.add(document)
            db.flush()
            document_ids.append(document.id)

        logger.info(f"Committing {len(document_ids)} documents to database...")
        db.commit()
        logger.info("Upload complete!")

        return FileUploadResponse(
            message="File uploaded and processed successfully",
            filename=file.filename,
            file_type=file.content_type or "unknown",
            chunks_created=len(chunks),
            document_ids=document_ids,
            total_characters=len(text),
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


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
                doc_metadata,
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
                    metadata=row.doc_metadata,
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
