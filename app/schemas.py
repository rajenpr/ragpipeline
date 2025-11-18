from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentInput(BaseModel):
    """Schema for document input."""

    content: str = Field(..., description="The text content to be stored")
    metadata: Optional[str] = Field(None, description="Optional metadata as JSON string")


class DocumentBatchInput(BaseModel):
    """Schema for batch document input."""

    documents: List[DocumentInput] = Field(..., description="List of documents to load")


class DocumentResponse(BaseModel):
    """Schema for document response."""

    id: int
    content: str
    metadata: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class LoadResponse(BaseModel):
    """Schema for load endpoint response."""

    message: str
    count: int
    document_ids: List[int]


class QueryInput(BaseModel):
    """Schema for query input."""

    query: str = Field(..., description="The query text")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    generate_response: bool = Field(True, description="Whether to generate AI response")


class RetrievedDocument(BaseModel):
    """Schema for retrieved document with similarity score."""

    id: int
    content: str
    metadata: Optional[str]
    similarity_score: float


class QueryResponse(BaseModel):
    """Schema for query endpoint response."""

    query: str
    retrieved_documents: List[RetrievedDocument]
    generated_response: Optional[str] = None
