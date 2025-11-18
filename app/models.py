from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database import Base
from app.config import get_settings

settings = get_settings()


class Document(Base):
    """Document model for storing text chunks and their embeddings."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    embedding = Column(Vector(settings.embedding_dimension), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, content={self.content[:50]}...)>"
