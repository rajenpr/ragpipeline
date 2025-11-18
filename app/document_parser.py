"""
Document parser utilities for extracting text from various file formats.
"""

import io
import os
import tempfile
from typing import Optional, List
from pathlib import Path
import chardet
from docx import Document as DocxDocument
from PyPDF2 import PdfReader


class DocumentParser:
    """Parser for extracting text from various document formats."""

    @staticmethod
    def detect_encoding(content: bytes) -> str:
        """
        Detect the encoding of text content.

        Args:
            content: Raw bytes content

        Returns:
            Detected encoding name
        """
        result = chardet.detect(content)
        return result.get('encoding', 'utf-8') or 'utf-8'

    @staticmethod
    def parse_txt(content: bytes) -> str:
        """
        Parse plain text file.

        Args:
            content: File content as bytes

        Returns:
            Extracted text
        """
        encoding = DocumentParser.detect_encoding(content)
        try:
            return content.decode(encoding)
        except Exception:
            return content.decode('utf-8', errors='ignore')

    @staticmethod
    def parse_docx(content: bytes) -> str:
        """
        Parse DOCX file.

        Args:
            content: File content as bytes

        Returns:
            Extracted text from all paragraphs
        """
        doc = DocxDocument(io.BytesIO(content))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

        # Also extract text from tables
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    tables_text.append(' | '.join(row_text))

        all_text = paragraphs + tables_text
        return '\n'.join(all_text)

    @staticmethod
    def parse_doc(content: bytes) -> str:
        """
        Parse DOC file using pypandoc.

        Args:
            content: File content as bytes

        Returns:
            Extracted text

        Note:
            Requires pandoc to be installed on the system.
            Falls back to basic extraction if pypandoc is not available.
        """
        try:
            import pypandoc

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Convert DOC to plain text using pandoc
                text = pypandoc.convert_file(tmp_path, 'plain', format='doc')
                return text
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except (ImportError, RuntimeError, OSError) as e:
            # If pypandoc is not available or pandoc is not installed
            raise ValueError(
                "DOC file parsing requires pandoc to be installed. "
                "Please install pandoc (https://pandoc.org/installing.html) "
                "or convert your DOC file to DOCX format. "
                f"Error: {str(e)}"
            )

    @staticmethod
    def parse_pdf(content: bytes) -> str:
        """
        Parse PDF file.

        Args:
            content: File content as bytes

        Returns:
            Extracted text from all pages
        """
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)

        text_parts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)

        return '\n\n'.join(text_parts)

    @staticmethod
    def parse_document(content: bytes, filename: str) -> str:
        """
        Parse document based on file extension.

        Args:
            content: File content as bytes
            filename: Name of the file (used to determine type)

        Returns:
            Extracted text content

        Raises:
            ValueError: If file type is not supported
        """
        file_ext = Path(filename).suffix.lower()

        parsers = {
            '.txt': DocumentParser.parse_txt,
            '.text': DocumentParser.parse_txt,
            '.md': DocumentParser.parse_txt,
            '.docx': DocumentParser.parse_docx,
            '.doc': DocumentParser.parse_doc,
            '.pdf': DocumentParser.parse_pdf,
        }

        parser = parsers.get(file_ext)
        if not parser:
            supported = ', '.join(parsers.keys())
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Supported formats: {supported}"
            )

        try:
            text = parser(content)
            if not text or not text.strip():
                raise ValueError("No text content could be extracted from the document")
            return text
        except Exception as e:
            raise ValueError(f"Error parsing {file_ext} file: {str(e)}")


class TextChunker:
    """Utility for splitting text into chunks for embedding."""

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be included

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Clean the text
        text = text.strip()

        # If text is smaller than chunk_size, return as single chunk
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + chunk_size

            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start:
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end),
                        text.rfind('\n', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1

            # Extract chunk
            chunk = text[start:end].strip()

            # Only add chunks that meet minimum size
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - chunk_overlap if end < len(text) else end

            # Prevent infinite loop
            if start <= 0:
                start = end

        return chunks

    @staticmethod
    def chunk_by_paragraphs(
        text: str,
        max_chunk_size: int = 1000
    ) -> List[str]:
        """
        Split text by paragraphs, grouping small paragraphs together.

        Args:
            text: Text to split
            max_chunk_size: Maximum size of each chunk in characters

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Split by paragraph breaks
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If adding this paragraph exceeds max size, save current chunk
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            # If single paragraph is larger than max size, split it
            if para_size > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # Split large paragraph into sentences
                chunks.extend(TextChunker.chunk_text(para, max_chunk_size))
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks
