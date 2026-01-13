"""
Document processing utilities for AI Council.
Handles PDF, Word (.docx), and text file extraction.
"""

import io
from typing import Tuple
from pathlib import Path


def extract_pdf_text(file_content: bytes) -> str:
    """Extract text from PDF bytes."""
    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(file_content))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)


def extract_docx_text(file_content: bytes) -> str:
    """Extract text from Word document bytes."""
    from docx import Document

    doc = Document(io.BytesIO(file_content))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_file(file_content: bytes) -> str:
    """Extract text from plain text file."""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        return file_content.decode('latin-1')


def process_uploaded_file(
    filename: str,
    file_content: bytes
) -> Tuple[str, str]:
    """
    Process an uploaded file and extract text.

    Args:
        filename: Original filename
        file_content: Raw file bytes

    Returns:
        Tuple of (extracted_text, content_type)

    Raises:
        ValueError: If file type is not supported
    """
    extension = Path(filename).suffix.lower()

    if extension == '.pdf':
        return extract_pdf_text(file_content), 'pdf'
    elif extension == '.docx':
        return extract_docx_text(file_content), 'docx'
    elif extension in ('.txt', '.md', '.markdown'):
        return extract_text_file(file_content), 'text'
    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported: .pdf, .docx, .txt, .md"
        )


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count
        model: Tokenizer model (cl100k_base works for Claude)

    Returns:
        Estimated token count
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int, model: str = "cl100k_base") -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        model: Tokenizer model

    Returns:
        Truncated text
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    except Exception:
        # Fallback: character-based truncation
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text
        return text[:char_limit] + "..."
