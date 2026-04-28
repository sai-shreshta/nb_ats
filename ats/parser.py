"""
parser.py — Extract raw text from PDF or DOCX resume files.
"""
import io

import pdfplumber
from docx import Document


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Dispatch to the correct parser based on file extension.
    Returns extracted plain text.
    Raises ValueError if the file type is unsupported or extraction fails.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _parse_pdf(file_bytes, filename)
    elif lower.endswith(".docx"):
        return _parse_docx(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Only .pdf and .docx are accepted.")


def _parse_pdf(file_bytes: bytes, filename: str) -> str:
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        result = "\n".join(text_parts).strip()
        if not result:
            raise ValueError("No text could be extracted from the PDF (possibly scanned/image-only).")
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse PDF '{filename}': {e}") from e


def _parse_docx(file_bytes: bytes, filename: str) -> str:
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        result = "\n".join(paragraphs).strip()
        if not result:
            raise ValueError("No text could be extracted from the DOCX file.")
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX '{filename}': {e}") from e
