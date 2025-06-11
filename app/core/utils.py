import hashlib
from pydantic import BaseModel, Field
from typing import List, Optional

def generate_doc_id(file_name: str) -> str:
    """Generates a unique document ID based on the file name."""
    return hashlib.md5(file_name.encode()).hexdigest()[:10]

class DocumentSnippet(BaseModel):
    """Data model for a snippet of a document."""
    doc_id: str
    content: str
    page: Optional[int] = None
    paragraph: Optional[int] = None

class DocumentAnswer(BaseModel):
    """Data model for an answer extracted from a single document."""
    doc_id: str
    extracted_answer: str = "No relevant information found in this document."
    source_page: Optional[int] = None
    source_paragraph: Optional[int] = None

class Theme(BaseModel):
    """Data model for a synthesized theme."""
    theme_title: str
    summary: str
    supporting_docs: List[str]