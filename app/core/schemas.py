from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class Citation(BaseModel):
    doc_id: str
    vendor: str
    version: str
    page_start: int
    page_end: int
    section_title: str
    chunk_id: str
    score: float


class EvidenceView(BaseModel):
    doc_id: str
    vendor: str
    version: str
    page_start: int
    page_end: int
    section_title: str
    chunk_id: str
    score: float
    preview: str


class AskResponse(BaseModel):
    status: Literal["ok", "refuse"]
    decision: Literal["answer", "refuse"]
    answer: Optional[str] = None
    reason: Optional[str] = None
    suggestion: Optional[str] = None
    citations: List[Citation] = []
    evidence: List[EvidenceView] = []