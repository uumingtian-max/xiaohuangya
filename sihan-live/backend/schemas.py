"""Request/response models for the sihan-live API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str


class IngestRequest(BaseModel):
    path: str = Field(..., description="Directory or file path to ingest (must be under an allowed root).")
    recursive: bool = Field(default=True, description="Recurse into subdirectories when path is a directory.")


class IngestResponse(BaseModel):
    documents_added: int
    chunks_added: int
    skipped_files: int
    storage_path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    content: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    reply: str
    state: str
    context_used: int
    references: list[SearchHit]


class StatsResponse(BaseModel):
    chunks: int
    sources: int
