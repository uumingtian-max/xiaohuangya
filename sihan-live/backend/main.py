"""FastAPI application factory for the sihan-live backend.

Endpoints (all but ``/health`` require ``X-Owner-Id`` + ``X-API-Key`` headers):

* ``GET  /health``         – liveness probe.
* ``POST /memory/ingest``  – import text files from an allowed path into the KB.
* ``POST /memory/search``  – TF-IDF retrieval over stored memory.
* ``POST /chat``           – RAG chat: retrieve memory, then answer in persona.
* ``GET  /memory/stats``   – chunk/source counts.
* ``GET  /memory/chunks``  – debug: list stored chunks.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .auth import UserContext, build_owner_guard
from .config import Settings, load_settings
from .llm_engine import PersonaEngine
from .memory import KnowledgeBase
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)


def create_app(config_path: Path | None = None) -> FastAPI:
    settings: Settings = load_settings(config_path)
    kb = KnowledgeBase(settings.knowledge_base)
    engine = PersonaEngine(persona_name=settings.app.persona_name, llm_config=settings.llm)
    require_owner = build_owner_guard(settings)

    app = FastAPI(title=settings.app.name, version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", app=settings.app.name)

    @app.post("/memory/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest, ctx: UserContext = Depends(require_owner)) -> IngestResponse:
        try:
            report = kb.ingest_path(payload.path, recursive=payload.recursive)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return IngestResponse(
            documents_added=report.documents_added,
            chunks_added=report.chunks_added,
            skipped_files=report.skipped_files,
            storage_path=str(settings.knowledge_base.storage_path),
        )

    @app.post("/memory/search", response_model=SearchResponse)
    def search(payload: SearchRequest, ctx: UserContext = Depends(require_owner)) -> SearchResponse:
        hits = kb.search(payload.query, top_k=payload.top_k)
        results = [SearchHit(content=h.content, source=h.source, score=h.score) for h in hits]
        return SearchResponse(query=payload.query, results=results)

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, ctx: UserContext = Depends(require_owner)) -> ChatResponse:
        hits = kb.search(payload.message, top_k=payload.top_k)
        answer = engine.generate_reply(payload.message, hits)
        references = [SearchHit(content=h.content, source=h.source, score=h.score) for h in hits]
        return ChatResponse(
            reply=answer.reply,
            state=answer.state,
            context_used=answer.context_used,
            references=references,
        )

    @app.get("/memory/stats", response_model=StatsResponse)
    def stats(ctx: UserContext = Depends(require_owner)) -> StatsResponse:
        return StatsResponse(**kb.stats())

    @app.get("/memory/chunks", response_model=list[SearchHit])
    def list_chunks(
        limit: int = Query(20, ge=1, le=100),
        ctx: UserContext = Depends(require_owner),
    ) -> list[SearchHit]:
        return [SearchHit(content=c.content, source=c.source, score=c.score) for c in kb.list_chunks(limit)]

    return app


# Module-level instance for `uvicorn backend.main:app`.
app = create_app()
