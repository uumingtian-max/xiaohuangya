"""A tiny, dependency-free RAG memory store.

Text files are chunked, vectorised with TF-IDF and persisted to a JSON file.
Retrieval ranks chunks by cosine similarity. It is intentionally small: no
external vector DB, good enough for a single owner's private knowledge base.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import KnowledgeBaseConfig

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".log"}


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def _flush_cjk(run: list[str], out: list[str]) -> None:
    """Emit CJK character unigrams and bigrams for a run of CJK chars.

    Chinese has no spaces, so whole-run tokens never match a substring query
    (e.g. query 「纪念日」 vs stored 「我们每月纪念日是」). Character n-grams give
    overlap without needing a heavyweight segmenter like jieba.
    """
    if not run:
        return
    out.extend(run)  # unigrams
    for a, b in zip(run, run[1:]):  # bigrams
        out.append(a + b)
    run.clear()


def _tokenize(text: str) -> list[str]:
    """Lowercase tokens: word tokens for alnum runs, n-grams for CJK runs."""
    tokens: list[str] = []
    cjk_run: list[str] = []
    word: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            if word:
                tokens.append("".join(word).lower())
                word.clear()
            cjk_run.append(ch)
        elif ch.isalnum():
            _flush_cjk(cjk_run, tokens)
            word.append(ch)
        else:
            if word:
                tokens.append("".join(word).lower())
                word.clear()
            _flush_cjk(cjk_run, tokens)
    if word:
        tokens.append("".join(word).lower())
    _flush_cjk(cjk_run, tokens)
    return tokens


def _chunk_text(content: str, size: int, overlap: int) -> list[str]:
    content = content.strip()
    if not content:
        return []
    if len(content) <= size:
        return [content]
    chunks: list[str] = []
    step = max(1, size - overlap)
    idx = 0
    while idx < len(content):
        chunk = content[idx : idx + size].strip()
        if chunk:
            chunks.append(chunk)
        idx += step
    return chunks


def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    # Iterate over the smaller vector for the dot product.
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
    dot = sum(val * vec2.get(tok, 0.0) for tok, val in vec1.items())
    if dot == 0.0:
        return 0.0
    norm1 = math.sqrt(sum(val * val for val in vec1.values()))
    norm2 = math.sqrt(sum(val * val for val in vec2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    text: str
    term_freq: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "text": self.text,
            "term_freq": self.term_freq,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "DocumentChunk":
        return cls(
            chunk_id=raw["chunk_id"],
            source_path=raw["source_path"],
            text=raw["text"],
            term_freq={k: float(v) for k, v in raw.get("term_freq", {}).items()},
        )


@dataclass(slots=True)
class RetrievedChunk:
    content: str
    source: str
    score: float
    chunk_id: str


@dataclass(slots=True)
class IngestReport:
    documents_added: int = 0
    chunks_added: int = 0
    skipped_files: int = 0
    errors: list[str] = field(default_factory=list)


class KnowledgeBase:
    """File-backed TF-IDF knowledge base for a single owner."""

    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self.config = config
        self.storage_path = config.storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[DocumentChunk] = []
        self._idf: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------ I/O
    def _load(self) -> None:
        if not self.storage_path.exists():
            self._chunks = []
        else:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._chunks = [DocumentChunk.from_dict(item) for item in raw.get("chunks", [])]
        self._rebuild_idf()

    def _persist(self) -> None:
        payload = {"chunks": [chunk.to_dict() for chunk in self._chunks]}
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------- indexing
    def _rebuild_idf(self) -> None:
        """Recompute inverse document frequency across all chunks."""
        n_docs = len(self._chunks)
        doc_freq: dict[str, int] = {}
        for chunk in self._chunks:
            for token in chunk.term_freq:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        # Smoothed IDF so a term present in every doc still carries weight.
        self._idf = {
            token: math.log((n_docs + 1) / (df + 1)) + 1.0 for token, df in doc_freq.items()
        }

    @staticmethod
    def _term_freq(text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0
        total = float(len(tokens))
        return {token: freq / total for token, freq in tf.items()}

    def _tfidf_vector(self, term_freq: dict[str, float]) -> dict[str, float]:
        return {
            token: tf * self._idf.get(token, 1.0)
            for token, tf in term_freq.items()
            if self._idf.get(token, 1.0) != 0.0
        }

    # --------------------------------------------------------------- access
    def _ensure_allowed(self, target: Path) -> Path:
        resolved = target.expanduser().resolve()
        if not self.config.allowed_roots:
            return resolved
        allowed = any(
            resolved == root.resolve() or root.resolve() in resolved.parents
            for root in self.config.allowed_roots
        )
        if not allowed:
            raise PermissionError(f"path {resolved} is outside the allowed roots")
        return resolved

    def _iter_files(self, root: Path, recursive: bool) -> Iterable[Path]:
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield root
            return
        glob = root.rglob("*") if recursive else root.glob("*")
        for file_path in glob:
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path

    # --------------------------------------------------------------- public
    def ingest_path(self, source_path: str | Path, recursive: bool = True) -> IngestReport:
        report = IngestReport()
        target = self._ensure_allowed(Path(source_path))
        if not target.exists():
            raise FileNotFoundError(f"path does not exist: {target}")

        seen_ids = {chunk.chunk_id for chunk in self._chunks}
        for file_path in self._iter_files(target, recursive):
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                report.skipped_files += 1
                report.errors.append(f"skip {file_path}: {exc}")
                continue

            chunks = _chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
            if not chunks:
                report.skipped_files += 1
                continue

            added_for_file = 0
            for idx, chunk_text in enumerate(chunks):
                chunk_hash = hashlib.sha1(
                    f"{file_path}:{idx}:{chunk_text}".encode("utf-8")
                ).hexdigest()
                if chunk_hash in seen_ids:
                    continue  # idempotent re-ingest
                seen_ids.add(chunk_hash)
                self._chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_hash,
                        source_path=str(file_path),
                        text=chunk_text,
                        term_freq=self._term_freq(chunk_text),
                    )
                )
                report.chunks_added += 1
                added_for_file += 1

            if added_for_file:
                report.documents_added += 1

        if report.chunks_added:
            self._rebuild_idf()
            self._persist()
        return report

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vec = self._tfidf_vector(self._term_freq(query))
        if not query_vec:
            return []
        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            score = _cosine_similarity(query_vec, self._tfidf_vector(chunk.term_freq))
            if score > 0:
                scored.append(
                    RetrievedChunk(
                        content=chunk.text,
                        source=chunk.source_path,
                        score=round(score, 6),
                        chunk_id=chunk.chunk_id,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def list_chunks(self, limit: int = 20) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(content=c.text, source=c.source_path, score=0.0, chunk_id=c.chunk_id)
            for c in self._chunks[:limit]
        ]

    def stats(self) -> dict[str, int]:
        return {
            "chunks": len(self._chunks),
            "sources": len({chunk.source_path for chunk in self._chunks}),
        }
