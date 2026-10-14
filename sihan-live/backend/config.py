"""Runtime configuration loading for the sihan-live backend.

A single YAML file drives the whole app. Three sections:

* ``app``            – identity, owner credentials, bind address, persona.
* ``knowledge_base`` – where the RAG store lives and how text is chunked.
* ``llm``            – how to reach the OpenAI-compatible chat endpoint
                       (e.g. a local ``llama-server``). Optional; when
                       ``enabled`` is false the persona engine answers from
                       retrieved memory alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppConfig:
    name: str
    owner_id: str
    api_key: str
    host: str
    port: int
    persona_name: str


@dataclass(slots=True)
class KnowledgeBaseConfig:
    storage_path: Path
    chunk_size: int
    chunk_overlap: int
    allowed_roots: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class LLMConfig:
    """OpenAI-compatible chat completion endpoint settings."""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-model"
    api_key: str | None = None
    timeout: float = 30.0
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass(slots=True)
class Settings:
    app: AppConfig
    knowledge_base: KnowledgeBaseConfig
    llm: LLMConfig


def _to_abs_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_settings(config_path: Path | None = None) -> Settings:
    """Load runtime settings from a YAML config file.

    Paths in the config are resolved relative to the config file's directory,
    so the app behaves the same regardless of the current working directory.
    """
    import yaml

    resolved_path = config_path or (Path(__file__).resolve().parents[1] / "config.yaml")
    resolved_path = Path(resolved_path)
    base_dir = resolved_path.parent

    raw: dict[str, Any]
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    app_raw = raw.get("app", {})
    kb_raw = raw.get("knowledge_base", {})
    llm_raw = raw.get("llm", {})

    app = AppConfig(
        name=app_raw.get("name", "sihan-live-backend"),
        owner_id=app_raw.get("owner_id", "nac"),
        api_key=app_raw.get("api_key", "local-dev-key"),
        host=app_raw.get("host", "0.0.0.0"),
        port=int(app_raw.get("port", 8080)),
        persona_name=app_raw.get("persona_name", "赵思涵"),
    )

    allowed_roots = [_to_abs_path(base_dir, root) for root in kb_raw.get("allowed_roots", [])]
    knowledge_base = KnowledgeBaseConfig(
        storage_path=_to_abs_path(base_dir, kb_raw.get("storage_path", "./data/knowledge_store.json")),
        chunk_size=int(kb_raw.get("chunk_size", 400)),
        chunk_overlap=int(kb_raw.get("chunk_overlap", 80)),
        allowed_roots=allowed_roots,
    )

    llm = LLMConfig(
        enabled=bool(llm_raw.get("enabled", False)),
        base_url=str(llm_raw.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/"),
        model=str(llm_raw.get("model", "local-model")),
        api_key=llm_raw.get("api_key"),
        timeout=float(llm_raw.get("timeout", 30.0)),
        temperature=float(llm_raw.get("temperature", 0.7)),
        max_tokens=int(llm_raw.get("max_tokens", 512)),
    )

    return Settings(app=app, knowledge_base=knowledge_base, llm=llm)
