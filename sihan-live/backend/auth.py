"""Single-owner header authentication.

The service is meant for one person ("仅本人可用"). Every request must carry
``X-Owner-Id`` and ``X-API-Key`` headers matching the configured owner; the
resolved identity is handed to route handlers as a :class:`UserContext`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import Header, HTTPException, status

from .config import Settings


@dataclass(slots=True)
class UserContext:
    owner_id: str


def build_owner_guard(settings: Settings) -> Callable[..., UserContext]:
    """Build a FastAPI dependency that enforces single-owner access."""

    def require_owner(
        x_owner_id: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> UserContext:
        if x_owner_id != settings.app.owner_id or x_api_key != settings.app.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: owner identity or API key mismatch.",
            )
        return UserContext(owner_id=x_owner_id)

    return require_owner
