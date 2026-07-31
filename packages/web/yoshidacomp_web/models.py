"""Pydantic request bodies."""

from __future__ import annotations

from pydantic import BaseModel


class LoadMatrixRequest(BaseModel):
    """Body of ``POST /api/session/matrix``."""

    path: str
