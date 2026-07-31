"""Shared FastAPI dependencies / helpers."""

from __future__ import annotations

from fastapi import HTTPException

from .matrix import MatrixContext
from .store import store


def require_matrix(matrix_id: str) -> MatrixContext:
    """Return the stored matrix context or raise 404."""
    try:
        return store.get(matrix_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"matrix not found: {matrix_id}"
        ) from exc
