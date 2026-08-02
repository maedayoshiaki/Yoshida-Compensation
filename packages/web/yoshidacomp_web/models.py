"""Pydantic request bodies."""

from __future__ import annotations

from pydantic import BaseModel


class LoadMatrixRequest(BaseModel):
    """Body of ``POST /api/session/matrix``."""

    path: str


class LoadTargetRequest(BaseModel):
    """Body of ``POST /api/target``.

    ``index`` selects which image becomes the *currently selected* one when
    ``path`` is a directory (0 = first, casefold-sorted). Ignored when
    ``path`` is a single file.
    """

    path: str
    index: int = 0
