"""Pydantic request bodies."""

from __future__ import annotations

from pydantic import BaseModel


class LoadMatrixRequest(BaseModel):
    """Body of ``POST /api/session/matrix``."""

    path: str


class LoadTargetRequest(BaseModel):
    """Body of ``POST /api/target``.

    ``index`` selects which image becomes the *currently selected* one when
    ``path`` is a directory (0 = first, casefold-sorted). When ``path`` is a
    single file, only ``0`` is valid -- a single file has exactly one
    selectable entry, so any other value is rejected (400), not silently
    ignored (``target.py::resolve_target``).
    """

    path: str
    index: int = 0


class LoadLutRequest(BaseModel):
    """Body of ``POST /api/lut``.

    No ``index``: an inverse LUT ``.npz`` holds exactly one table, so unlike a
    target there is nothing to select within it.
    """

    path: str
