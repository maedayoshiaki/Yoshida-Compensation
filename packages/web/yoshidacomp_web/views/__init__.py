"""View registry and the built-in views."""

from __future__ import annotations

from . import coef_tiles  # noqa: F401  registers "coef_tiles"
from . import surface  # noqa: F401  registers "surface"
from .base import (
    REGISTRY,
    ParamSpec,
    ViewNotApplicable,
    ViewResult,
    ViewSpec,
    register,
)

__all__ = [
    "REGISTRY",
    "ParamSpec",
    "ViewNotApplicable",
    "ViewResult",
    "ViewSpec",
    "register",
]
