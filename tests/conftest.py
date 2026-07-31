"""Test bootstrap for the vendored GrayCode src layout."""

from __future__ import annotations

import sys
from pathlib import Path

_GRAYCODE_SRC = Path(__file__).resolve().parents[1] / "external" / "GrayCode" / "src"
if str(_GRAYCODE_SRC) not in sys.path:
    sys.path.insert(0, str(_GRAYCODE_SRC))
