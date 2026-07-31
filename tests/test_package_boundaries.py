"""Layering guards for the yoshidacomp workspace.

core は Web層に依存しない。core はカメラSDKなしで import できる。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_DIR = _REPO_ROOT / "packages" / "core" / "yoshidacomp"
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(yoshidacomp_web)\b", re.MULTILINE)


def test_core_does_not_import_web() -> None:
    offenders = []
    for path in _CORE_DIR.rglob("*.py"):
        if _IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"core must not import the web layer: {offenders}"


def test_core_imports_without_camera_sdk() -> None:
    """edsdk / rawpy / fastapi を遮断しても core が import できること。

    Note: the blocker uses ``find_spec`` on ``sys.meta_path``, not the
    legacy ``find_module``/``load_module`` finder protocol -- that API was
    removed from the import system in Python 3.12 and silently does
    nothing on this repo's Python 3.13 venv, which would make this test
    vacuously pass without blocking anything. The script below asserts
    the block actually took effect (``rawpy`` really is installed in this
    venv, so an unblocked import would otherwise succeed) before trusting
    the rest of the run, and re-checks that none of the blocked names
    leaked into ``sys.modules`` via a transitive import.
    """
    script = """
import sys

BLOCKED = {"edsdk", "rawpy", "fastapi", "uvicorn"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Blocker())

# Sanity-check the blocker itself before trusting anything else: rawpy is
# actually installed in this venv, so if this import succeeds the blocker
# is not working and the rest of this test would be vacuous.
try:
    import rawpy  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError("blocker failed to block rawpy; test would be vacuous")

import yoshidacomp
import yoshidacomp.config
import yoshidacomp.capture
import yoshidacomp.compensation.pipeline
import yoshidacomp.compensation.inverse_lut
import yoshidacomp.geometry.warp

leaked = BLOCKED & set(sys.modules)
assert not leaked, f"blocked modules leaked into sys.modules: {leaked}"

print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
