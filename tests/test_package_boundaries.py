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
    vacuously pass without blocking anything. The script below therefore
    self-checks the blocker before trusting the rest of the run, and
    re-checks that none of the blocked names leaked into ``sys.modules``
    via a transitive import.

    The self-check probes ``PIL``, not one of the real targets. Probing
    ``rawpy`` -- which is what the blocker exists to block -- proves
    nothing in a venv installed without the optional ``camera`` extra:
    the import would fail because the package is absent, the assertion
    would pass, and the blocker could be completely inert. ``PIL`` is a
    module that is *guaranteed present* (pillow is a hard dependency of
    ``yoshidacomp``) and *never imported by core* (zero references under
    ``packages/core/yoshidacomp``), so the script can first confirm it is
    installed and then confirm the blocker makes it unimportable.

    自己チェックが実際の遮断対象ではなく ``PIL`` を使う理由: ``rawpy`` は
    遮断対象そのものなので、任意 extra ``camera`` 抜きの venv では単に
    未インストールで import が失敗し、遮断器が完全に無効でもアサーション
    が通ってしまう。``PIL`` は必ず存在し（pillow は ``yoshidacomp`` の必須
    依存）、かつ core からは一切 import されない（``packages/core/
    yoshidacomp`` 配下に参照ゼロ）ため、インストール済みであることを確認
    した上で遮断されることを確認できる。

    Includes ``yoshidacomp.calibration.{cap_chess,solve_pnp,cam_in_calib}``
    in addition to the modules this test originally covered: those two
    scripts had a module-scope ``from edsdk.camera_controller import
    CameraController`` (carried over verbatim from ``src/python/``) that
    this test previously didn't exercise, silently violating the plan's
    "rawpy / edsdk-python stay behind the optional `camera` extra, never
    imported from module scope" constraint.
    """
    script = """
import importlib.util
import sys

BLOCKED = {"edsdk", "rawpy", "fastapi", "uvicorn"}

# Probe module for the blocker self-check: guaranteed installed (pillow is a
# required dependency of yoshidacomp) and never imported by core.
PROBE = "PIL"

if importlib.util.find_spec(PROBE) is None:
    raise AssertionError(
        f"{PROBE} must be installed for the blocker self-check to prove "
        "anything; pillow is a required dependency of yoshidacomp"
    )


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED | {PROBE}:
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Blocker())

# Sanity-check the blocker itself before trusting anything else. PIL was just
# confirmed to be installed, so if this import still succeeds the blocker is
# inert and the rest of this test would be vacuous.
try:
    import PIL  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError(f"blocker failed to block {PROBE}; test would be vacuous")

import yoshidacomp
import yoshidacomp.config
import yoshidacomp.capture
import yoshidacomp.compensation.pipeline
import yoshidacomp.compensation.inverse_lut
import yoshidacomp.geometry.warp
import yoshidacomp.calibration.cap_chess
import yoshidacomp.calibration.solve_pnp
import yoshidacomp.calibration.cam_in_calib

leaked = (BLOCKED | {PROBE}) & set(sys.modules)
assert not leaked, f"blocked modules leaked into sys.modules: {leaked}"

print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
