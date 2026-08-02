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


def test_web_does_not_import_matplotlib_or_qt() -> None:
    """Web層は描画ライブラリを引かない（描画はブラウザ側）。"""
    import re

    web_dir = _REPO_ROOT / "packages" / "web" / "yoshidacomp_web"
    pattern = re.compile(r"^\s*(?:from|import)\s+(matplotlib|PyQt6|PyQt5)\b", re.MULTILINE)
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in web_dir.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"web must not import a plotting stack: {offenders}"


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


def test_importing_pipeline_does_not_pull_torch() -> None:
    """``pipeline`` の import だけでは torch / psutil / color_mixing_matrix /
    scipy を引かないこと。

    Moved here from ``tests/test_pipeline_gamma_numpy.py`` (Task 1 of the
    target-and-lut plan): this is a layering guard like the rest of this
    file, not a gamma-linearizer-specific test.

    History (see ``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md``
    for full measurements): this started out weaker -- only asserting
    ``color_mixing_matrix`` and ``psutil`` were absent -- because
    ``yoshidacomp.geometry.warp`` (also imported by ``pipeline``, for
    ``prepare_target_image_for_compensation()``) used to do an
    unconditional ``import torch`` at its own module scope. Fixed by moving
    that import inside the two functions that use it (mirroring the
    lazy-import-per-function pattern this same file already uses for
    ``graycode``), which changes no module attribute so
    ``tests/test_sample_mocked.py``'s ``warp_image``/``invwarp_image``
    monkeypatches on the ``pipeline`` module keep working. A further
    ``scipy.interpolate`` import (in ``inverse_lut.py``, used only by LUT
    *construction*, not the ``apply_inverse_lut`` hot path) was fixed the
    same way. Net: ``import yoshidacomp.compensation.pipeline`` went from
    ~9.4-13.0s to ~0.5s wall time.

    This test covers *import* only. ``generate_projection_image`` still
    pulls torch on its *first call* regardless of linearizer, because
    ``calc_compensation_image`` (the compensation matrix multiply, called
    unconditionally) needs it -- see
    ``test_generate_projection_image_gamma_call_pulls_only_what_the_matrix_step_needs``
    below, which covers that.

    Run in a subprocess: in-process, some other test collected earlier in
    the same session may have already imported these modules, which would
    make the assertion pass regardless of whether ``pipeline.py`` (or
    ``warp.py``/``inverse_lut.py``) is actually responsible.
    """
    code = (
        "import sys; import yoshidacomp.compensation.pipeline; "
        "mods = sys.modules; "
        "print('color_mixing_matrix=%s' % any('color_mixing_matrix' in m for m in mods)); "
        "print('psutil=%s' % ('psutil' in mods)); "
        "print('geometry.warp=%s' % ('yoshidacomp.geometry.warp' in mods)); "
        "print('torch=%s' % ('torch' in mods)); "
        "print('scipy=%s' % any('scipy' in m for m in mods)); "
        "print('cv2=%s' % ('cv2' in mods))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    reported = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())

    assert reported["color_mixing_matrix"] == "False", out.stdout
    assert reported["psutil"] == "False", out.stdout
    assert reported["torch"] == "False", out.stdout
    assert reported["scipy"] == "False", out.stdout
    # geometry.warp and cv2 are legitimately still imported -- pipeline.py
    # needs WarpMethod/invwarp_image/warp_image from warp, and cv2 is an
    # existing, accepted dependency -- but none of them may bring torch,
    # psutil, or scipy along for the ride.
    assert reported["geometry.warp"] == "True", out.stdout
    assert reported["cv2"] == "True", out.stdout


def test_generate_projection_image_gamma_call_pulls_only_what_the_matrix_step_needs() -> None:
    """``generate_projection_image(..., linearizer="gamma")`` の初回呼び出し後、
    ``psutil``・``color_mixing_matrix``・``scipy`` は sys.modules に無いこと。

    This is the test the review that reversed Task 1's original "keep
    torch" decision asked for: import-only coverage let an 8s-plus,
    CUDA-initializing *first call* stall through even though the module
    import itself was fast. ``_linearize_gamma`` now calls
    ``_apply_inverse_gamma_numpy`` (numpy), not the torch implementation,
    so the gamma step itself contributes none of ``torch``/``psutil``/
    ``color_mixing_matrix``.

    **``torch`` is deliberately NOT asserted absent here -- it will be
    present, and that is correct, not a regression.**
    ``calc_compensation_image`` (``photometric_compensation.py``), the
    compensation *matrix-multiply* step, is called unconditionally by
    ``generate_projection_image`` regardless of which linearizer is
    selected -- confirmed by measuring ``linearizer="none"`` alone, which
    pulls torch and costs the same ~7.7s as ``linearizer="gamma"`` did
    before this task's numpy switch. Profiling that cost (see the task
    report) found it is overwhelmingly the cost of ``import torch`` itself
    (~7.1s of it, independent of CUDA specifically: ``torch.cuda.is_available()``
    is ~0.15s and the first real CUDA op is ~0.38s more) -- a cost that was
    always going to be paid on any real (non-``"none"``) call to
    ``generate_projection_image``, before or after this task, because the
    matrix step is genuine GPU-scale linear algebra, not the trivial
    elementwise power the brief singled out gamma correction for. Switching
    the gamma linearizer to numpy measurably shrinks the first-call cost
    (~7.9s to ~7.6-7.7s -- the marginal cost of the old gamma-specific
    torch call plus the ``color_mixing_matrix``/``psutil`` import it used to
    trigger) but does **not** eliminate the dominant stall; that remains
    ``calc_compensation_image``'s and is out of this task's scope (see the
    task report's "Addendum 3" section for the full numbers and why this
    was not also rewritten here).

    Run in a subprocess for the same reason as the import-only test above.
    """
    import numpy as np

    code = (
        "import sys\n"
        "import numpy as np\n"
        "from dataclasses import dataclass\n"
        "from yoshidacomp.compensation.pipeline import generate_projection_image\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Projector:\n"
        "    width: int = 4\n"
        "    height: int = 4\n"
        "    gamma: float = 2.2\n"
        "\n"
        "matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)\n"
        "for c in range(3):\n"
        "    matrices[:, :, c, c] = 1.0\n"
        "target = np.full((4, 4, 3), 200, dtype=np.uint8)\n"
        "generate_projection_image(\n"
        "    target, matrices, projector=Projector(), active_region=(0, 0, 4, 4),\n"
        "    linearizer='gamma', target_decode='none',\n"
        ")\n"
        "mods = sys.modules\n"
        "print('color_mixing_matrix=%s' % any('color_mixing_matrix' in m for m in mods))\n"
        "print('psutil=%s' % ('psutil' in mods))\n"
        "print('scipy=%s' % any('scipy' in m for m in mods))\n"
        "print('torch=%s' % ('torch' in mods))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    reported = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())

    assert reported["color_mixing_matrix"] == "False", out.stdout
    assert reported["psutil"] == "False", out.stdout
    assert reported["scipy"] == "False", out.stdout
    # See docstring: torch is expected here, from calc_compensation_image,
    # not from the gamma step.
    assert reported["torch"] == "True", out.stdout
