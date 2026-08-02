"""``POST /api/generate``: save the CLI's projection images to disk.

このルータがするのは (a) 3つのIDを解決すること、(b) パラメータを検証して
断ること、(c) ``generate.write_projection_images`` を1回呼ぶこと、
(d) 例外をステータスコードへ写すこと の4つで、**画素にも並びにも触れない**。
保存の中身は ``generate.py``、補償の数式は
``pipeline.generate_projection_image()`` にある。

**検証をここに置く理由。** 断らなかった場合に何になるかで決めてある:

- 未知の ``linearizer`` は ``generate_projection_image`` の ``ValueError``
  ＝ 500 になる。ドロップダウンにない値を送っただけで「サーバが壊れた」と
  返るのは誤りなので 400 にする。``target_decode`` も同じ。
- ``linearizer="inverse_lut"`` で ``lut_id`` 未指定は ``_linearize_inverse_lut``
  の ``ValueError`` ＝ 500 になる。LUT を選び忘れただけである。
- カメラ座標の行列は ``calc_compensation_image`` で numpy の形状
  ``ValueError`` ＝ 500 になる。判定と断り文は
  ``views/target_view.py::ViewParams.require_projector_space`` と共有する
  （``compensation`` / ``gamut`` と同じ問いであり、写すと約20行の重複になる）。

**ステータスコードの割り振り。** 未知のIDは3つとも 404（``deps.require_*``
がそのために存在する）。``views/`` 側が未知のIDを 400 にしているのは、
そちらではIDが**クエリ文字列のパラメータ**であってリソースの指定ではない
ためで、ここは POST の本文で3つのリソースを名指ししている。``lut_id`` の
**未指定**だけは 404 ではなく 400 — 存在しないリソースではなく、
組み合わせとして成り立たない要求だからである。

共有している断り文が ``view "generate"`` と名乗る点について:
``ViewParams`` はビュー名を引数に取り、それを全ての断り文に埋める
（利用者が見る唯一の手掛かりが「どれが断ったか」であるため）。ここはビュー
ではないので語としては正確でないが、``"generate"`` と名乗る文が
``/api/generate`` から返ることは利用者にとって曖昧ではなく、
``_camera_space_refusal`` を語彙のためだけに分岐させるより、20行の推論を
1箇所に保つほうが価値が高い。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

from fastapi import APIRouter, HTTPException

from yoshidacomp.compensation.pipeline import LINEARIZERS, TargetDecodeMode
from yoshidacomp.config import get_config

from .. import generate as generate_module
from ..deps import require_lut, require_matrix, require_target
from ..models import GenerateRequest
from ..views import target_view
from ..views.base import ViewNotApplicable

router = APIRouter()

# Reads ``target_decode`` / ``target_gamma`` and refuses in this endpoint's
# name, from the same defaults ``compensation`` and ``gamut`` read. See the
# module docstring for the one word this makes slightly inaccurate.
_PARAMS = target_view.ViewParams("generate", dict(target_view.TARGET_DEFAULTS))


@router.post("/generate")
def generate_projection_files(request: GenerateRequest) -> dict[str, Any]:
    """Write one target's compensation and projection images to *output_dir*.

    保存するのは**選択中の1枚**だけである（理由は ``generate.py`` の冒頭）。
    """
    # ``Path("")`` is ``Path(".")`` -- the server's working directory, which
    # for this GUI is the repository root, with ``data/`` and ``.git`` under
    # it. The non-empty check downstream would refuse it, but "repository root
    # is not empty" explains nothing about what the user did wrong. Same
    # precedent as ``routers/session.py`` / ``target.py`` / ``lut.py``.
    if not request.output_dir.strip():
        raise HTTPException(
            status_code=400,
            detail="output_dir is empty; expected the path of a directory to "
            "write the generated images into",
        )

    matrix = require_matrix(request.matrix_id)
    target = require_target(request.target_id)

    linearizer = request.linearizer
    if linearizer not in LINEARIZERS:
        # Sourced from the pipeline's own registry, so a linearizer added there
        # is accepted here without a second edit and one removed there cannot
        # be requested here.
        raise HTTPException(
            status_code=400,
            detail=f"linearizer must be one of {list(LINEARIZERS)}, got "
            f"{linearizer!r}",
        )

    decode_modes = get_args(TargetDecodeMode)
    if request.target_decode is not None and request.target_decode not in decode_modes:
        raise HTTPException(
            status_code=400,
            detail=f"target_decode must be one of {list(decode_modes)}, got "
            f"{request.target_decode!r}",
        )

    lut = None
    if linearizer == "inverse_lut":
        if not (request.lut_id or "").strip():
            raise HTTPException(
                status_code=400,
                detail='linearizer="inverse_lut" needs a measured LUT: pass '
                "lut_id=<id>, the id returned by POST /api/lut, or choose "
                'linearizer="gamma" for the analytic inverse gamma. '
                "（inverse_lut は lut_id=<id> を必要とします。）",
            )
        lut = require_lut(request.lut_id)  # 404 for an id that is not loaded.

    projector = get_config().projector
    params = {
        "target_decode": request.target_decode,
        "target_gamma": request.target_gamma,
    }
    try:
        # Both refusals are the shared ones: a camera-space matrix has no
        # active region to place the target in, and a non-positive
        # ``target_gamma`` is a number box someone typed 0 into.
        _height, _width, active_region = _PARAMS.require_projector_space(
            matrix, projector
        )
        decode_mode, decode_gamma = _PARAMS.target_decode(params)
    except ViewNotApplicable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return generate_module.write_projection_images(
            matrix=matrix,
            target=target,
            projector=projector,
            active_region=active_region,
            output_dir=Path(request.output_dir),
            linearizer=linearizer,
            lut=lut,
            target_decode=decode_mode,
            target_gamma=decode_gamma,
            overwrite=request.overwrite,
        )
    except generate_module.OutputDirNotEmpty as exc:
        # 409, and nothing has been written. The status is the smaller half of
        # the guarantee; ``test_an_existing_directory_is_refused_with_409_and_
        # nothing_is_written`` compares the directory's contents before and
        # after, because an implementation that writes and *then* reports the
        # conflict passes a status-only assertion.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        # A mistyped path, nearly always: a drive that is not mounted, a
        # character Windows will not accept in a filename, a directory that is
        # read-only, or ``output_dir`` naming an existing *file* (``mkdir``
        # raises ``FileExistsError``/``NotADirectoryError``, both ``OSError``).
        # None of those is a server fault, and the OS message names the path,
        # which is the one thing the user needs to see.
        # 打ち間違えたパスは 500 ではない。OS のメッセージがパスを含んでおり、
        # 利用者が見るべきなのはそれである。
        raise HTTPException(
            status_code=400,
            detail=f"could not write to {request.output_dir!r}: {exc}",
        ) from exc
