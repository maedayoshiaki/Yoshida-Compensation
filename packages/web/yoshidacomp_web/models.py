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


class GenerateRequest(BaseModel):
    """Body of ``POST /api/generate``.

    ``linearizer`` は必須にしてある。CLI の ``--linearizer`` が
    ``required=True`` であるのと同じ理由で、これは「省くと既定で走る」もので
    はない -- 省略を ``"none"`` として受けると、ガンマのつもりで押した保存が
    黙って線形化なしの PNG を書く。既定値を持つのは復号側だけである。

    ``target_decode`` / ``target_gamma`` / ``lut_id`` が ``None`` を既定に
    しているのは、**既定値をここに書かないため**である。実際の既定は
    ``views/target_view.py::TARGET_DEFAULTS``（``"srgb"`` / ``2.2``）で、
    ``ViewParams.value`` が ``None`` をそれで埋める。ここに ``= "srgb"`` と
    書くと同じ値が2箇所になり、``compensation`` ビューと ``/api/generate``
    が別の既定で動く状態が静かに作れてしまう -- この繰り返しが
    ``TARGET_DEFAULTS`` を作らせた欠陥そのものである。

    ``overwrite`` の既定は ``False``。既定で上書きしないのではなく、
    **明示されない限り上書きしない**（``generate.py`` の冒頭を参照）。
    """

    matrix_id: str
    target_id: str
    output_dir: str
    linearizer: str
    lut_id: str | None = None
    target_decode: str | None = None
    target_gamma: float | None = None
    overwrite: bool = False
