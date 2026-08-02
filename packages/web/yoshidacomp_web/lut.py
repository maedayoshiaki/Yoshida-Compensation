"""Measured inverse LUT loading (path-only, like the matrix and the target).

``.npz`` を1つ読み、``yoshidacomp.compensation.inverse_lut.load_inverse_lut``
が返す ``InverseLut`` をそのまま持つ。線形化の実体はコア側にあり、この層は
「読む・検証する・IDを配る」だけを行う。

**なぜ検証するのか。** ``load_inverse_lut`` は ``lut``/``t_grid`` を
``np.asarray(..., dtype=np.float32)`` するだけで、形も中身も見ない。形が
違っても読み込みは成功し、失敗するのは ``apply_inverse_lut`` の
``lut.lut[channel]`` — すなわち**補償を描こうとした瞬間**であり、そこは
``views/compensation.py`` の中なので ``IndexError`` はビューの 500 になる。
「Task 3 で読み込んだ LUT の形が違った」を「サーバが壊れた」として返すことに
なるので、読み込み時に断る。断る側の例外は ``ValueError`` に揃え、router が
400 へ写す（``routers/session.py``/``routers/target.py`` と同じ規約）。

``load_inverse_lut`` を import しても scipy は来ない。計画B3 Task 1 が
``PchipInterpolator`` の import を ``_build_channel_lut``（LUT*構築*時のみ）
の中へ遅延させたため、読み込み経路には残っていない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yoshidacomp.compensation.inverse_lut import InverseLut, load_inverse_lut

# The only extension this endpoint reads. ``load_inverse_lut`` uses
# ``with np.load(...)``, and ``np.load`` on a ``.npy`` returns a bare ndarray --
# which is not a context manager, so the failure is ``TypeError: 'numpy.ndarray'
# object does not support the context manager protocol``. That is a true
# sentence about the code and tells the user nothing about their mistake, which
# was pointing at the wrong kind of file. Same precedent as ``target.py``'s
# ``SUPPORTED_TARGET_IMAGE_SUFFIXES`` check.
# 読むのは ``.npz`` だけ。``.npy`` を渡すと ``np.load`` が素の ndarray を返し、
# ``with`` が ``TypeError`` になる——コードとしては正しい文だが、利用者の誤り
# （ファイルの種類違い）については何も言っていない。
_SUFFIX = ".npz"

# The channel count an inverse LUT has, by construction: ``apply_inverse_lut``
# indexes ``lut.lut[channel]`` for channel in range(3).
# 逆LUTのチャンネル数は3で固定される（``apply_inverse_lut`` が0..2で引く）。
_CHANNELS = 3


@dataclass(frozen=True)
class LutContext:
    """A loaded inverse LUT plus where it came from.

    Attributes:
        lut: The ``InverseLut`` as ``load_inverse_lut`` returned it -- passed
            to ``generate_projection_image(lut=...)`` unchanged, so the GUI and
            the CLI linearize through the same object.
            ``load_inverse_lut`` の戻り値そのもの。
        path: The ``.npz`` that was posted. 元パス。
    """

    lut: InverseLut
    path: Path


def _validate(lut: InverseLut, path: Path) -> None:
    """Refuse an ``.npz`` that parsed but is not an inverse LUT.

    Raises:
        ValueError: the arrays are not ``lut (3, R)`` / ``t_grid (R,)`` with
            ``R >= 2``, or either holds a non-finite value. The message always
            names ``path`` and what was actually found, since the router hands
            it straight to the user.
    """
    table = np.asarray(lut.lut)
    grid = np.asarray(lut.t_grid)
    if table.ndim != 2 or table.shape[0] != _CHANNELS:
        raise ValueError(
            f"{path}: 'lut' must be ({_CHANNELS}, R) -- one row per channel -- "
            f"but its shape is {tuple(table.shape)}"
        )
    if grid.ndim != 1:
        raise ValueError(
            f"{path}: 't_grid' must be one-dimensional (R,), but its shape is "
            f"{tuple(grid.shape)}"
        )
    if table.shape[1] != grid.shape[0]:
        raise ValueError(
            f"{path}: 'lut' has {table.shape[1]} sample(s) per channel but "
            f"'t_grid' has {grid.shape[0]}; they must be the same length"
        )
    if grid.shape[0] < 2:
        raise ValueError(
            f"{path}: an inverse LUT needs at least 2 samples, found "
            f"{grid.shape[0]}"
        )
    # ``np.interp`` does not raise on NaN -- it propagates it -- so a LUT with a
    # hole in it would linearize the whole frame to NaN and be drawn as the
    # out-of-region mask colour: a picture of "everything is masked" with no
    # route back to "the LUT file has NaN in it".
    # ``np.interp`` は NaN を伝播するだけなので、穴のあるLUTはフレーム全体を
    # NaN にし、領域外マスクと同じ色で描かれる（原因へ辿れない）。
    if not np.isfinite(table).all() or not np.isfinite(grid).all():
        raise ValueError(
            f"{path}: the LUT holds non-finite values (NaN or inf); "
            "np.interp would propagate them into every pixel"
        )


def load_lut(path: Path) -> LutContext:
    """Load a measured inverse LUT from a ``.npz``.

    Raises:
        FileNotFoundError: ``path`` does not exist. 存在しないパス。
        ValueError: ``path`` exists but cannot become an inverse LUT -- a
            directory, a suffix other than ``.npz``, an archive numpy cannot
            read, one without a ``lut``/``t_grid`` entry, or arrays of the
            wrong shape or contents (see :func:`_validate`). All of these are
            routine mistakes (the wrong file pasted into the box), not
            exceptional, and the router turns them into a 400.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise ValueError(
            f"not a file: {path} -- expected an inverse LUT {_SUFFIX} file"
        )
    if path.suffix.lower() != _SUFFIX:
        raise ValueError(
            f"not a supported inverse LUT file: {path} -- expected {_SUFFIX}"
        )
    try:
        loaded = load_inverse_lut(path)
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        # Every one of these is "this file is not the thing you said it was":
        # ValueError from a corrupt archive, KeyError for a ``.npz`` without a
        # ``lut`` entry, TypeError/OSError/EOFError from a truncated or
        # mislabelled one. None of them is a bug in this server, so none may be
        # a 500. The original message is kept -- numpy's is specific about
        # which of those it is.
        # いずれも「指定されたファイルが逆LUTではない」であり、サーバのバグでは
        # ないので 500 にしない。numpy の元メッセージは具体的なので残す。
        raise ValueError(f"could not read {path} as an inverse LUT: {exc}") from exc
    _validate(loaded, path)
    return LutContext(lut=loaded, path=path)
