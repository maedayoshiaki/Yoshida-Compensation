# coding: utf-8
"""プロジェクタ投影 (projector-controller 経由)。

投影ウィンドウの作成は **projector-controller** (`projector_controller` パッケージ、
`C:/py_scripts/projector-controller`) を用いる。A3 評価で「既知パターンを投影して撮影する」
ために使う。projector_controller は任意依存 (lazy import) で、不在なら明確なエラーを出す。

projector_controller.ProjectionWindow は画像ファイルパスを ``show_image`` で表示する。
パターンをプロジェクタ解像度ちょうどで生成し、プロジェクタ表示にフルスクリーン + fit_mode
``"native"`` (等倍・無拡大) で出すと、プロジェクタ画素 = パターン画素の 1:1 投影になる。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def project_image(
    image: "np.ndarray | str | Path",
    *,
    display: int = 1,
    fullscreen: bool = True,
    position: Optional[Sequence[int]] = None,
    size: Optional[Sequence[int]] = None,
    duration: Optional[float] = None,
    fit_mode: str = "native",
) -> None:
    """projector_controller.ProjectionWindow で画像を投影する。

    Args:
        image: 投影する画像。PNG 等のファイルパス、または uint8 ndarray (BGR/グレー)。
        display: 投影先ディスプレイ番号 (プロジェクタは通常 1)。
        fullscreen: フルスクリーン投影 (プロジェクタ全面)。
        position: ``(x, y)`` デスクトップ絶対座標 (fullscreen=False のとき)。
        size: ``(width, height)`` ウィンドウサイズ (fullscreen=False のとき)。
        duration: 表示秒数。None なら Esc / ウィンドウクローズまで待つ。
        fit_mode: "native"(等倍) / "contain" / "cover" / "stretch"。パターンを
            プロジェクタ解像度で生成した場合は "native" で 1:1 投影。

    Raises:
        ImportError: projector_controller が未インストールのとき。
    """
    try:
        from projector_controller import ProjectionWindow
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "projector-controller が必要です。2dsr-prc では editable 依存として "
            "入っています (uv sync)。単体なら `uv add projector-controller`。"
        ) from e

    tmp_path: Optional[str] = None
    if isinstance(image, (str, Path)):
        img_path = str(image)
    else:
        import cv2

        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp_path, np.asarray(image))
        img_path = tmp_path

    kwargs: dict = {"display": display, "fullscreen": fullscreen, "fit_mode": fit_mode}
    if position is not None:
        kwargs["position"] = tuple(int(v) for v in position)
    if size is not None:
        kwargs["size"] = tuple(int(v) for v in size)

    try:
        with ProjectionWindow(**kwargs) as window:
            window.show_image(img_path)
            window.wait(duration)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
