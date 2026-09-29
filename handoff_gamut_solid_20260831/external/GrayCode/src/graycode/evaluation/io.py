# coding: utf-8
"""外部フォーマットの対応マップ読み込み (2dsr-prc 連携)。

2dsr-prc は GrayCode 結果を ``ProjectorCameraMap`` として ``.npz`` に保存する
(`src/prc/geometry.py`)。本モジュールはその ``.npz`` を graycode.evaluation の指標が
そのまま使える対応配列に変換する。これにより 2dsr-prc 側のディレクトリで GrayCode を
撮った後、その出力 (`output/<run>/p2c.npz`) に対して評価を実行できる。

npz キー (2dsr-prc ProjectorCameraMap.save):
  - ``p2c``: (N,4) float32, 行 ``[proj_x, proj_y, cam_x, cam_y]`` (graycode 保存順)
  - ``proj_size``: (2,) int, ``[height, width]``
  - ``coord_convention``: 文字列, 期待値 ``"pixel-is-point"`` (旧 UV 規約は警告)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_projector_camera_map_npz(
    path: str | Path,
) -> tuple[np.ndarray, tuple[int, int] | None, str | None]:
    """2dsr-prc の ``ProjectorCameraMap`` .npz を読み込む。

    Returns:
        (correspondences, proj_size, coord_convention)
        - correspondences: (N,4) float64 ``[cam_x, cam_y, proj_x, proj_y]``
          (graycode.evaluation の指標が期待する列順に並べ替え済み)。
        - proj_size: (height, width) or None。
        - coord_convention: 文字列 or None。"pixel-is-point" 以外なら呼び出し側で警告。
    """
    p = Path(path)
    data = np.load(str(p), allow_pickle=False)
    keys = set(data.files)
    if "p2c" not in keys:
        raise ValueError(
            f"{p}: 'p2c' キーがありません (ProjectorCameraMap .npz ではない)。keys={sorted(keys)}"
        )
    # p2c: [proj_x, proj_y, cam_x, cam_y] → corr: [cam_x, cam_y, proj_x, proj_y]
    p2c = np.asarray(data["p2c"], dtype=np.float64).reshape(-1, 4)
    corr = p2c[:, [2, 3, 0, 1]].copy()

    proj_size: tuple[int, int] | None = None
    if "proj_size" in keys:
        ps = np.asarray(data["proj_size"]).ravel()
        if ps.size >= 2:
            proj_size = (int(ps[0]), int(ps[1]))  # (height, width)

    conv: str | None = None
    if "coord_convention" in keys:
        try:
            conv = str(np.asarray(data["coord_convention"]).item())
        except Exception:
            conv = str(data["coord_convention"])
    return corr, proj_size, conv


def p2c_grid_from_correspondences(corr: np.ndarray) -> np.ndarray:
    """corr ``[cam_x,cam_y,proj_x,proj_y]`` を P2C 格子用の ``[proj_x,proj_y,cam_x,cam_y]`` に戻す。

    2dsr-prc の p2c はプロジェクタ格子上で密 (N=projH*projW, ラスタ順) なので、これを
    そのまま A1 (cycle) の dense_p2c として渡せる (列順を戻すだけ)。
    """
    c = np.asarray(corr, dtype=np.float64)
    return c[:, [2, 3, 0, 1]]
