# coding: utf-8
"""ピクセル座標変換の単一の真実源 (single source of truth)。

このプロジェクトは内部で **単一のピクセル規約 "pixel-is-point"** を用いる:
**整数 i = 画素 i の中心**（画素 i は [i-0.5, i+0.5) を占める）。これはカメラ・
プロジェクタの双方に共通で、OpenCV / numpy / scipy / matplotlib と同じ標準規約。

唯一の例外は GPU サンプリング (``torch.nn.functional.grid_sample``) との境界で、
そこだけ "pixel-is-area / texel-center"（中心 = i+0.5）規約への変換が必要になる。
その半画素変換は :func:`point_to_normalized` 1 か所に閉じ込めてある。

座標↔画素インデックスの変換はすべてこのモジュールの関数を経由すること
（各所で再実装しない）。各関数の docstring が「正準定義」。背景・業界比較・移行履歴は
``COORDINATES.md`` を参照。

────────────────────────────────────────────────────────────────────
規約のまとめ
────────────────────────────────────────────────────────────────────
- 画素 i の中心 = 整数 i。画素 i の範囲 = [i-0.5, i+0.5)。
- 連続座標 c → 画素インデックス = round(c) = floor(c + 0.5)  (:func:`to_pixel`)。
- 全 n 画素の中心座標 = {0, 1, ..., n-1} = arange(n)  (:func:`pixel_centers`)。
- GrayCode の step=s: 縮小座標 g 1 つが s 画素のブロックを表す。代表座標は
  ブロック中心 = s*g + (s-1)/2  (:func:`block_center`、pixel-is-point)。
- grid_sample(align_corners=False) へ渡すときだけ texel 規約へ:
  正規化座標 = 2*(p+0.5)/size - 1  (:func:`point_to_normalized`)。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pixel_centers",
    "to_pixel",
    "point_to_normalized",
    "reduced_size",
    "block_of",
    "block_center",
]


def _floor(a):
    """backend 非依存の floor。numpy 配列/スカラ と torch テンソルの双方に対応。"""
    floor = getattr(a, "floor", None)
    if callable(floor):  # torch.Tensor は .floor() を持つ
        return a.floor()
    return np.floor(a)


# ── 画素中心 = 整数 (pixel-is-point, カメラ・プロジェクタ共通) ────────


def pixel_centers(n: int) -> np.ndarray:
    """全 ``n`` 画素の中心座標 (= 0, 1, ..., n-1)。

    画素 i の中心が整数 i なので、画素中心の座標列は arange そのもの。
    補間のクエリ格子（カメラ/プロジェクタの全画素）を作るのに使う。
    """
    return np.arange(n, dtype=np.float64)


def to_pixel(c):
    """連続座標 → 画素インデックス (float, 未キャスト)。

    画素中心 = 整数なので最近傍画素は round(c) = floor(c + 0.5)。
    返り値は float (numpy なら ``.astype(np.int32)``、torch なら ``.long()`` で
    呼び出し側がキャストする)。入力 dtype/backend をそのまま保つ。
    """
    return _floor(c + 0.5)


def point_to_normalized(p, size: int):
    """pixel-is-point 座標 → ``F.grid_sample(align_corners=False)`` 用の正規化座標。

    grid_sample(align_corners=False) は内部で texel 規約 (画素 i の中心 = i+0.5) を
    用い、正規化座標 g と配列画素中心座標 q を
        q = ((g + 1) * size - 1) / 2
    で対応づける。pixel-is-point 座標 p は texel 座標で p+0.5 に当たるので、
    q = p+0.5 を解いて
        g = 2*(p + 0.5)/size - 1
    を得る。**この +0.5 が pixel-is-point ↔ texel 規約の唯一の境界変換**である
    (導出は COORDINATES.md)。
    """
    return 2.0 * (p + 0.5) / size - 1.0


# ── GrayCode の step (ブロック / 縮小解像度) ─────────────────────────


def reduced_size(full: int, step: int) -> int:
    """全解像度 ``full`` を step で縮小したグレイコード解像度。

    ``(full - 1) // step + 1``。step 画素ごとに 1 つのグレイコード値を割り当てる
    ときに必要なブロック数。gen_graycode と decode が同じ値を使う必要がある。
    """
    return (full - 1) // step + 1


def block_of(pixel, step: int):
    """全解像度の画素インデックス → 所属するグレイコードブロック (= pixel // step)。

    パターン生成時のブロック展開 ``img[y, x] = pat[y // hs, x // ws]`` に対応。
    """
    return pixel // step


def block_center(g, step: int, full: int | None = None):
    """縮小グレイコード座標 ``g`` → ブロック中心の座標 (pixel-is-point)。

    decode が記録するプロジェクタ座標 (GT) の定義。ブロック g は全解像度画素
    [step*g, step*g+step-1] を覆い、その中心が ``step*g + (step-1)/2``
    (整数=画素中心 規約)。

    - step=1: g （画素 g の中心、デコード値そのもの）。
    - step が奇数: 整数 (ブロック中央の画素中心)。
    - step が偶数: 半整数 (隣り合う2画素中心の中間)。いずれも pixel-is-point 座標
      として正しい点推定 (一様照射ブロックの幾何中心)。

    ``full`` (全解像度) を与えると、最終ブロックが部分的 (full が step の倍数でない)
    な場合に、実際に覆う画素 [step*g, min(step*g+step, full)-1] の中心を返し、
    座標が [0, full-1] を超えないようにする。full=None なら従来式 (上記)。
    """
    lo = step * g
    hi = lo + step - 1
    if full is not None:
        hi = np.minimum(hi, full - 1)
    return (lo + hi) / 2
