# coding: utf-8
"""A3 用の既知パターン生成・検出ヘルパ (ChArUco / 市松)。

A3 (絶対誤差) は「自作した既知パターンの特徴点について、真のプロジェクタ座標が
設計値として分かっている」ことを使う。本モジュールは:

1. プロジェクタ解像度の既知パターン画像を生成し、
2. その**真のプロジェクタ座標テーブル** ``{feature_id: (proj_x, proj_y)}`` を返す。

真座標は「ノイズのない生成画像に同じ検出器をかけて」得る方式とし、カメラ画像側に
同じ検出器を使えば差分が純粋に GrayCode 対応の誤差になる。ID を持つ **ChArUco を推奨**
(向き/並びの曖昧さが無い)。市松は順序依存のため向きが安定する場面向け。

ChArUco/ArUco: Garrido-Jurado et al. 2014。サブピクセルコーナー: Zhang 2000 / OpenCV。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2

    _CV2 = True
    _HAS_ARUCO = hasattr(cv2, "aruco")
except ImportError:  # pragma: no cover
    _CV2 = False
    _HAS_ARUCO = False


# ── ChArUco (推奨: ID 付きで曖昧さなし) ──────────────────────────────


def _get_dictionary(name: str = "DICT_5X5_100"):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_len: float,
    marker_len: float,
    dictionary_name: str = "DICT_5X5_100",
):
    """``cv2.aruco.CharucoBoard`` を作る。``square_len``/``marker_len`` は任意単位 (比のみ重要)。"""
    if not _HAS_ARUCO:
        raise ImportError("cv2.aruco is required (install opencv-contrib-python).")
    dictionary = _get_dictionary(dictionary_name)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_len, marker_len, dictionary
    )


def detect_charuco_corners(image: np.ndarray, board) -> tuple[np.ndarray, np.ndarray]:
    """画像から ChArUco コーナーをサブピクセル検出。``(coords(K,2) float, ids(K,) int)``。

    pixel-is-point 規約 (OpenCV のサブピクセル座標は整数=画素中心) でそのまま返す。
    """
    if not _HAS_ARUCO:
        raise ImportError("cv2.aruco is required.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return np.empty((0, 2), np.float64), np.empty((0,), np.int64)
    coords = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int64).reshape(-1)
    return coords, ids


def generate_charuco_pattern(
    proj_width: int,
    proj_height: int,
    squares_x: int = 12,
    squares_y: int = 8,
    square_px: int = 120,
    marker_ratio: float = 0.75,
    margin_px: int = 40,
    dictionary_name: str = "DICT_5X5_100",
) -> tuple[np.ndarray, dict[int, tuple[float, float]], object]:
    """プロジェクタ解像度の ChArUco 画像と真座標テーブルを返す。

    真座標は「生成したクリーン画像に :func:`detect_charuco_corners` をかけて」得るので、
    カメラ側で同じ検出器を使えば差分が純粋に GrayCode 対応誤差になる。

    Returns:
        (proj_image(H,W) uint8, {charuco_id: (proj_x, proj_y)}, board)
    """
    if not _HAS_ARUCO:
        raise ImportError("cv2.aruco is required (install opencv-contrib-python).")
    board = make_charuco_board(
        squares_x, squares_y, float(square_px), float(square_px) * marker_ratio, dictionary_name
    )
    board_w = squares_x * square_px
    board_h = squares_y * square_px
    board_img = board.generateImage((board_w, board_h))

    canvas = np.full((proj_height, proj_width), 255, dtype=np.uint8)
    ox = max(margin_px, (proj_width - board_w) // 2)
    oy = max(margin_px, (proj_height - board_h) // 2)
    h = min(board_h, proj_height - oy)
    w = min(board_w, proj_width - ox)
    canvas[oy : oy + h, ox : ox + w] = board_img[:h, :w]

    coords, ids = detect_charuco_corners(canvas, board)
    true_coords = {int(i): (float(c[0]), float(c[1])) for c, i in zip(coords, ids)}
    return canvas, true_coords, board


# ── 市松 (順序依存。向きが安定する場合のみ) ──────────────────────────


def generate_checkerboard_pattern(
    proj_width: int,
    proj_height: int,
    squares_x: int = 13,
    squares_y: int = 9,
    square_px: int = 120,
    margin_px: int = 60,
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    """プロジェクタ解像度の市松画像と内側コーナーの真座標テーブルを返す。

    内側コーナー ``(i, j)`` (i=1..squares_x-1, j=1..squares_y-1) は連続座標
    ``(ox + i*square_px - 0.5, oy + j*square_px - 0.5)`` (pixel-is-point: 画素境界)。
    ``id = (j-1)*(squares_x-1) + (i-1)`` の行優先。検出側 (:func:`detect_checkerboard_corners`)
    と ID 規約を一致させること。

    Returns:
        (proj_image(H,W) uint8, {corner_id: (proj_x, proj_y)})
    """
    canvas = np.full((proj_height, proj_width), 255, dtype=np.uint8)
    ox = max(margin_px, (proj_width - squares_x * square_px) // 2)
    oy = max(margin_px, (proj_height - squares_y * square_px) // 2)
    for j in range(squares_y):
        for i in range(squares_x):
            if (i + j) % 2 == 1:
                y0 = oy + j * square_px
                x0 = ox + i * square_px
                canvas[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    true_coords: dict[int, tuple[float, float]] = {}
    inner_x = squares_x - 1
    for j in range(1, squares_y):
        for i in range(1, squares_x):
            cid = (j - 1) * inner_x + (i - 1)
            true_coords[cid] = (
                float(ox + i * square_px - 0.5),
                float(oy + j * square_px - 0.5),
            )
    return canvas, true_coords


def detect_checkerboard_corners(
    image: np.ndarray, inner_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """市松の内側コーナーをサブピクセル検出。``inner_size=(squares_x-1, squares_y-1)``。

    ``(coords(K,2) float, ids(K,) int)`` を返す。ID は検出順 (行優先) の連番。
    向き反転の曖昧さがあるため、可能なら ChArUco を使うこと。
    """
    if not _CV2:
        raise ImportError("opencv (cv2) is required.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, inner_size)
    if not ok or corners is None:
        return np.empty((0, 2), np.float64), np.empty((0,), np.int64)
    coords = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    ids = np.arange(len(coords), dtype=np.int64)
    return coords, ids
