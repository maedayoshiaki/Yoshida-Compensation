# coding: UTF-8

import sys
import glob
import re
from pathlib import Path
from typing import List, Tuple

from collections import defaultdict

import cv2
import numpy as np

from . import coords
from .config import (
    get_config,
    reload_config,
    resolve_output_path,
    split_cli_config_path,
)


def print_usage() -> None:
    print(
        "Usage : python decode.py "
        "<projector image height> <projector image width> "
        "[graycode height_step(default=1)] [graycode width_step(default=1)] "
        "[--config <config.toml>]"
    )
    print()


def numerical_sort(text: str, re_num: re.Pattern[str]) -> int:
    """キャプチャファイル名中の連番でソートするためのキーを返す。"""
    return int(re_num.split(text)[-2])


def load_images(pattern: str) -> List[np.ndarray]:
    """指定パターンの画像をグレースケールで読み込む。"""
    filenames = sorted(glob.glob(pattern))
    return [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]


def save_decode_debug(
    debug_dir: str | Path,
    bit_imgs: List[np.ndarray],
    white: np.ndarray,
    black: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """撮影グレイコード・白黒参照・有効画素マスクを ``debug_dir`` に保存する。

    in-memory のライブ校正でも生キャプチャと投影範囲マスクを後から可視化できるよう、
    decode の入力をそのまま書き出す (capture_NN.png / white.png / black.png /
    valid_mask.png)。可視化は ``scripts/visualize_graycode.py`` 参照。
    """
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(bit_imgs):
        cv2.imwrite(str(d / f"capture_{i:02d}.png"), img)
    cv2.imwrite(str(d / "white.png"), white)
    cv2.imwrite(str(d / "black.png"), black)
    cv2.imwrite(str(d / "valid_mask.png"), (valid_mask.astype(np.uint8) * 255))


def decode_c2p(
    bit_imgs: List[np.ndarray],
    white: np.ndarray,
    black: np.ndarray,
    *,
    proj_height: int,
    proj_width: int,
    height_step: int = 1,
    width_step: int = 1,
    black_threshold: int,
    white_threshold: int,
    debug_dir: str | Path | None = None,
) -> tuple[List[Tuple[Tuple[int, int], Tuple[float, float]]], Tuple[int, int]]:
    """Decode in-memory gray-code captures into a camera→projector correspondence.

    Pure (no file I/O) core of :func:`main`, reusable for live calibration.
    ``bit_imgs`` are the grayscale gray-code bit captures (white/black already
    removed); ``white``/``black`` are the reference captures. Returns
    ``(c2p_list, (cam_height, cam_width))`` with ``c2p_list`` =
    ``[((cam_x, cam_y), (proj_x, proj_y)), ...]`` using the same
    sub-block-centered projector-coordinate convention as the CLI.

    If ``debug_dir`` is given, the raw captures / white / black / valid mask are
    written there (via :func:`save_decode_debug`) for later visualization.
    """
    gc_width = coords.reduced_size(proj_width, width_step)
    gc_height = coords.reduced_size(proj_height, height_step)
    graycode = cv2.structured_light.GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(black_threshold)
    graycode.setWhiteThreshold(white_threshold)

    cam_height, cam_width = white.shape[:2]
    diff = white.astype(np.int16) - black.astype(np.int16)
    valid_mask = diff > black_threshold
    if debug_dir is not None:
        save_decode_debug(debug_dir, bit_imgs, white, black, valid_mask)
    ys, xs = np.where(valid_mask)

    c2p_list: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []
    for x, y in zip(xs, ys):
        err, proj_pix = graycode.getProjPixel(bit_imgs, int(x), int(y))
        if not err:
            fixed_pix = (
                coords.block_center(proj_pix[0], width_step, proj_width),
                coords.block_center(proj_pix[1], height_step, proj_height),
            )
            c2p_list.append(((int(x), int(y)), fixed_pix))
    if debug_dir is not None:
        # 生(穴埋め前)の c2p も保存 → 可視化で「穴埋めなし」プロジェクタ視点や
        # densify 済みカメラ視点の元データに使える。
        np.save(Path(debug_dir) / "result_c2p.npy", np.array(c2p_list, dtype=object))
    return c2p_list, (cam_height, cam_width)


def main(argv: list[str] | None = None) -> tuple[int, int] | None:
    if argv is None:
        argv = sys.argv
    try:
        argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(e)
        print_usage()
        return None

    if config_path is not None:
        reload_config(config_path)

    # 3個: decode.py H W
    # 5個: decode.py H W height_step width_step
    if len(argv) not in (3, 5):
        print_usage()
        return

    try:
        proj_height = int(argv[1])
        proj_width = int(argv[2])
        height_step = int(argv[3]) if len(argv) == 5 else 1
        width_step = int(argv[4]) if len(argv) == 5 else 1
    except ValueError:
        print("height, width, height_step, width_step は整数で指定してください。")
        print_usage()
        return

    cfg = get_config()
    captured_dir = Path(cfg.paths.captured_dir)
    black_thr = cfg.decode.black_threshold
    white_thr = cfg.decode.white_threshold

    gc_width = coords.reduced_size(proj_width, width_step)
    gc_height = coords.reduced_size(proj_height, height_step)

    graycode = cv2.structured_light.GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    re_num = re.compile(r"(\d+)")

    filenames = sorted(
        glob.glob(str(captured_dir / "capture_*.png")),
        key=lambda t: numerical_sort(t, re_num),
    )

    expected_num = graycode.getNumberOfPatternImages() + 2
    if len(filenames) != expected_num:
        print(f"Number of images is not right (right number is {expected_num})")
        return

    imgs: List[np.ndarray] = [
        cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in filenames
    ]
    black = imgs.pop()
    white = imgs.pop()
    cam_height, cam_width = white.shape[:2]
    print("camera image size :", white.shape)
    print()

    viz_c2p = np.zeros((cam_height, cam_width, 3), np.uint8)

    # 差分画像をNumPyで一括計算して「候補ピクセル」のマスクを作成
    diff = white.astype(np.int16) - black.astype(np.int16)
    valid_mask = diff > black_thr

    # 有効画素のインデックスをまとめて取得
    ys, xs = np.where(valid_mask)

    # valid maskを保存
    cv2.imwrite(
        str(resolve_output_path("valid_mask.png")),
        (valid_mask * 255).astype(np.uint8),
    )

    c2p_list: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []

    # OpenCV の GrayCodePattern は現状ピクセル単位でしか問い合わせられないため
    # getProjPixel 呼び出し自体はループになるが、Python 側での条件分岐等は最小限にする
    for x, y in zip(xs, ys):
        err, proj_pix = graycode.getProjPixel(imgs, x, y)
        if not err:
            # プロジェクタ座標をステップサイズ分拡大して中心にオフセットをかける
            # これにより，得られたプロジェクタ座標（ブロック）の中心を指すようになる

            # 【修正箇所】
            # proj_pix[0] は X座標(width方向) なので width_step を掛ける
            # proj_pix[1] は Y座標(height方向) なので height_step を掛ける
            fixed_pix = (
                coords.block_center(proj_pix[0], width_step, proj_width),
                coords.block_center(proj_pix[1], height_step, proj_height),
            )

            viz_c2p[y, x, :] = [
                fixed_pix[0] % (np.iinfo(viz_c2p.dtype).max + 1),
                fixed_pix[1] % (np.iinfo(viz_c2p.dtype).max + 1),
                np.iinfo(viz_c2p.dtype).max // 2,
            ]
            c2p_list.append(((x, y), fixed_pix))

    print("=== Result ===")
    print(f"Decoded c2p correspondences : {len(c2p_list)}")
    viz_path = resolve_output_path("visualize_c2p.png")
    cv2.imwrite(str(viz_path), viz_c2p)
    print(f"Visualized image : '{viz_path}'")
    c2p_csv = resolve_output_path("result_c2p.csv")
    with open(c2p_csv, "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for (cam_x, cam_y), (proj_x, proj_y) in c2p_list:
            f.write(f"{cam_x}, {cam_y}, {proj_x}, {proj_y}\n")

    # NumPy形式でも保存
    c2p_npy = resolve_output_path("result_c2p.npy")
    np.save(c2p_npy, np.array(c2p_list, dtype=object))

    print(f"NumPy array : '{c2p_npy}'")
    print(f"output : '{c2p_csv}'")
    print()

    # --- P2C (Projector → Camera) マップ生成 ---
    # 同じプロジェクタ座標に複数のカメラ座標が対応する場合も全て保持する
    p2c_dict: dict[Tuple[float, float], List[Tuple[int, int]]] = defaultdict(list)
    for (cam_x, cam_y), (proj_x, proj_y) in c2p_list:
        p2c_dict[(proj_x, proj_y)].append((cam_x, cam_y))

    # CSV保存（1対応1行）
    p2c_csv = resolve_output_path("result_p2c.csv")
    with open(p2c_csv, "w", encoding="utf-8") as f:
        f.write("proj_x, proj_y, cam_x, cam_y\n")
        for (proj_x, proj_y), cam_list in sorted(p2c_dict.items()):
            for cam_x, cam_y in cam_list:
                f.write(f"{proj_x}, {proj_y}, {cam_x}, {cam_y}\n")

    # NumPy形式でも保存（辞書をそのまま保持）
    p2c_npy = resolve_output_path("result_p2c.npy")
    np.save(p2c_npy, np.array(dict(p2c_dict), dtype=object))

    total_correspondences = sum(len(v) for v in p2c_dict.values())
    print("=== P2C Result ===")
    print(f"Unique projector pixels : {len(p2c_dict)}")
    print(f"Total p2c correspondences : {total_correspondences}")
    print(f"NumPy dict : '{p2c_npy}'")
    print(f"output : '{p2c_csv}'")
    print()

    return (cam_height, cam_width)


if __name__ == "__main__":
    main()
