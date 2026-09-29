# coding: utf-8

import os
import sys

import cv2
import numpy as np

# 追加: Scipyの補完モジュール
try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from . import coords
from .config import (
    get_config,
    reload_config,
    resolve_output_path,
    split_cli_config_path,
)

_INPAINT_METHOD_MAP = {
    "TELEA": cv2.INPAINT_TELEA,
    "NS": cv2.INPAINT_NS,
}


def load_c2p_numpy(
    map_file_path: str,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """互換API: c2pのnpyを読み込んで従来形式のリストを返す。

    戻り値: List[((cam_x, cam_y), (proj_x, proj_y))]
    """
    arr = load_c2p_numpy_array(map_file_path)
    return [
        ((float(cam_x), float(cam_y)), (float(proj_x), float(proj_y)))
        for cam_x, cam_y, proj_x, proj_y in arr
    ]


def load_c2p_numpy_array(map_file_path: str) -> np.ndarray:
    """内部用: c2pのnpyを読み込んで (N,4) float32 配列で返す。

    互換入力:
      - decode.py 互換: dtype=object の [[x,y],[u,v]] / ndarray要素
      - 数値配列: (N,4) [cam_x,cam_y,proj_x,proj_y]
      - 参考: (H,W,2) [proj_x,proj_y]
    """
    map_data = np.load(map_file_path, allow_pickle=True)

    # 既に (N,4) の数値配列
    if isinstance(map_data, np.ndarray) and map_data.dtype != object:
        if map_data.ndim == 2 and map_data.shape[1] == 4:
            return map_data.astype(np.float32, copy=False)

        # (H,W,2) の密マップ
        if map_data.ndim == 3 and map_data.shape[2] == 2:
            h, w, _ = map_data.shape
            out = np.empty((h * w, 4), dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            for y in range(h):
                row = slice(y * w, (y + 1) * w)
                out[row, 0] = x_coords
                out[row, 1] = np.float32(y)
                out[row, 2] = map_data[y, :, 0].astype(np.float32, copy=False)
                out[row, 3] = map_data[y, :, 1].astype(np.float32, copy=False)
            return out

    # decode.py 互換: dtype=object の配列
    if not (
        isinstance(map_data, np.ndarray)
        and map_data.dtype == object
        and map_data.ndim >= 1
    ):
        raise TypeError("Unsupported c2p numpy format")

    def _as_seq(x):
        # object配列の要素が np.ndarray で入ってくる場合がある
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    n = int(map_data.shape[0])
    out = np.empty((n, 4), dtype=np.float32)

    # 先頭だけチェック
    for i in range(min(10, n)):
        item = _as_seq(map_data[i])
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(_as_seq(item[0]), (list, tuple))
            or len(item[0]) != 2
            or not isinstance(_as_seq(item[1]), (list, tuple))
            or len(item[1]) != 2
        ):
            raise TypeError(f"map_list[{i}] must be [[x,y],[u,v]]")

    for i in range(n):
        cam_xy, proj_uv = _as_seq(map_data[i])
        cam_xy = _as_seq(cam_xy)
        proj_uv = _as_seq(proj_uv)
        out[i, 0] = float(cam_xy[0])
        out[i, 1] = float(cam_xy[1])
        out[i, 2] = float(proj_uv[0])
        out[i, 3] = float(proj_uv[1])
    return out


def interpolate_c2p_array(
    cam_height: int,
    cam_width: int,
    c2p_list: np.ndarray,
) -> np.ndarray:
    """
    [Original Method]
    Fill missing correspondences using cv2.inpaint (float32).
    """
    # Initialize maps with NaN
    proj_x_map = np.full((cam_height, cam_width), np.nan, dtype=np.float32)
    proj_y_map = np.full((cam_height, cam_width), np.nan, dtype=np.float32)

    # Fill known correspondences (vectorized)
    if not (
        isinstance(c2p_list, np.ndarray)
        and c2p_list.ndim == 2
        and c2p_list.shape[1] == 4
    ):
        raise TypeError("c2p_list must be a NumPy array with shape (N,4)")

    cam_x = c2p_list[:, 0]
    cam_y = c2p_list[:, 1]
    proj_x = c2p_list[:, 2]
    proj_y = c2p_list[:, 3]

    # 画素インデックス = round(x) = floor(x+0.5) (pixel-is-point)。
    ix = coords.to_pixel(cam_x).astype(np.int32)
    iy = coords.to_pixel(cam_y).astype(np.int32)
    valid = (
        (0 <= ix)
        & (ix < cam_width)
        & (0 <= iy)
        & (iy < cam_height)
        & ~np.isnan(proj_x)
        & ~np.isnan(proj_y)
    )

    ix_v = ix[valid]
    iy_v = iy[valid]
    proj_x_map[iy_v, ix_v] = proj_x[valid].astype(np.float32, copy=False)
    proj_y_map[iy_v, ix_v] = proj_y[valid].astype(np.float32, copy=False)

    # Solve for each channel
    c2p_cfg = get_config().interpolate_c2p
    cv_method = _INPAINT_METHOD_MAP.get(c2p_cfg.inpaint_method, cv2.INPAINT_TELEA)
    proj_x_filled = _inpaint_fill_float32(
        proj_x_map, radius=c2p_cfg.inpaint_radius, method=cv_method
    )
    proj_y_filled = _inpaint_fill_float32(
        proj_y_map, radius=c2p_cfg.inpaint_radius, method=cv_method
    )

    # Build output as (N,4) float32
    out = np.empty((cam_height * cam_width, 4), dtype=np.float32)
    x_coords = np.arange(cam_width, dtype=np.float32)
    for y in range(cam_height):
        row = slice(y * cam_width, (y + 1) * cam_width)
        out[row, 0] = x_coords
        out[row, 1] = np.float32(y)
        out[row, 2] = proj_x_filled[y, :].astype(np.float32, copy=False)
        out[row, 3] = proj_y_filled[y, :].astype(np.float32, copy=False)

    return out


def interpolate_c2p_delaunay(
    cam_height: int,
    cam_width: int,
    c2p_list: np.ndarray,
    return_mask: bool = False,
    extrapolate: bool = True,
):
    """
    [New Method]
    Fill missing correspondences using Scipy's Delaunay Triangulation (LinearNDInterpolator).
    Fills convex hull exterior with NearestNDInterpolator.

    return_mask=True のとき (out, extrapolated) を返す。extrapolated は (H*W,) bool で、
    凸包の外側 (NearestND で外挿された = 信頼度の低い) 画素が True (M2)。
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for delaunay interpolation.")

    print("Running Delaunay Interpolation...")

    if not (
        isinstance(c2p_list, np.ndarray)
        and c2p_list.ndim == 2
        and c2p_list.shape[1] == 4
    ):
        raise TypeError("c2p_list must be a NumPy array with shape (N,4)")

    # 1. データのクレンジング (NaN除去)
    # c2p_list: [cam_x, cam_y, proj_x, proj_y]
    valid_mask = ~np.isnan(c2p_list).any(axis=1)
    valid_data = c2p_list[valid_mask]

    if len(valid_data) < 4:
        raise ValueError("Not enough points for Delaunay triangulation.")

    # 入力点 (Camera Coordinates) と 値 (Projector Coordinates)
    points = valid_data[:, 0:2]  # cam_x, cam_y
    values = valid_data[:, 2:4]  # proj_x, proj_y

    # 2. 補完グリッドの作成
    # カメラ全画素の中心座標 (XY 規約: 画素中心 = 整数 なので arange そのもの)
    grid_x, grid_y = np.meshgrid(
        coords.pixel_centers(cam_width), coords.pixel_centers(cam_height)
    )
    # (H*W, 2) の形に変形
    query_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    # 3. ドロネー分割による線形補完 (Linear Interpolation)
    # 凸包の内部を補完します。外部はNaNになります。
    print("  - Building Delaunay triangulation and interpolating (Linear)...")
    lin_interp = LinearNDInterpolator(points, values)
    interpolated_values = lin_interp(query_points)  # shape: (N_pixels, 2)

    # 4. NaN部分（凸包の外側）の処理
    # ドロネー補完はデータの外側を推論できないため、既定では NaN を Nearest Neighbor で
    # 埋める。``extrapolate=False`` のときは埋めず NaN のまま残す（投影領域マスク外を
    # 穴埋めしない＝外挿オフ。評価では NaN が自動で除外される）。
    nan_mask = np.isnan(interpolated_values[:, 0])
    if extrapolate and np.any(nan_mask):
        print(f"  - Filling {np.sum(nan_mask)} outside points with Nearest Neighbor...")
        # NearestNDInterpolator は `scipy.spatial.KDTree` を使用します
        near_interp = NearestNDInterpolator(points, values)
        interpolated_values[nan_mask] = near_interp(query_points[nan_mask])
    elif np.any(nan_mask):
        print(f"  - Leaving {np.sum(nan_mask)} outside points as NaN (extrapolate=False)")

    # 5. 結果の整形 (N, 4)
    out = np.empty((cam_height * cam_width, 4), dtype=np.float32)
    out[:, 0] = query_points[:, 0].astype(np.float32)  # grid_x
    out[:, 1] = query_points[:, 1].astype(np.float32)  # grid_y
    out[:, 2] = interpolated_values[:, 0].astype(np.float32)  # proj_x
    out[:, 3] = interpolated_values[:, 1].astype(np.float32)  # proj_y

    if return_mask:
        return out, nan_mask
    return out


def interpolate_c2p_list(
    cam_height: int,
    cam_width: int,
    c2p_list: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """互換API: 従来形式のリストを受け、従来形式のリストを返す。"""

    # まず (N,4) に正規化して計算
    arr = np.empty((len(c2p_list), 4), dtype=np.float32)
    for i, ((cam_x, cam_y), (proj_x, proj_y)) in enumerate(c2p_list):
        arr[i, 0] = float(cam_x)
        arr[i, 1] = float(cam_y)
        arr[i, 2] = float(proj_x)
        arr[i, 3] = float(proj_y)

    out = interpolate_c2p_array(cam_height, cam_width, arr)
    return [
        ((float(cam_x), float(cam_y)), (float(proj_x), float(proj_y)))
        for cam_x, cam_y, proj_x, proj_y in out
    ]


def _inpaint_fill_float32(
    data: np.ndarray,
    radius: float,
    method: int,
) -> np.ndarray:
    """NaN を穴として cv2.inpaint で埋める（1ch float32）。

    OpenCV(4.11.0) では 32-bit float 1ch がサポートされるため、uint8 量子化は行わない。
    """

    if data.ndim != 2:
        raise TypeError("data must be a 2D array")

    src = data.astype(np.float32, copy=True)
    mask = np.isnan(src)
    if not np.any(mask):
        return src

    # inpaintのマスクは 0/255 の uint8
    mask_u8 = mask.astype(np.uint8) * 255

    # NaNは仮に0で埋めておく（maskで穴扱いになる）
    src[mask] = 0.0

    # float32 のまま inpaint
    dst = cv2.inpaint(src, mask_u8, float(radius), int(method))

    # 既知値は厳密に保持
    dst[~mask] = data[~mask].astype(np.float32, copy=False)
    return dst


def create_vis_image(
    cam_height: int,
    cam_width: int,
    c2p_list_interp: np.ndarray,
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    if not (
        isinstance(c2p_list_interp, np.ndarray)
        and c2p_list_interp.ndim == 2
        and c2p_list_interp.shape[1] == 4
    ):
        raise TypeError("c2p_list_interp must be a NumPy array with shape (N,4)")

    arr = c2p_list_interp.astype(np.float32, copy=False)

    cam_x = arr[:, 0]
    cam_y = arr[:, 1]
    proj_x = arr[:, 2]
    proj_y = arr[:, 3]

    # 画素インデックス = round(x) = floor(x+0.5) (pixel-is-point)。
    ix = coords.to_pixel(cam_x).astype(np.int32)
    iy = coords.to_pixel(cam_y).astype(np.int32)

    # 有効な点だけをマスク
    valid = (
        (0 <= ix)
        & (ix < cam_width)
        & (0 <= iy)
        & (iy < cam_height)
        & ~np.isnan(proj_x)
        & ~np.isnan(proj_y)
    )

    ix_v = ix[valid]
    iy_v = iy[valid]
    proj_x_v = proj_x[valid]
    proj_y_v = proj_y[valid]

    vis_image = np.zeros((cam_height, cam_width, 3), dtype=dtype)

    # 色をベクトル化して計算
    r = (proj_x_v).astype(np.int32) % (np.iinfo(dtype).max + 1)
    g = (proj_y_v).astype(np.int32) % (np.iinfo(dtype).max + 1)
    b = 128 * np.ones_like(r, dtype=np.int32)

    vis_image[iy_v, ix_v, 0] = r.astype(dtype)
    vis_image[iy_v, ix_v, 1] = g.astype(dtype)
    vis_image[iy_v, ix_v, 2] = b.astype(dtype)

    return vis_image


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    try:
        argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(e)
        print(
            "Usage : python interpolate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width> [method] [--config <config.toml>]"
        )
        print("   method: 'delaunay' or 'inpaint' (省略時は config の default_method)")
        print()
        return

    if config_path is not None:
        reload_config(config_path)

    if len(argv) < 4:
        print(
            "Usage : python interpolate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width> [method] [--config <config.toml>]"
        )
        print("   method: 'delaunay' or 'inpaint' (省略時は config の default_method)")
        print()
        return

    try:
        cam_height = int(argv[2])
        cam_width = int(argv[3])
        c2p_numpy_filename = str(argv[1])
        c2p_cfg = get_config().interpolate_c2p
        method = c2p_cfg.default_method
        if len(argv) >= 5:
            method = str(argv[4]).lower()
    except ValueError:
        print("cam_height, cam_width は整数で指定してください。")
        print()
        return

    try:
        # 内部計算は省メモリな (N,4) 配列で扱う
        c2p_arr = load_c2p_numpy_array(c2p_numpy_filename)
    except Exception as e:
        print(f"Error loading c2p numpy file: {e}")
        return

    print(
        f"Loaded {len(c2p_arr)} camera-to-projector correspondences from '{c2p_numpy_filename}'"
    )
    print(f"Target size: {cam_width}x{cam_height}, Method: {method}")

    # メソッド分岐
    extrapolated = None
    if method == "delaunay":
        if not SCIPY_AVAILABLE:
            print(
                "Error: 'scipy' module is not installed. Please install it with 'pip install scipy'."
            )
            return
        c2p_list_interp, extrapolated = interpolate_c2p_delaunay(
            cam_height, cam_width, c2p_arr, return_mask=True
        )
    else:
        # Default to inpaint
        if method != "inpaint":
            print(f"Unknown method '{method}', falling back to 'inpaint'.")
        c2p_list_interp = interpolate_c2p_array(cam_height, cam_width, c2p_arr)

    base = os.path.splitext(os.path.basename(c2p_numpy_filename))[0]

    # create image for visualization
    vis_image = create_vis_image(
        cam_height, cam_width, c2p_list_interp, dtype=np.dtype(np.uint8)
    )
    vis_path = resolve_output_path(f"{base}_compensated_{method}_vis.png")
    cv2.imwrite(str(vis_path), vis_image)
    print(f"Saved visualization image to '{vis_path}'")

    # 凸包外 (Nearest 外挿) 画素のマスクを保存 (M2: 補間値と外挿値を区別する)
    if extrapolated is not None:
        mask_path = resolve_output_path(f"{base}_compensated_{method}_extrapolated.png")
        mask_img = extrapolated.reshape(cam_height, cam_width).astype(np.uint8) * 255
        cv2.imwrite(str(mask_path), mask_img)
        frac = 100.0 * float(np.count_nonzero(extrapolated)) / extrapolated.size
        print(f"Saved extrapolation mask to '{mask_path}' ({frac:.1f}% extrapolated)")

    out_path = resolve_output_path(f"{base}_compensated_{method}.npy")
    # 外部互換性のため従来形式 dtype=object の (N,2,2) で保存
    n = cam_height * cam_width
    legacy = np.empty((n, 2, 2), dtype=object)
    legacy[:, 0, 0] = c2p_list_interp[:, 0].astype(np.float64, copy=False)
    legacy[:, 0, 1] = c2p_list_interp[:, 1].astype(np.float64, copy=False)
    legacy[:, 1, 0] = c2p_list_interp[:, 2].astype(np.float64, copy=False)
    legacy[:, 1, 1] = c2p_list_interp[:, 3].astype(np.float64, copy=False)
    np.save(out_path, legacy)
    print(f"Saved compensated correspondences to '{out_path}' (legacy object format)")

    csv_path = resolve_output_path(f"result_c2p_compensated_{method}.csv")
    precision = get_config().interpolate_c2p.csv_precision
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for row in c2p_list_interp:
            f.write(
                f"{row[0]:.{precision}f}, {row[1]:.{precision}f}, "
                f"{row[2]:.{precision}f}, {row[3]:.{precision}f}\n"
            )

    print(f"output : '{csv_path}'")
    print()


if __name__ == "__main__":
    main()
