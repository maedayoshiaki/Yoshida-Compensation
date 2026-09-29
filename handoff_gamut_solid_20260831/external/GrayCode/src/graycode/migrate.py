# coding: utf-8
"""座標規約の移行: 旧 UV 規約 (proj 中心 = i+0.5) → pixel-is-point (proj 中心 = 整数)。

旧バージョンが出力した対応点ファイルの **プロジェクタ座標から一律 0.5 を減算**する
(カメラ座標は不変)。これは ``block_center`` を UV 中心 ``step*(g+0.5)`` から
pixel-is-point 中心 ``step*g+(step-1)/2`` に変えたことに対応し、差は step によらず
常に 0.5 (COORDINATES.md / coords.block_center 参照)。

対象:
  - result_c2p.npy          : dtype=object (N,2,2) [[cam],[proj]]
  - result_p2c.npy          : 0-d object dict {(px,py): [(cx,cy), ...]}
  - result_*_compensated_delaunay.npy (p2c): (N,4) float [proj,proj,cam,cam]
  - result_c2p_compensated_*.npy           : dtype=object (N,2,2)
  - .npz                    : {p2c: (N,4)[proj,proj,cam,cam], proj_size, ...}
                              (2dsr-prc ProjectorCameraMap.save 形式)
  - 各 .csv                 : ヘッダの proj_x / proj_y 列を検出して減算

非破壊。入力 ``foo.npy`` に対し ``foo_pixelpoint.npy`` を出力する
(再実行で二重適用しないよう、元ファイルは書き換えない)。

CLI:
    uv run python -m graycode.migrate result_c2p.npy result_p2c.npy p2c.npz result_p2c.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJ_DELTA = -0.5  # UV(area center) -> pixel-is-point(center)
_PROJ_COL_NAMES = ("proj_x", "proj_y")


# ── 純粋関数 (テスト対象) ────────────────────────────────────────────


def shift_proj_dict(d: dict, delta: float = PROJ_DELTA) -> dict:
    """p2c dict {(px,py): [(cx,cy),...]} のキー(proj)を delta シフトする。"""
    return {
        (float(px) + delta, float(py) + delta): [tuple(c) for c in v]
        for (px, py), v in d.items()
    }


def shift_c2p_object_array(arr: np.ndarray, delta: float = PROJ_DELTA) -> np.ndarray:
    """c2p object 配列 ((N,2,2)/レガシー) の proj を delta シフトし (N,2,2) object で返す。"""
    flat = _c2p_object_to_n4(arr)
    flat[:, 2:4] += delta  # proj_x, proj_y
    return _n4_to_c2p_object(flat)


def shift_p2c_compensated_array(
    arr: np.ndarray, delta: float = PROJ_DELTA
) -> np.ndarray:
    """p2c_compensated (N,4)[proj,proj,cam,cam] の proj 列を delta シフトする。"""
    out = arr.astype(np.float32, copy=True)
    out[:, 0:2] += delta
    return out


# ── 内部ヘルパー ─────────────────────────────────────────────────────


def _c2p_object_to_n4(arr: np.ndarray) -> np.ndarray:
    """object な c2p ((N,2,2) 等) を (N,4)[cam,cam,proj,proj] float64 に正規化。"""
    n = len(arr)
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        item = arr[i]
        cam, proj = item[0], item[1]
        out[i, 0] = float(cam[0])
        out[i, 1] = float(cam[1])
        out[i, 2] = float(proj[0])
        out[i, 3] = float(proj[1])
    return out


def _n4_to_c2p_object(flat: np.ndarray) -> np.ndarray:
    """(N,4)[cam,cam,proj,proj] を decode/compensated 互換の (N,2,2) object に。"""
    n = len(flat)
    legacy = np.empty((n, 2, 2), dtype=object)
    legacy[:, 0, 0] = flat[:, 0]
    legacy[:, 0, 1] = flat[:, 1]
    legacy[:, 1, 0] = flat[:, 2]
    legacy[:, 1, 1] = flat[:, 3]
    return legacy


# ── ファイル単位の移行 ───────────────────────────────────────────────


def migrate_npy(in_path: str, out_path: str, delta: float = PROJ_DELTA) -> str:
    """.npy を自動判別して移行し、形式の説明文字列を返す。"""
    data = np.load(in_path, allow_pickle=True)

    # (N,4) 数値 = p2c_compensated [proj,proj,cam,cam]
    if isinstance(data, np.ndarray) and data.dtype != object:
        if data.ndim == 2 and data.shape[1] == 4:
            np.save(out_path, shift_p2c_compensated_array(data, delta))
            return "p2c_compensated (N,4): proj cols 0,1 shifted"
        raise ValueError(f"Unsupported numeric npy shape {data.shape}")

    # object: 0-d dict = p2c, それ以外 = c2p object
    if isinstance(data, np.ndarray) and data.dtype == object:
        if data.ndim == 0:
            d = data.item()
            if not isinstance(d, dict):
                raise ValueError("0-d object npy is not a dict")
            np.save(out_path, np.array(shift_proj_dict(d, delta), dtype=object))
            return "p2c dict: keys (proj) shifted"
        flat = _c2p_object_to_n4(data)
        flat[:, 2:4] += delta
        np.save(out_path, _n4_to_c2p_object(flat))
        return "c2p object (N,2,2): proj shifted"

    raise ValueError("Unsupported npy format")


def migrate_csv(in_path: str, out_path: str, delta: float = PROJ_DELTA) -> str:
    """ヘッダの proj_x / proj_y 列を検出して delta シフトする。"""
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError("empty csv")

    header = [h.strip() for h in lines[0].split(",")]
    proj_idx = [i for i, h in enumerate(header) if h in _PROJ_COL_NAMES]
    if not proj_idx:
        raise ValueError(f"no proj_x/proj_y columns in header: {header}")

    out_lines = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split(",")]
        for i in proj_idx:
            cells[i] = f"{float(cells[i]) + delta:g}"
        out_lines.append(", ".join(cells))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return f"csv: columns {proj_idx} ({[header[i] for i in proj_idx]}) shifted"


def migrate_npz(in_path: str, out_path: str, delta: float = PROJ_DELTA) -> str:
    """``p2c`` 配列を持つ .npz を移行する (2dsr-prc ProjectorCameraMap.save 形式)。

    ``p2c`` は (N,4)[proj_x, proj_y, cam_x, cam_y]。proj 列のみ delta シフトし、
    ``proj_size`` など他のキーはそのまま保持して再保存する。
    """
    with np.load(in_path, allow_pickle=True) as data:
        if "p2c" not in data.files:
            raise ValueError(f".npz has no 'p2c' array (keys: {data.files})")
        p2c = np.asarray(data["p2c"], dtype=np.float32).reshape(-1, 4).copy()
        others = {k: np.asarray(data[k]) for k in data.files if k != "p2c"}

    p2c[:, 0:2] += delta  # proj_x, proj_y
    np.savez(out_path, p2c=p2c, **others)
    kept = ", ".join(others) or "(none)"
    return f"npz: p2c proj cols 0,1 shifted; preserved keys: {kept}"


def migrate_file(in_path: str, delta: float = PROJ_DELTA) -> str:
    """拡張子で .npy/.csv/.npz を振り分け、<stem>_pixelpoint<suffix> に出力する。"""
    p = Path(in_path)
    out = p.with_name(f"{p.stem}_pixelpoint{p.suffix}")
    if p.suffix == ".npy":
        desc = migrate_npy(str(p), str(out), delta)
    elif p.suffix == ".csv":
        desc = migrate_csv(str(p), str(out), delta)
    elif p.suffix == ".npz":
        desc = migrate_npz(str(p), str(out), delta)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    print(f"  {p.name} -> {out.name}  [{desc}]")
    return str(out)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    files = argv[1:]
    if not files:
        print(
            "Usage: python -m graycode.migrate <file.npy|file.csv|file.npz> [...]\n"
            "  旧 UV 規約のファイルを pixel-is-point 規約へ移行 (proj 座標 -0.5)。\n"
            "  各入力に対し <stem>_pixelpoint<suffix> を出力 (非破壊)。"
        )
        return
    print("Migrating projector coordinates (UV -> pixel-is-point, proj -= 0.5):")
    for f in files:
        try:
            migrate_file(f)
        except Exception as e:  # noqa: BLE001
            print(f"  {f}: ERROR {e}")


if __name__ == "__main__":
    main()
