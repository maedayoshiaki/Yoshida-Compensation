"""result_c2p.npy を読み取り、result_p2c.npy / result_p2c.csv を生成する."""

from collections import defaultdict
from typing import List, Tuple

import numpy as np

from .config import resolve_output_path


def gen_p2c_from_c2p(c2p_path: str = "result_c2p.npy") -> None:
    """C2Pマップを読み込み、逆引きのP2Cマップを生成・保存する."""
    c2p_array = np.load(c2p_path, allow_pickle=True)

    # P2C 辞書を構築（同一プロジェクタ座標に複数カメラ座標が対応しうる）
    p2c_dict: dict[Tuple[float, float], List[Tuple[int, int]]] = defaultdict(list)
    for (cam_x, cam_y), (proj_x, proj_y) in c2p_array:
        p2c_dict[(proj_x, proj_y)].append((int(cam_x), int(cam_y)))

    # CSV 保存
    csv_path = resolve_output_path("result_p2c.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("proj_x, proj_y, cam_x, cam_y\n")
        for (proj_x, proj_y), cam_list in sorted(p2c_dict.items()):
            for cam_x, cam_y in cam_list:
                f.write(f"{proj_x}, {proj_y}, {cam_x}, {cam_y}\n")

    # NumPy 形式で保存
    npy_path = resolve_output_path("result_p2c.npy")
    np.save(npy_path, np.array(dict(p2c_dict), dtype=object))

    total_correspondences = sum(len(v) for v in p2c_dict.values())
    print("=== P2C Result ===")
    print(f"Input C2P correspondences : {len(c2p_array)}")
    print(f"Unique projector pixels   : {len(p2c_dict)}")
    print(f"Total P2C correspondences : {total_correspondences}")
    print(f"Output : '{npy_path}', '{csv_path}'")


if __name__ == "__main__":
    gen_p2c_from_c2p()
