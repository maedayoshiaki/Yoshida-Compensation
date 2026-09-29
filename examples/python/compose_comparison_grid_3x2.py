"""Convert six-panel horizontal comparison images into a 3x2 grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in sorted(args.input_dir.glob("comparison_*.png")):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] % 6:
            raise ValueError(f"Expected six equal-width panels: {source}")
        panel_width = image.shape[1] // 6
        panels = [
            image[:, index * panel_width : (index + 1) * panel_width]
            for index in range(6)
        ]
        grid = np.vstack(
            [
                np.hstack(panels[:3]),
                np.hstack(panels[3:]),
            ]
        )
        destination = args.output_dir / source.name.replace(
            ".png", "_3x2.png"
        )
        if not cv2.imwrite(str(destination), grid):
            raise OSError(f"Could not write {destination}")
        print(destination)


if __name__ == "__main__":
    main()
