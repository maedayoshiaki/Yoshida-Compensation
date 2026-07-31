"""Run validate_inverse_lut with the other displays blanked (same covers
as the calibration passes, so ambient conditions match)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples" / "python"))

SCRATCH = Path(__file__).resolve().parent

try:
    import validate_inverse_lut

    extra = sys.argv[1:]
    validate_inverse_lut.main(
        [
            "validate_inverse_lut.py",
            "--config", str(SCRATCH / "config_display5.toml"),
            "--num-levels", "33",
        ]
        + (extra if extra else
           ["--out-prefix", str(SCRATCH / "validation" / "inverse_lut_validation")])
    )
finally:
    cv2.destroyAllWindows()
