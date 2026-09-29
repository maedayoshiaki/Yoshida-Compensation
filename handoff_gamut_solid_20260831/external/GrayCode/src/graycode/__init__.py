"""graycode — projector–camera gray-code geometric calibration and image warping.

Public reuse surface for downstream projects (e.g. radiometric compensation
loops that drive their own camera). Camera capture is injectable via the
``capture_fn`` argument of :func:`run_capture` / :func:`run_calibration`; the
Canon EDSDK default in :func:`capture` is imported lazily so this package
imports without the Canon SDK.

``PixelMapWarperTorch`` lives in :mod:`graycode.warp_image`, whose only heavy
dependency is PyTorch (``import torch`` at module load). To keep the gray-code
generate/decode/calibration path importable **without** torch (it needs none),
that symbol is exposed lazily via a module-level ``__getattr__`` (PEP 562):
accessing ``graycode.PixelMapWarperTorch`` (or ``from graycode import
PixelMapWarperTorch``) imports the torch-backed module on demand, so ``import
graycode`` alone stays torch-free.
"""

from typing import TYPE_CHECKING, Any

from . import coords
from .cap_graycode import capture, run_calibration, run_capture
from .decode import decode_c2p
from .gen_graycode import generate_expanded_patterns, save_patterns
from .interpolate_c2p import load_c2p_numpy_array
from .interpolate_p2c import load_p2c_numpy_array

if TYPE_CHECKING:
    from .warp_image import PixelMapWarperTorch

__all__ = [
    "PixelMapWarperTorch",
    "capture",
    "coords",
    "decode_c2p",
    "generate_expanded_patterns",
    "load_c2p_numpy_array",
    "load_p2c_numpy_array",
    "run_calibration",
    "run_capture",
    "save_patterns",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the torch-backed warp symbol (PEP 562).

    Keeps ``import graycode`` free of PyTorch: only an explicit access of
    ``PixelMapWarperTorch`` triggers ``import graycode.warp_image`` (and thus
    ``import torch``).
    """
    if name == "PixelMapWarperTorch":
        from .warp_image import PixelMapWarperTorch

        return PixelMapWarperTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
