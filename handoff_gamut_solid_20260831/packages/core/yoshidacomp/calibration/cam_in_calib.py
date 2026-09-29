from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationConfig:
    image_dir: Path = Path("data/chess_captured")
    checkerboard: tuple[int, int] = (10, 7)
    square_size_mm: float = 19.0
    use_sb_detector: bool = True
    min_required_images: int = 8
    image_pattern: str = "chess_*.png"


@dataclass(frozen=True)
class CalibrationResult:
    rms: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    mean_reprojection_error_px: float
    used_images: int
    total_images: int


CONFIG = CalibrationConfig()


def list_input_images(config: CalibrationConfig) -> list[Path]:
    return sorted(config.image_dir.glob(config.image_pattern))


def build_object_points(config: CalibrationConfig) -> np.ndarray:
    cols, rows = config.checkerboard
    object_points = np.zeros((cols * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= config.square_size_mm
    return object_points


def detect_corners(image: np.ndarray, config: CalibrationConfig) -> np.ndarray | None:
    """Detect checkerboard corners and return them as float32 points."""
    if config.use_sb_detector and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(image, config.checkerboard)
        if found:
            return corners.astype(np.float32)
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, config.checkerboard, flags)
    if not found:
        return None
    return cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)


def collect_calibration_points(
    image_paths: list[Path],
    config: CalibrationConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Load images and collect matching 3D/2D points for calibration."""
    object_points_template = build_object_points(config)
    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] could not read: {image_path}")
            continue

        if image_size is None:
            image_size = (image.shape[1], image.shape[0])

        corners = detect_corners(image, config)
        if corners is None:
            print(f"[skip] corners not found: {image_path.name}")
            continue

        object_points_list.append(object_points_template.copy())
        image_points_list.append(corners)

    if image_size is None:
        raise RuntimeError("No readable images were found for calibration.")

    return object_points_list, image_points_list, image_size


def compute_mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_error_sq = 0.0
    total_points = 0

    for object_set, image_set, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected_points, _ = cv2.projectPoints(object_set, rvec, tvec, camera_matrix, distortion)
        error = cv2.norm(image_set, projected_points, cv2.NORM_L2)
        total_error_sq += error * error
        total_points += len(object_set)

    return float(np.sqrt(total_error_sq / total_points))


def calibrate_from_dir(config: CalibrationConfig) -> CalibrationResult:
    image_paths = list_input_images(config)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {config.image_dir}")

    object_points, image_points, image_size = collect_calibration_points(image_paths, config)
    used_images = len(object_points)
    total_images = len(image_paths)

    print(f"Detected corners in {used_images}/{total_images} images.")
    if used_images < config.min_required_images:
        raise RuntimeError(
            "Not enough valid calibration images. Capture more views with varied angles and positions."
        )

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    mean_error = compute_mean_reprojection_error(
        object_points,
        image_points,
        tuple(rvecs),
        tuple(tvecs),
        camera_matrix,
        distortion,
    )

    return CalibrationResult(
        rms=float(rms),
        camera_matrix=camera_matrix,
        distortion=distortion,
        mean_reprojection_error_px=mean_error,
        used_images=used_images,
        total_images=total_images,
    )


def save_results(config: CalibrationConfig, result: CalibrationResult) -> None:
    config.image_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.image_dir / "camera_calib.npz"
    np.savez(
        str(output_path),
        K=result.camera_matrix,
        dist=result.distortion,
        checkerboard=np.array(config.checkerboard, dtype=np.int32),
        square_size=float(config.square_size_mm),
    )
    print(f"Saved: {output_path.name}")


def main() -> None:
    config = CONFIG
    print(f"Input dir: {config.image_dir.resolve()}")

    result = calibrate_from_dir(config)

    print("\n=== Result ===")
    print(f"Used images             : {result.used_images}/{result.total_images}")
    print(f"RMS (OpenCV)            : {result.rms:.6f}")
    print(f"Mean reproj error (px)  : {result.mean_reprojection_error_px:.6f}")
    print("\nCamera Matrix K:\n", result.camera_matrix)
    print("\nDistortion coeffs:\n", result.distortion.ravel())

    save_results(config, result)
    print("\nDone.")


if __name__ == "__main__":
    main()
