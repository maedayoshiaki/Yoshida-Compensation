import json
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

from edsdk.camera_controller import CameraController
from src.python.camera.canon_edsdk import capture_srgb_uint8
from src.python.config import get_config, reload_config, split_cli_config_path


@dataclass(frozen=True)
class SolvePnPConfig:
    calib_dir: Path = Path("data/chess_captured")
    calibration_file_name: str = "camera_calib.npz"
    output_dir: Path = Path("data/pose_output")
    output_json_name: str = "pose_unity.json"
    use_ransac: bool = False
    ransac_reproj_err_px: float = 5.0
    ransac_confidence: float = 0.999
    ransac_iters: int = 200
    solvepnp_flag: int = cv2.SOLVEPNP_ITERATIVE
    click_window_name: str = "Click points (Fullscreen)"
    result_window_name: str = "Pose Result"

    @property
    def calibration_path(self) -> Path:
        return self.calib_dir / self.calibration_file_name

    @property
    def output_json_path(self) -> Path:
        return self.output_dir / self.output_json_name


CONFIG = SolvePnPConfig()
POINT_LABELS = tuple("ABCDEFGHIJKLMN")
OBJECT_POINTS_MM = np.array(
    [
        [0.0, -0.0, -0.0],
        [-24.0372, 60.2031, 17.1096],
        [-24.3812, 118.8160, 23.5571],
        [-34.8000, 228.7060, 29.1758],
        [95.4303, 46.8298, 40.6512],
        [81.4433, 289.8030, 61.7818],
        [-504.1710, 79.8121, 34.5012],
        [-724.9400, 107.7810, 35.6400],
        [-675.5070, 246.6890, 31.8095],
        [-766.1540, 251.8850, 7.5298],
        [-874.5570, 222.6600, 1.79518],
        [-916.1390, 254.2860, 5.71086],
        [-910.4120, 88.6131, -2.51968],
        [-816.9360, 26.4739, 8.34665],
    ],
    dtype=np.float32,
)


def load_intrinsics(config: SolvePnPConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load the camera matrix and distortion coefficients from calibration output."""
    if not config.calibration_path.exists():
        raise FileNotFoundError(
            "Calibration file not found. Run cam_in_calib.py first.\n"
            f"Expected:\n  {config.calibration_path.resolve()}"
        )

    data = np.load(str(config.calibration_path))
    camera_matrix = data["K"].astype(np.float64)
    distortion = data["dist"].astype(np.float64).reshape(-1, 1)
    return camera_matrix, distortion


def load_camera_settings(argv: list[str]) -> tuple[str, str, str, str]:
    """Load camera capture settings from config.toml."""
    clean_argv, config_path = split_cli_config_path(argv)
    if len(clean_argv) != 1:
        print("Usage: python src/python/solvePnP.py [--config <config.toml>]")
        raise SystemExit(1)

    default_config_path = Path(__file__).resolve().parents[2] / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    camera_config = get_config().camera

    if camera_config.backend.strip().lower() != "canon_edsdk":
        raise RuntimeError(
            "solvePnP.py requires camera.backend = \"canon_edsdk\" in config.toml."
        )

    return (
        camera_config.av,
        camera_config.tv,
        camera_config.iso,
        camera_config.image_quality,
    )


def capture_one_image_from_canon(
    av: str,
    tv: str,
    iso: str,
    image_quality: str,
) -> np.ndarray:
    """Capture one still image from the Canon camera and convert it to BGR."""
    with CameraController(register_property_events=False) as camera:
        camera.set_properties(
            av=av,
            tv=tv,
            iso=iso,
            image_quality=image_quality,
        )
        captured_rgb = capture_srgb_uint8(camera, image_quality)

    if captured_rgb.ndim == 2:
        return cv2.cvtColor(captured_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(captured_rgb, cv2.COLOR_RGB2BGR)


def collect_click_points_fullscreen(
    image_bgr: np.ndarray,
    labels: tuple[str, ...],
    window_name: str,
) -> np.ndarray:
    """Collect one clicked image point per label using a fullscreen OpenCV window."""
    base_image = image_bgr.copy()
    display_image = image_bgr.copy()
    clicked_points: list[tuple[int, int]] = []

    def redraw_points() -> None:
        nonlocal display_image
        display_image = base_image.copy()
        for index, (u, v) in enumerate(clicked_points):
            cv2.circle(display_image, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(
                display_image,
                labels[index],
                (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def draw_overlay(image: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        next_label = labels[len(clicked_points)] if len(clicked_points) < len(labels) else "(done)"
        cv2.putText(
            overlay,
            "Left click: add | Backspace: undo | Enter: finish | Esc: quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"Clicked: {len(clicked_points)}/{len(labels)}   Next: {next_label}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < len(labels):
            clicked_points.append((int(x), int(y)))
            redraw_points()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, draw_overlay(display_image))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_name)
            raise SystemExit("Cancelled by user.")

        if key == 8 and clicked_points:
            clicked_points.pop()
            redraw_points()

        if key == 13:
            if len(clicked_points) == len(labels):
                break
            print(f"Need {len(labels)} points, but got {len(clicked_points)}")

    cv2.destroyWindow(window_name)
    return np.array(clicked_points, dtype=np.float32)


def solve_pose(
    object_points_mm: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    config: SolvePnPConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Solve camera pose using either solvePnP or solvePnPRansac."""
    if config.use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_mm,
            image_points,
            camera_matrix,
            distortion,
            flags=config.solvepnp_flag,
            reprojectionError=config.ransac_reproj_err_px,
            confidence=config.ransac_confidence,
            iterationsCount=config.ransac_iters,
        )
        if (not success) or (inliers is None) or (len(inliers) < 4):
            raise RuntimeError("solvePnPRansac failed or not enough inliers.")
        return rvec, tvec, inliers.reshape(-1).astype(int).tolist()

    success, rvec, tvec = cv2.solvePnP(
        object_points_mm,
        image_points,
        camera_matrix,
        distortion,
        flags=config.solvepnp_flag,
    )
    if not success:
        raise RuntimeError("solvePnP failed.")
    return rvec, tvec, list(range(len(object_points_mm)))


def rotation_matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized quaternion [x, y, z, w]."""
    m00, m01, m02 = rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2]
    m10, m11, m12 = rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2]
    m20, m21, m22 = rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif (m00 > m11) and (m00 > m22):
        scale = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale

    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def convert_pose_to_unity(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV world-to-camera pose into Unity camera position and rotation."""
    rotation_world_to_camera, _ = cv2.Rodrigues(rvec)
    translation_world_to_camera = tvec.astype(np.float64).reshape(3, 1)

    rotation_camera_to_world = rotation_world_to_camera.T
    camera_position_mm = -rotation_camera_to_world @ translation_world_to_camera

    camera_position_unity = np.array(
        [
            camera_position_mm[0, 0] / 1000.0,
            -camera_position_mm[1, 0] / 1000.0,
            camera_position_mm[2, 0] / 1000.0,
        ],
        dtype=np.float64,
    )

    axis_flip = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    rotation_unity = axis_flip @ rotation_camera_to_world @ axis_flip
    quaternion_unity = rotation_matrix_to_quaternion_xyzw(rotation_unity)

    return camera_position_unity, quaternion_unity


def compute_reprojection_errors(
    object_points_mm: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    inlier_indices: list[int],
) -> tuple[np.ndarray, float, float, float]:
    projected_points, _ = cv2.projectPoints(object_points_mm, rvec, tvec, camera_matrix, distortion)
    projected_points = projected_points.reshape(-1, 2)
    point_errors = np.linalg.norm(projected_points - image_points, axis=1)

    mean_error_all = float(np.mean(point_errors))
    max_error_all = float(np.max(point_errors))

    if len(inlier_indices) == len(object_points_mm):
        mean_error_inliers = mean_error_all
    else:
        inlier_mask = np.zeros((len(object_points_mm),), dtype=bool)
        inlier_mask[np.array(inlier_indices, dtype=int)] = True
        mean_error_inliers = float(np.mean(point_errors[inlier_mask]))

    return projected_points, mean_error_all, mean_error_inliers, max_error_all


def save_pose_json(
    config: SolvePnPConfig,
    image_shape: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera_position_unity: np.ndarray,
    quaternion_unity: np.ndarray,
    inlier_indices: list[int],
    mean_error_all: float,
    mean_error_inliers: float,
    max_error_all: float,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_shape

    payload = {
        "image_width": width,
        "image_height": height,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "dist": distortion.reshape(-1).astype(float).tolist(),
        "camera_position_m": camera_position_unity.astype(float).tolist(),
        "camera_rotation_xyzw": quaternion_unity.astype(float).tolist(),
        "use_ransac": bool(config.use_ransac),
        "inlier_indices": inlier_indices,
        "reprojection_error_mean_all_px": mean_error_all,
        "reprojection_error_mean_inliers_px": mean_error_inliers,
        "reprojection_error_max_all_px": max_error_all,
    }

    with config.output_json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Saved Unity-friendly JSON: {config.output_json_path.resolve()}")


def show_pose_result(
    image_bgr: np.ndarray,
    labels: tuple[str, ...],
    image_points: np.ndarray,
    projected_points: np.ndarray,
    window_name: str,
) -> None:
    """Render clicked points and reprojected points for a visual sanity check."""
    visualized = image_bgr.copy()

    for label, clicked_point, projected_point in zip(labels, image_points, projected_points):
        u, v = int(round(clicked_point[0])), int(round(clicked_point[1]))
        pu, pv = int(round(projected_point[0])), int(round(projected_point[1]))
        cv2.circle(visualized, (u, v), 6, (0, 0, 255), -1)
        cv2.circle(visualized, (pu, pv), 6, (255, 0, 0), 2)
        cv2.putText(
            visualized,
            label,
            (u + 6, v - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        visualized,
        "Clicked=red | Projected=blue ring (press any key to close)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, visualized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    config = CONFIG
    if len(POINT_LABELS) != len(OBJECT_POINTS_MM):
        raise ValueError("POINT_LABELS and OBJECT_POINTS_MM must have the same length.")

    av, tv, iso, image_quality = load_camera_settings(argv)
    camera_matrix, distortion = load_intrinsics(config)
    image_bgr = capture_one_image_from_canon(av, tv, iso, image_quality)

    image_height, image_width = image_bgr.shape[:2]
    print(f"Captured resolution: {image_width} x {image_height}")

    image_points = collect_click_points_fullscreen(
        image_bgr,
        POINT_LABELS,
        config.click_window_name,
    )
    rvec, tvec, inlier_indices = solve_pose(
        OBJECT_POINTS_MM,
        image_points,
        camera_matrix,
        distortion,
        config,
    )

    camera_position_unity, quaternion_unity = convert_pose_to_unity(rvec, tvec)
    print("Unity position (m):", camera_position_unity)
    print("Unity quaternion [x,y,z,w]:", quaternion_unity)

    projected_points, mean_error_all, mean_error_inliers, max_error_all = compute_reprojection_errors(
        OBJECT_POINTS_MM,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        distortion,
        inlier_indices,
    )
    print("Reprojection error mean (all) [px]:", mean_error_all)
    print("Reprojection error mean (inliers) [px]:", mean_error_inliers)

    save_pose_json(
        config,
        (image_height, image_width),
        camera_matrix,
        distortion,
        camera_position_unity,
        quaternion_unity,
        inlier_indices,
        mean_error_all,
        mean_error_inliers,
        max_error_all,
    )
    show_pose_result(
        image_bgr,
        POINT_LABELS,
        image_points,
        projected_points,
        config.result_window_name,
    )


if __name__ == "__main__":
    main()
