"""投影コード生成の量子化順序を固定するテスト。

契約: 線形化(逆ガンマ・逆LUT)は float の補償値に適用し、8bit 量子化は
最後に 1 回だけ ``floor(x * 255 + 0.5)`` で行う。

旧実装は補償値をリニア空間で uint8 化してから線形化していた。リニアの
最小刻み 1/255 は逆ガンマ 1/2.2 で表示コード約 20 に写るため、表示コード
1〜19 が到達不能になり、暗部の階調が量子化で潰れる。さらに逆ガンマ関数の
uint8 化は切り捨て(``(x * 255).astype(uint8)``)で、最大 1 コードの
下方向バイアスがあった。このファイルの各テストは旧実装では落ちる
(RED を実測してから修正を書いた)。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest


@dataclass(frozen=True)
class _Projector:
    width: int
    height: int
    gamma: float = 2.2


def _diverse_target_and_matrices(size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """全 256 値を含むターゲットと、非対称・バイアス付きの行列を返す。

    ターゲットの R は 0..255 の全値、G/B はそれと非対称な系列。行列は
    非対角とバイアスを持ち、負クリップも発生する(G のバイアス -0.01)。
    """
    values = np.arange(size * size, dtype=np.uint8).reshape(size, size)
    target = np.stack(
        [
            values,
            (values.astype(np.int32) * 3 % 256).astype(np.uint8),
            255 - values,
        ],
        axis=-1,
    )
    matrices = np.zeros((size, size, 4, 3), dtype=np.float32)
    matrices[:, :, 0, 0] = 0.9
    matrices[:, :, 1, 0] = 0.05
    matrices[:, :, 1, 1] = 0.85
    matrices[:, :, 2, 1] = 0.1
    matrices[:, :, 2, 2] = 0.95
    matrices[:, :, 0, 2] = 0.05
    matrices[:, :, 3, 0] = 0.02
    matrices[:, :, 3, 1] = -0.01
    matrices[:, :, 3, 2] = 0.01
    return target, matrices


# ---------------------------------------------------------------------------
# 1. generate_projection_image: 順序の契約
# ---------------------------------------------------------------------------


def test_gamma_projection_preserves_dark_detail_below_one_linear_code() -> None:
    """リニア 1 コード未満の暗部が、逆ガンマ後の投影コードで区別できること。

    行列 0.4 倍・ターゲット値 1, 2, 3 の補償値は 0.4/255, 0.8/255, 1.2/255。
    リニア空間で先に uint8 化すると 0, 1, 1 になり、投影コードは 0, 20, 20 に
    潰れる(値 1 は消灯、値 2 と 3 は区別不能)。float のまま逆ガンマへ
    通せば約 14, 19, 22 になる。
    """
    from yoshidacomp.compensation.pipeline import generate_projection_image

    matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 0.4

    target = np.zeros((4, 4, 3), dtype=np.uint8)
    target[:, 0] = 1
    target[:, 1] = 2
    target[:, 2] = 3

    _, projected = generate_projection_image(
        target,
        matrices,
        projector=_Projector(width=4, height=4),
        active_region=(0, 0, 4, 4),
        linearizer="gamma",
        target_decode="none",
    )

    codes = projected[0, :3, 0].astype(np.int64)
    # 値 1 が消灯しない
    assert codes[0] > 0
    # 値 1・2・3 が単調に区別される
    assert codes[0] < codes[1] < codes[2]
    # float64 での厳密値から高々 1 コードしかずれない
    reference = [
        round(((0.4 * v / 255.0) ** (1.0 / 2.2)) * 255.0) for v in (1, 2, 3)
    ]
    np.testing.assert_allclose(codes, reference, atol=1)


def test_gamma_projection_quantizes_once_after_float_linearization() -> None:
    """投影コード = floor(pow(float補償値, 1/γ) * 255 + 0.5)。量子化は最後の 1 回だけ。"""
    from yoshidacomp.compensation.photometric_compensation import (
        calc_compensation_image,
    )
    from yoshidacomp.compensation.pipeline import (
        _apply_inverse_gamma_numpy,
        decode_target,
        embed_in_active_region,
        generate_projection_image,
    )
    from yoshidacomp.compensation.projection_region import mask_to_active_region

    target, matrices = _diverse_target_and_matrices()
    region = (0, 0, 16, 16)
    projector = _Projector(width=16, height=16)

    _, projected = generate_projection_image(
        target,
        matrices,
        projector=projector,
        active_region=region,
        linearizer="gamma",
        target_decode="none",
    )

    target_linear = embed_in_active_region(
        decode_target(target, mode="none"), (16, 16), region
    )
    comp_float = calc_compensation_image(
        target_image=target_linear,
        color_mixing_matrices=matrices,
        dtype=np.float32,
        clip=True,
    )
    comp_float = mask_to_active_region(comp_float, region)
    linearized = _apply_inverse_gamma_numpy(comp_float, gamma=1.0 / projector.gamma)
    expected = np.floor(np.clip(linearized, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # 参照値は旧順序(リニア空間で先に uint8 化してから線形化)と実際に
    # 異なること。この保証が無いと、順序を戻す退行を検出できない。
    old_order = _apply_inverse_gamma_numpy(
        mask_to_active_region(
            calc_compensation_image(
                target_image=target_linear,
                color_mixing_matrices=matrices,
                dtype=np.uint8,
                clip=True,
            ),
            region,
        ),
        gamma=1.0 / projector.gamma,
    )
    assert not np.array_equal(expected, old_order)

    assert projected.dtype == np.uint8
    np.testing.assert_array_equal(projected, expected)


def test_inverse_lut_projection_quantizes_once_after_float_linearization() -> None:
    """逆LUT経路も float 補償値に LUT を適用し、量子化は最後の 1 回だけ。"""
    from yoshidacomp.compensation.inverse_lut import InverseLut, apply_inverse_lut
    from yoshidacomp.compensation.photometric_compensation import (
        calc_compensation_image,
    )
    from yoshidacomp.compensation.pipeline import (
        decode_target,
        embed_in_active_region,
        generate_projection_image,
    )
    from yoshidacomp.compensation.projection_region import mask_to_active_region

    target, matrices = _diverse_target_and_matrices()
    region = (0, 0, 16, 16)
    t_grid = np.linspace(0.0, 1.0, 256).astype(np.float32)
    curve = np.power(t_grid, np.float32(1.0 / 2.2)).astype(np.float32)
    lut = InverseLut(
        lut=np.stack([curve, curve, curve]),
        t_grid=t_grid,
        knots=[],
        meta={"source": "channel_ramp"},
    )

    _, projected = generate_projection_image(
        target,
        matrices,
        projector=_Projector(width=16, height=16),
        active_region=region,
        linearizer="inverse_lut",
        lut=lut,
        target_decode="none",
    )

    target_linear = embed_in_active_region(
        decode_target(target, mode="none"), (16, 16), region
    )
    comp_float = calc_compensation_image(
        target_image=target_linear,
        color_mixing_matrices=matrices,
        dtype=np.float32,
        clip=True,
    )
    comp_float = mask_to_active_region(comp_float, region)
    linearized = apply_inverse_lut(comp_float, lut, dtype=np.float32)
    expected = np.floor(np.clip(linearized, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    old_order = apply_inverse_lut(
        mask_to_active_region(
            calc_compensation_image(
                target_image=target_linear,
                color_mixing_matrices=matrices,
                dtype=np.uint8,
                clip=True,
            ),
            region,
        ),
        lut,
        dtype=np.uint8,
    )
    assert not np.array_equal(expected, old_order)

    assert projected.dtype == np.uint8
    np.testing.assert_array_equal(projected, expected)


# ---------------------------------------------------------------------------
# 2. 逆ガンマ関数の整数化: 切り捨てではなく四捨五入
# ---------------------------------------------------------------------------


def test_inverse_gamma_uint8_quantization_rounds_half_up() -> None:
    """uint8 化は floor(x*255+0.5)。切り捨てだと値 1 の出力が 21 でなく 20 になる。"""
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    image = np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2)
    out = _apply_inverse_gamma_numpy(image, gamma=1.0 / 2.2)

    normalized = image.astype(np.float32) / 255.0
    powered = np.clip(
        np.power(normalized, np.float32(1.0 / 2.2), dtype=np.float32), 0.0, 1.0
    )
    expected = np.floor(powered * 255.0 + 0.5).astype(np.uint8)
    truncated = (powered * 255.0).astype(np.uint8)

    # このスイープには四捨五入と切り捨てが食い違う入力が実在する
    # (無ければこのテストは切り捨て実装も通してしまう)。
    assert not np.array_equal(expected, truncated)
    # 明示的な 1 点: (1/255)^(1/2.2)*255 ≈ 20.56 → 21 (切り捨てなら 20)
    assert int(out[1, 0, 0]) == 21
    np.testing.assert_array_equal(out, expected)


def test_torch_inverse_gamma_uint8_quantization_rounds_half_up() -> None:
    """torch 版 apply_inverse_gamma_correction も同じ四捨五入規約に従うこと。

    numpy 版との全数一致は tests/test_pipeline_gamma_numpy.py が固定して
    いる。ここでは切り捨て実装を直接弾く 1 点(値 1 → 21)と、numpy 版
    との一致を確認する。
    """
    pytest.importorskip("torch")
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    image = np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2)
    torch_out = np.asarray(
        apply_inverse_gamma_correction(image, gamma=1.0 / 2.2, device="cpu")
    )
    assert int(torch_out[1, 0, 0]) == 21
    np.testing.assert_array_equal(
        torch_out, _apply_inverse_gamma_numpy(image, gamma=1.0 / 2.2)
    )


def test_inverse_gamma_uint16_quantization_rounds_half_up() -> None:
    """uint16 経路も同じ四捨五入規約に従うこと(全 65536 値)。"""
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    values = np.arange(65536, dtype=np.uint16).reshape(256, 256, 1).repeat(3, axis=2)
    out = _apply_inverse_gamma_numpy(values, gamma=1.0 / 2.2)

    normalized = values.astype(np.float32) / 65535.0
    powered = np.clip(
        np.power(normalized, np.float32(1.0 / 2.2), dtype=np.float32), 0.0, 1.0
    )
    expected = np.floor(powered * 65535.0 + 0.5).astype(np.uint16)
    truncated = (powered * 65535.0).astype(np.uint16)
    assert not np.array_equal(expected, truncated)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 3. CLI (sample.py / sample_inverse_lut.py) の暗部保持
# ---------------------------------------------------------------------------


def _install_optional_dependency_stubs() -> None:
    if "colour" not in sys.modules:
        colour = types.ModuleType("colour")
        colour.characterisation = types.SimpleNamespace(
            polynomial_expansion_Finlayson2015=lambda *args, **kwargs: None
        )
        colour.algebra = types.SimpleNamespace(vecmul=lambda *args, **kwargs: None)
        colour.XYZ_to_sRGB = lambda image, apply_cctf_encoding=False: image
        sys.modules["colour"] = colour

    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(available=1 << 30)
        sys.modules["psutil"] = psutil


def _load_module_from(path: Path, name: str):
    _install_optional_dependency_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPO_ROOT = Path(__file__).resolve().parents[1]
sample = _load_module_from(
    _REPO_ROOT / "examples" / "python" / "sample.py", "sample_module_qorder"
)
sample_inverse_lut = _load_module_from(
    _REPO_ROOT / "examples" / "python" / "sample_inverse_lut.py",
    "sample_inverse_lut_module_qorder",
)


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _load_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_identity_p2c_map(path: Path, width: int, height: int) -> None:
    rows = []
    for y in range(height):
        for x in range(width):
            rows.append([float(x), float(y), float(x), float(y)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows, dtype=np.float32))


def _write_config(path: Path, *, width: int, height: int, extra: str = "") -> None:
    path.write_text(
        f"""
[projector]
gamma = 2.2
width = {width}
height = {height}
pos_x = 0
pos_y = 0

[camera]
backend = "opencv"
device_index = 0
wait_key_ms = 1

[paths]
c2p_map = ""
p2c_map = "maps/p2c.npy"
warp_method = "p2c"
target_image_space = "projector"
target_image_dir = "data/target_images"
linear_pattern_dir = "data/linear_proj_patterns"
inv_gamma_pattern_dir = "data/inv_gamma_proj_patterns"
captured_image_dir = "data/captured_images"
compensation_image_dir = "data/compensation_images"
inv_gamma_comp_dir = "data/inv_gamma_comp_images"

[compensation]
num_divisions = 2
use_gpu = false
{extra}
""".strip(),
        encoding="utf-8",
    )


def _stub_window_calls(monkeypatch, module) -> None:
    for func_name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "resizeWindow",
        "imshow",
        "destroyWindow",
    ):
        monkeypatch.setattr(module.cv2, func_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(module.cv2, "waitKey", lambda *args, **kwargs: 0)


def _setup_dark_target_run(tmp_path, monkeypatch, *, extra_config: str = "") -> Path:
    """全画素値 1 のターゲットと恒等マップの実行環境を tmp_path に用意する。

    ターゲット値 1 は sRGB 復号後リニア約 0.0003。恒等に近い行列の下で、
    旧実装の投影コードは必ず 0(消灯)になり、float 経路なら数コードの
    非ゼロ値になる。
    """
    width, height = 4, 3
    data_dir = tmp_path / "data"
    _save_identity_p2c_map(tmp_path / "maps" / "p2c.npy", width, height)

    target_image = np.full((height, width, 3), 1, dtype=np.uint8)
    _save_rgb_image(data_dir / "target_images" / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, width=width, height=height, extra=extra_config)
    monkeypatch.chdir(tmp_path)
    return config_path


def _mock_capture_with_linear_patterns(monkeypatch, module, width, height) -> None:
    """撮影を「リニアパターンそのもの」で置き換え、恒等に近い行列を fit させる。"""
    captured_images = iter(
        module.generate_projection_patterns(
            width, height, num_divisions=2, dtype=np.dtype(np.uint8)
        )
    )
    monkeypatch.setattr(module, "capture_image", lambda: next(captured_images, None))


def test_sample_py_preserves_dark_detail_in_inverse_gamma_output(
    tmp_path, monkeypatch
) -> None:
    """sample.py の逆ガンマ投影 PNG で、リニア 1 コード未満の暗部が消灯しないこと。

    旧実装(補償値を uint8 化してから逆ガンマ)ではコード 0 に潰れる。
    float 経路なら pow(0.0003, 1/2.2)*255 ≈ 6 前後の非ゼロコードになる。
    """
    config_path = _setup_dark_target_run(tmp_path, monkeypatch)
    _stub_window_calls(monkeypatch, sample)
    _mock_capture_with_linear_patterns(monkeypatch, sample, 4, 3)

    sample.main(["sample.py", "--config", str(config_path)])

    png = _load_rgb_image(
        tmp_path / "data" / "inv_gamma_comp_images" / "inv_gamma_comp_image_00.png"
    )
    assert png.min() >= 1, "暗部が消灯している(量子化がガンマより前にある)"
    assert png.max() <= 15, "暗部の投影コードが想定より大きすぎる"


def test_sample_inverse_lut_py_preserves_dark_detail_in_lut_output(
    tmp_path, monkeypatch
) -> None:
    """sample_inverse_lut.py の逆LUT投影 PNG でも暗部が消灯しないこと。

    ガンマ形状の LUT (L(t) = t^(1/2.2)) を与える。旧実装は補償値を
    uint8 化(→ 0)してから LUT を引くため必ず 0。float 経路なら LUT の
    区分線形補間で 1〜3 コードの非ゼロ値になる。
    """
    from yoshidacomp.compensation import inverse_lut as ilut

    config_path = _setup_dark_target_run(
        tmp_path,
        monkeypatch,
        extra_config='\n[inverse_lut]\ninverse_lut_path = "data/inverse_lut.npz"',
    )
    t_grid = np.linspace(0.0, 1.0, 256).astype(np.float32)
    curve = np.power(t_grid, np.float32(1.0 / 2.2)).astype(np.float32)
    ilut.save_inverse_lut(
        tmp_path / "data" / "inverse_lut.npz",
        ilut.InverseLut(
            lut=np.stack([curve, curve, curve]),
            t_grid=t_grid,
            knots=[],
            meta={"source": "channel_ramp"},
        ),
    )

    inner_sample = sample_inverse_lut._load_sample_module()
    _stub_window_calls(monkeypatch, inner_sample)
    _stub_window_calls(monkeypatch, sample_inverse_lut)
    _mock_capture_with_linear_patterns(monkeypatch, inner_sample, 4, 3)
    monkeypatch.setattr(
        sample_inverse_lut, "_load_sample_module", lambda: inner_sample
    )

    sample_inverse_lut.main(
        ["sample_inverse_lut.py", "--config", str(config_path)]
    )

    png = _load_rgb_image(
        tmp_path / "data" / "inv_lut_comp_images" / "inv_lut_comp_image_00.png"
    )
    assert png.min() >= 1, "暗部が消灯している(量子化がLUTより前にある)"
    assert png.max() <= 15, "暗部の投影コードが想定より大きすぎる"
