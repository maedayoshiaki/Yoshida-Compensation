# パッケージ分割と純計算層の抽出 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 現在の `src/python/` を uv workspace の正式パッケージ `yoshidacomp` へ移し、`examples/python/sample.py` に混在しているハードウェア非依存の計算をライブラリ層へ抽出する。

**Architecture:** `packages/core`（配布名 `yoshidacomp`）を setuptools パッケージとして作り、リポジトリルートを uv workspace root にする。移行中は `src/python/` も import 可能なまま残し、モジュールを1つずつ移して都度テストを通す。最後に `src/python/` を削除する。Web層（`packages/web`）は本計画の対象外で、後続の計画で追加する。

**Tech Stack:** Python 3.13, uv workspace, setuptools, pytest, numpy, torch, OpenCV

## Global Constraints

- `requires-python = ">=3.13"`
- `psutil` は `yoshidacomp/compensation/color_mixing_matrix.py` が module scope で import するため**必須依存**（現在は `sample` extra に入っているが誤り）
- `rawpy` と `edsdk-python` は optional extra `camera` に置く。`yoshidacomp` の module scope から import しない
- `torch` は `pytorch-cu128` インデックスから取得する（workspace root の `[tool.uv.sources]` で指定）
- GrayCode は `external/GrayCode` の submodule のまま。`tests/conftest.py` の `sys.path` 挿入は維持する
- コミットメッセージは日本語。コード・コメント・docstring は現行の英日併記スタイルを維持する
- **各タスクの完了時に `.venv/Scripts/python.exe -m pytest -q` が全て緑であること。**
  期待件数は各タスクの Step 5 に記載（開始時点は 70 passed）。移行中も `src/python/`
  は import 可能なまま残すため、どのタスクの終了時点でも suite は緑になる
- 実行は `.venv/Scripts/python.exe` を使う（`uv run` でも可）

---

### Task 1: uv workspace と空の core パッケージ

**Files:**
- Create: `packages/core/pyproject.toml`
- Create: `packages/core/yoshidacomp/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_package_layout.py`

**Interfaces:**
- Consumes: なし
- Produces: import 可能な `yoshidacomp` パッケージ。`yoshidacomp.__version__ -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_package_layout.py`:

```python
"""Package layout guards for the yoshidacomp core distribution."""

from __future__ import annotations


def test_core_package_is_importable() -> None:
    import yoshidacomp

    assert yoshidacomp.__version__ == "0.1.0"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp'`

- [ ] **Step 3: core パッケージを作る**

`packages/core/yoshidacomp/__init__.py`:

```python
"""Projector-camera photometric compensation core.

プロジェクタ・カメラ間の測光補償のコアライブラリ。
ハードウェア非依存の計算のみを含む。
"""

__version__ = "0.1.0"
```

`packages/core/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "yoshidacomp"
version = "0.1.0"
description = "Projector-camera photometric compensation core"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.3.4",
    "opencv-contrib-python>=4.11.0.86",
    "pillow>=12.0.0",
    "psutil>=7.0.0",
    "scipy>=1.16.3",
    "torch>=2.9.1",
]

[project.optional-dependencies]
camera = [
    "rawpy",
    "edsdk-python @ https://github.com/youme-town/edsdk-python/releases/download/v0.2.0/edsdk_python-0.2.0-cp313-cp313-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.13'",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["yoshidacomp*"]
```

- [ ] **Step 4: root pyproject.toml を workspace root にする**

`pyproject.toml` を次の内容へ差し替える:

```toml
[project]
name = "yoshida-compensation"
version = "0.1.0"
description = "Projector-camera photometric compensation research workspace"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "yoshidacomp",
]

[project.optional-dependencies]
sample = [
    "yoshidacomp[camera]",
    # Plotting for validate_inverse_lut.py (linearity check plots).
    "matplotlib",
]

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
yoshidacomp = { workspace = true }
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[dependency-groups]
dev = [
    "pytest>=9.0.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: 同期してテストが通ることを確認**

Run: `uv sync`
Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 71 passed（既存70 + 新規1）

`uv sync` が既存の EDSDK 環境を壊した場合は `docs/` の EDSDK セットアップ手順に従って復旧する。

- [ ] **Step 6: コミット**

```bash
git add pyproject.toml uv.lock packages/core tests/test_package_layout.py
git commit -m "build: uv workspace化してcoreパッケージ yoshidacomp を追加

src/python は当面そのまま残し、モジュールを順次移行する。
psutil は color_mixing_matrix.py が module scope で import するため
sample extra から必須依存へ移した。"
```

---

### Task 2: config.py の移行

**Files:**
- Create: `packages/core/yoshidacomp/config.py`（`src/python/config.py` を移動）
- Delete: `src/python/config.py`
- Modify: `src/python/{color_mixing_matrix,inverse_lut,cap_chess,solvePnP,project_and_capture}.py`, `src/python/camera/{base,factory}.py`, `examples/python/*.py`, `tests/{test_canon_edsdk,test_inverse_lut}.py`
- Test: `tests/test_package_layout.py`

**Interfaces:**
- Consumes: Task 1 の `yoshidacomp` パッケージ
- Produces: `yoshidacomp.config` に `AppConfig`, `ProjectorConfig`, `CameraConfig`, `PathsConfig`, `CompensationConfig`, `InverseLutConfig`, `load_config(path=None)`, `get_config(path=None)`, `reload_config(path=None)`, `generate_config_file(...)`, `split_cli_config_path(argv)` を公開

- [ ] **Step 1: 失敗するテストを追加**

`tests/test_package_layout.py` に追記:

```python
def test_config_lives_in_core() -> None:
    from pathlib import Path

    from yoshidacomp.config import (
        AppConfig,
        get_config,
        reload_config,
        split_cli_config_path,
    )

    assert AppConfig().projector.gamma == 2.2
    assert callable(get_config)
    assert callable(reload_config)
    assert split_cli_config_path(["prog", "--config", "x.toml"]) == (
        ["prog"],
        Path("x.toml"),
    )


def test_config_still_reads_the_repository_config_toml() -> None:
    """_PROJECT_ROOT の階層修正が正しいことの確認。

    誤ると config.toml を見つけられず全既定値へ静かにフォールバックする。
    """
    from yoshidacomp.config import reload_config

    cfg = reload_config()
    assert cfg.projector.active_width == 722
    assert cfg.projector.active_height == 722
    assert cfg.compensation.num_divisions == 4
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py::test_config_lives_in_core -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp.config'`

- [ ] **Step 3: ファイルを移動**

```bash
git mv src/python/config.py packages/core/yoshidacomp/config.py
```

`config.py` はプロジェクトルートを `Path(__file__).resolve().parent.parent.parent` で求めている。移動後の階層は `packages/core/yoshidacomp/config.py` なので、`parents[3]` がリポジトリルートになる。該当箇所を次のように直す:

```python
# packages/core/yoshidacomp/config.py の該当行
# 変更前: _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
```

- [ ] **Step 4: import を一括置換**

```bash
grep -rl "src\.python\.config" src tests examples --include='*.py' \
  | xargs sed -i 's/from src\.python\.config import/from yoshidacomp.config import/g'
grep -rn "src\.python\.config" src tests examples --include='*.py'
```

最後の `grep` は何も出力しないこと。出力があれば `import src.python.config` 形式なので手で直す。

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 73 passed（71 + 新規2）

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: config を yoshidacomp.config へ移行

プロジェクトルートの算出を parents[3] に修正。
import 20箇所を一括置換した。"
```

---

### Task 3: projection_region.py の移行

**Files:**
- Create: `packages/core/yoshidacomp/compensation/__init__.py`
- Create: `packages/core/yoshidacomp/compensation/projection_region.py`（移動）
- Delete: `src/python/projection_region.py`
- Modify: `examples/python/{build_inverse_lut,calibrate_color_matrix,calibrate_graycode_geometry,capture_grayscale_levels,check_projection_exposure,rebuild_compensation_with_map,sample,sample_inverse_lut}.py`, `tests/test_projection_region.py`

**Interfaces:**
- Consumes: Task 2 の `yoshidacomp.config`
- Produces: `yoshidacomp.compensation.projection_region` に
  `resolve_active_region(projector) -> tuple[int,int,int,int]`,
  `mask_to_active_region(image, region) -> ndarray`,
  `embed_in_active_region(image, frame_size, region) -> ndarray`,
  Protocol `ProjectorRegionConfig`

- [ ] **Step 1: 失敗するテストを追加**

`tests/test_package_layout.py` に追記:

```python
def test_projection_region_lives_in_core() -> None:
    import numpy as np
    from yoshidacomp.compensation.projection_region import (
        embed_in_active_region,
        mask_to_active_region,
    )

    image = np.ones((4, 4, 3), dtype=np.uint8)
    masked = mask_to_active_region(image, (1, 1, 2, 2))
    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[1, 1].tolist() == [1, 1, 1]

    embedded = embed_in_active_region(image, (8, 6), (2, 1, 4, 4))
    assert embedded.shape == (6, 8, 3)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py::test_projection_region_lives_in_core -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp.compensation'`

- [ ] **Step 3: サブパッケージを作って移動**

`packages/core/yoshidacomp/compensation/__init__.py`:

```python
"""Photometric compensation modules.

測光補償モジュール群。
"""
```

```bash
git mv src/python/projection_region.py packages/core/yoshidacomp/compensation/projection_region.py
```

- [ ] **Step 4: import を一括置換**

```bash
grep -rl "src\.python\.projection_region" src tests examples --include='*.py' \
  | xargs sed -i 's/from src\.python\.projection_region import/from yoshidacomp.compensation.projection_region import/g'
grep -rn "src\.python\.projection_region" src tests examples --include='*.py'
```

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 74 passed（73 + 新規1）

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: projection_region を yoshidacomp.compensation へ移行"
```

---

### Task 4: photometric_compensation.py と color_mixing_matrix.py の移行

**Files:**
- Create: `packages/core/yoshidacomp/compensation/photometric_compensation.py`（移動）
- Create: `packages/core/yoshidacomp/compensation/color_mixing_matrix.py`（移動）
- Delete: `src/python/photometric_compensation.py`, `src/python/color_mixing_matrix.py`
- Modify: `examples/python/{sample,sample_inverse_lut,calibrate_color_matrix,rebuild_compensation_with_map}.py`, `tests/test_projection_region.py`

**Interfaces:**
- Consumes: Task 2 の `yoshidacomp.config`
- Produces:
  - `yoshidacomp.compensation.photometric_compensation.calc_compensation_image(target_image, color_mixing_matrices, dtype=np.float32) -> ndarray`
  - `yoshidacomp.compensation.color_mixing_matrix.generate_projection_patterns(x_size, y_size, num_divisions=None, dtype=np.uint8) -> list[ndarray]`
  - `yoshidacomp.compensation.color_mixing_matrix.apply_inverse_gamma_correction(image, gamma=None, device="cuda")` — **指数をそのまま pow に渡す。明るくするには `gamma=1/2.2` を渡すこと**
  - `yoshidacomp.compensation.color_mixing_matrix.calc_color_mixing_matrices(proj_images, captured_images, ref_x=0, ref_y=0, safety_margin=None, min_batch_size=None, use_gpu=None) -> ndarray`

- [ ] **Step 1: 失敗するテストを追加**

`tests/test_package_layout.py` に追記:

```python
def test_compensation_math_lives_in_core() -> None:
    import numpy as np
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
        generate_projection_patterns,
    )
    from yoshidacomp.compensation.photometric_compensation import calc_compensation_image

    patterns = generate_projection_patterns(2, 2, num_divisions=2, dtype=np.uint8)
    assert len(patterns) == 8
    assert patterns[0].shape == (2, 2, 3)

    # gamma=1/2.2 で明るくなること（この関数は指数をそのまま pow に使う）
    mid = np.full((1, 1, 3), 64, dtype=np.uint8)
    brightened = apply_inverse_gamma_correction(mid, gamma=1 / 2.2, device="cpu")
    assert int(brightened[0, 0, 0]) > 64

    identity = np.zeros((1, 1, 4, 3), dtype=np.float32)
    identity[0, 0, 0, 0] = 1.0
    identity[0, 0, 1, 1] = 1.0
    identity[0, 0, 2, 2] = 1.0
    target = np.full((1, 1, 3), 0.5, dtype=np.float32)
    out = calc_compensation_image(target, identity, dtype=np.float32)
    assert np.allclose(out, 0.5, atol=1e-5)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py::test_compensation_math_lives_in_core -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: ファイルを移動**

```bash
git mv src/python/photometric_compensation.py packages/core/yoshidacomp/compensation/photometric_compensation.py
git mv src/python/color_mixing_matrix.py packages/core/yoshidacomp/compensation/color_mixing_matrix.py
```

- [ ] **Step 4: import を一括置換**

```bash
grep -rl "src\.python\.photometric_compensation\|src\.python\.color_mixing_matrix" src tests examples --include='*.py' \
  | xargs sed -i \
    -e 's/from src\.python\.photometric_compensation import/from yoshidacomp.compensation.photometric_compensation import/g' \
    -e 's/from src\.python\.color_mixing_matrix import/from yoshidacomp.compensation.color_mixing_matrix import/g'
grep -rn "src\.python\.photometric_compensation\|src\.python\.color_mixing_matrix" src tests examples --include='*.py'
```

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 76 passed（74 + 新規1、および Task 4 の回帰テスト1件）

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: 補償計算2モジュールを yoshidacomp.compensation へ移行"
```

---

### Task 5: camera/ を capture/ へ移行

**Files:**
- Create: `packages/core/yoshidacomp/capture/{__init__,base,factory,canon_edsdk,opencv}.py`（移動）
- Delete: `src/python/camera/`
- Modify: `src/python/{inverse_lut,cap_chess,solvePnP}.py`, `examples/python/calibrate_graycode_geometry.py`, `tests/test_canon_edsdk.py`
- Test: `tests/test_canon_edsdk.py`（既存・import 変更のみ）

**Interfaces:**
- Consumes: Task 2 の `yoshidacomp.config`
- Produces: `yoshidacomp.capture` に `create_camera_backend(camera_config)`, `CameraBackend`（抽象、`capture_linear_rgb(tv=None)`）, `CameraCaptureError`, `CanonEdsdkCamera`, `capture_srgb_uint8`

- [ ] **Step 1: 失敗するテストを追加**

`tests/test_package_layout.py` に追記:

```python
def test_capture_package_exposes_backend_factory() -> None:
    """`yoshidacomp.capture` が import でき、バックエンド生成口を公開すること。

    SDK非搭載環境でも import できることの保証は Task 10 の
    `tests/test_package_boundaries.py::test_core_imports_without_camera_sdk`
    が担う（このvenvには rawpy と edsdk が実在するため、ここでは検証できない）。
    """
    import yoshidacomp.capture as capture

    assert hasattr(capture, "create_camera_backend")
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py::test_capture_package_exposes_backend_factory -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp.capture'`

- [ ] **Step 3: ディレクトリを移動**

```bash
git mv src/python/camera packages/core/yoshidacomp/capture
```

- [ ] **Step 4: import を一括置換**

```bash
grep -rl "src\.python\.camera\|src\.python import camera" src tests examples --include='*.py' \
  | xargs sed -i \
    -e 's/from src\.python\.camera\.base import/from yoshidacomp.capture.base import/g' \
    -e 's/from src\.python\.camera\.canon_edsdk import/from yoshidacomp.capture.canon_edsdk import/g' \
    -e 's/from src\.python\.camera\.factory import/from yoshidacomp.capture.factory import/g' \
    -e 's/from src\.python\.camera import/from yoshidacomp.capture import/g'
grep -rn "src\.python\.camera" src tests examples --include='*.py'
```

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 77 passed（76 + 新規1）

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: camera を yoshidacomp.capture へ移行

Densoten の shared/camera と同様、撮影系を独立サブパッケージにした。
edsdk と rawpy はメソッド内で遅延importするため、SDK無しでも import できる。"
```

---

### Task 6: inverse_lut.py の移行

**Files:**
- Create: `packages/core/yoshidacomp/compensation/inverse_lut.py`（移動）
- Delete: `src/python/inverse_lut.py`
- Modify: `examples/python/{build_inverse_lut,calibrate_color_matrix,sample_inverse_lut,validate_inverse_lut,verify_hdr_tv,rebuild_compensation_with_map}.py`, `tests/{test_inverse_lut,test_sample_inverse_lut_mocked,test_validate_inverse_lut_mocked}.py`

**Interfaces:**
- Consumes: Task 2 の `yoshidacomp.config`、Task 5 の `yoshidacomp.capture`
- Produces: `yoshidacomp.compensation.inverse_lut` に
  `InverseLut`（`lut: (3, R) float32`, `t_grid: (R,) float32`, `knots: list`, `meta: dict`）,
  `load_inverse_lut(path) -> InverseLut`, `save_inverse_lut(path, lut) -> Path`,
  `apply_inverse_lut(image, lut, dtype=np.float32) -> ndarray`,
  `build_inverse_lut(data, *, source=None, lut_resolution=None, trust_low=None, trust_high=None) -> InverseLut`,
  `generate_calibration_patterns(...)`, `measure_calibration_samples(...)`, `normalize_source(source) -> str`

- [ ] **Step 1: 失敗するテストを追加**

`tests/test_package_layout.py` に追記:

```python
def test_inverse_lut_lives_in_core(tmp_path) -> None:
    import numpy as np
    from yoshidacomp.compensation.inverse_lut import (
        InverseLut,
        apply_inverse_lut,
        load_inverse_lut,
        save_inverse_lut,
    )

    t_grid = np.linspace(0.0, 1.0, 256).astype(np.float32)
    lut = InverseLut(
        lut=np.stack([t_grid, t_grid, t_grid]).astype(np.float32),
        t_grid=t_grid,
        knots=[],
        meta={"source": "channel_ramp"},
    )
    path = save_inverse_lut(tmp_path / "identity.npz", lut)
    reloaded = load_inverse_lut(path)
    assert reloaded.lut.shape == (3, 256)

    # identity LUT なので入力がそのまま返る
    image = np.full((1, 1, 3), 0.25, dtype=np.float32)
    out = apply_inverse_lut(image, reloaded, dtype=np.float32)
    assert np.allclose(out, 0.25, atol=1e-3)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py::test_inverse_lut_lives_in_core -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: ファイルを移動**

```bash
git mv src/python/inverse_lut.py packages/core/yoshidacomp/compensation/inverse_lut.py
```

- [ ] **Step 4: import を一括置換**

`from src.python import inverse_lut as ilut` 形式が複数あるので2種類を置換する:

```bash
grep -rl "src\.python.*inverse_lut" src tests examples --include='*.py' \
  | xargs sed -i \
    -e 's/from src\.python import inverse_lut as ilut/from yoshidacomp.compensation import inverse_lut as ilut/g' \
    -e 's/from src\.python import inverse_lut/from yoshidacomp.compensation import inverse_lut/g' \
    -e 's/from src\.python\.inverse_lut import/from yoshidacomp.compensation.inverse_lut import/g'
grep -rn "src\.python.*inverse_lut" src tests examples --include='*.py'
```

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 78 passed（77 + 新規1）

`tests/test_sample_inverse_lut_mocked.py` と `tests/test_validate_inverse_lut_mocked.py` は
`sys.modules` にスタブを差し込む形式なので、スタブ側のキー名も `src.python.*` から
`yoshidacomp.*` へ直す必要がある。失敗したら該当箇所を修正する。

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: inverse_lut を yoshidacomp.compensation へ移行"
```

---

### Task 7: 校正スクリプトの移行と src/python の削除

**Files:**
- Create: `packages/core/yoshidacomp/calibration/{__init__,cam_in_calib,cap_chess,solve_pnp}.py`（移動）
- Create: `examples/python/project_and_capture.py`（`src/python/project_and_capture.py` を移動）
- Delete: `src/python/`（空になる）
- Modify: `tests/test_package_layout.py`

**Interfaces:**
- Consumes: Task 2 の `yoshidacomp.config`、Task 5 の `yoshidacomp.capture`
- Produces: `yoshidacomp.calibration.{cam_in_calib, cap_chess, solve_pnp}`。`cap_chess` と `solve_pnp` は `main(argv=None)`、`cam_in_calib` は引数なしの `main()` を持つ

- [ ] **Step 1: 失敗するテストを追加**

`tests/test_package_layout.py` に追記:

```python
def test_src_python_is_gone() -> None:
    """移行完了後、src/python は存在しないこと。"""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "src" / "python").exists()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_layout.py::test_src_python_is_gone -q`
Expected: FAIL — `assert not True`

- [ ] **Step 3: 残りを移動**

```bash
mkdir -p packages/core/yoshidacomp/calibration
git mv src/python/cam_in_calib.py packages/core/yoshidacomp/calibration/cam_in_calib.py
git mv src/python/cap_chess.py     packages/core/yoshidacomp/calibration/cap_chess.py
git mv src/python/solvePnP.py      packages/core/yoshidacomp/calibration/solve_pnp.py
git mv src/python/project_and_capture.py examples/python/project_and_capture.py
```

`packages/core/yoshidacomp/calibration/__init__.py`:

```python
"""Camera and projector-camera geometric calibration entry points.

カメラおよびプロジェクタ・カメラ間の幾何校正のエントリポイント。
"""
```

- [ ] **Step 4: 残った import と sys.path 細工を除去**

`examples/python/project_and_capture.py` は `examples.python.sample.capture_image()` を
使うため、リポジトリルートを `sys.path` に入れる必要がある。他のスクリプトと同じ形に揃える:

```python
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

`examples/python/*.py` の各ファイルにある `_REPO_ROOT` の `sys.path` 挿入は、
`yoshidacomp` がインストール済みになったため **`examples.python.sample` を import する
スクリプトだけに必要**である。次のコマンドで残存を確認し、不要なものを削除する:

```bash
grep -rn "sys.path.insert" examples tests --include='*.py'
grep -rn "src\.python" . --include='*.py'
```

2つ目の `grep` は何も出力しないこと。

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 79 passed（78 + 新規1）

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: 校正スクリプトを yoshidacomp.calibration へ移し src/python を削除

solvePnP.py は Densoten と同じ solve_pnp.py へ改名した。
project_and_capture.py は対話スクリプトなので examples/python へ移した。"
```

---

### Task 8: 幾何ワープの抽出

**Files:**
- Create: `packages/core/yoshidacomp/geometry/__init__.py`
- Create: `packages/core/yoshidacomp/geometry/warp.py`
- Modify: `examples/python/sample.py`（該当関数を削除して re-export に）
- Test: `tests/test_geometry_warp.py`

**Interfaces:**
- Consumes: Task 4 の `yoshidacomp.compensation`
- Produces: `yoshidacomp.geometry.warp` に
  - `center_rect(image_w, image_h, proj_w, proj_h) -> tuple[int,int,int,int]`
  - `create_warper(pixel_map_path, warp_method)` — GrayCode の `PixelMapWarperTorch` を返す
  - `warp_camera_array_to_projector(array, pixel_map_path, proj_width, proj_height, image_width, image_height, warp_method="p2c") -> ndarray`
  - `warp_color_mixing_matrices_to_projector(color_mixing_matrices, pixel_map_path, proj_width, proj_height, image_width, image_height, warp_method="p2c") -> ndarray` — 入力 `(H,W,4,3)`、出力 `(h,w,4,3)` float32
  - `warp_image(...)`, `invwarp_image(...)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_geometry_warp.py`:

```python
"""Geometry warp extraction tests (no hardware, synthetic P2C map)."""

from __future__ import annotations

import numpy as np
import pytest


def test_center_rect_centers_and_clamps() -> None:
    from yoshidacomp.geometry.warp import center_rect

    assert center_rect(100, 50, 200, 100) == (50, 25, 100, 50)
    # 画像がプロジェクタより大きい場合はクランプされる
    x, y, w, h = center_rect(400, 300, 200, 100)
    assert (x, y) == (0, 0)
    assert w <= 200 and h <= 100


def test_identity_p2c_preserves_matrix(tmp_path) -> None:
    """恒等P2C（proj座標=cam座標）なら行列がほぼそのまま返ること。"""
    from yoshidacomp.geometry.warp import warp_color_mixing_matrices_to_projector

    width, height = 8, 6
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    rows = np.stack(
        [xs.ravel(), ys.ravel(), xs.ravel(), ys.ravel()], axis=1
    ).astype(np.float32)
    map_path = tmp_path / "identity_p2c.npy"
    np.save(map_path, rows)

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, 0, 0] = 2.0
    matrices[:, :, 3, 1] = -0.5

    warped = warp_color_mixing_matrices_to_projector(
        matrices, str(map_path), width, height, width, height, warp_method="p2c"
    )
    assert warped.shape == (height, width, 4, 3)
    assert warped[height // 2, width // 2, 0, 0] == pytest.approx(2.0, abs=1e-4)
    assert warped[height // 2, width // 2, 3, 1] == pytest.approx(-0.5, abs=1e-4)


def test_rejects_wrong_matrix_shape(tmp_path) -> None:
    from yoshidacomp.geometry.warp import warp_color_mixing_matrices_to_projector

    map_path = tmp_path / "p2c.npy"
    np.save(map_path, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        warp_color_mixing_matrices_to_projector(
            np.zeros((4, 4, 3), dtype=np.float32), str(map_path), 4, 4, 4, 4
        )
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_geometry_warp.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp.geometry'`

- [ ] **Step 3: warp.py を作る**

`packages/core/yoshidacomp/geometry/__init__.py`:

```python
"""Projector-camera geometric mapping.

プロジェクタ・カメラ間の幾何対応。
"""
```

`packages/core/yoshidacomp/geometry/warp.py` を作り、`examples/python/sample.py` から
次の関数を**そのまま**移す（ロジックは変更しない）。行番号は `v1.0-hdr-lut` 時点のもの。

| 移す関数 | sample.py の行 |
| --- | --- |
| `center_rect` | 111 |
| `_create_warper` → `create_warper` として公開 | 315 |
| `_warp_camera_array_to_projector` → `warp_camera_array_to_projector` | 330 |
| `warp_image` | 818 |
| `warp_color_mixing_matrices_to_projector` | 872 |
| `invwarp_image` | 936 |

GrayCode の import は Densoten 互換のため次の形にする（ファイル先頭ではなく
`create_warper` の中で行い、`import yoshidacomp` が torch を引かないようにする）:

```python
def create_warper(pixel_map_path: str, warp_method: WarpMethod) -> "PixelMapWarperTorch":
    """Build a PixelMapWarperTorch for the given correspondence map.

    与えられた対応マップから PixelMapWarperTorch を構築する。

    GrayCode は submodule 配置とベンダリング配置の両方があり得るため、
    その差異をこの関数だけで吸収する。import を関数内に置くことで
    ``import yoshidacomp`` が torch を引かないようにしている。
    """
    try:
        from procam.graycode.warp_image import PixelMapWarperTorch  # vendored layout
    except ImportError:
        from graycode.warp_image import PixelMapWarperTorch  # submodule layout

    if warp_method == "p2c":
        from graycode.interpolate_p2c import load_p2c_numpy_array

        pixel_map = load_p2c_numpy_array(pixel_map_path)
    elif warp_method == "c2p":
        from graycode.interpolate_c2p import load_c2p_numpy_array

        pixel_map = load_c2p_numpy_array(pixel_map_path)
    else:
        raise ValueError(f"unknown warp_method: {warp_method!r}")

    return PixelMapWarperTorch(pixel_map)
```

`interpolate_p2c` / `interpolate_c2p` の import も、ベンダリング配置では
`procam.graycode.*` になる。上の `try/except` と同じ形で吸収するか、
モジュール参照を1つの変数にまとめてから属性アクセスする。
`sample.py:315` の `_create_warper` の分岐条件と読み込み関数はそのまま引き継ぐこと。

- [ ] **Step 4: sample.py を re-export に置き換える**

`examples/python/sample.py` から移した関数の本体を削除し、先頭付近に追加する:

```python
from yoshidacomp.geometry.warp import (
    center_rect,
    invwarp_image,
    warp_color_mixing_matrices_to_projector,
    warp_image,
)
```

`_warp_camera_array_to_projector` と `_create_warper` を参照していた箇所は
`warp_camera_array_to_projector` / `create_warper` に置き換える。

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 82 passed（79 + 新規3）

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "refactor: 幾何ワープを yoshidacomp.geometry.warp へ抽出

GrayCode の import は create_warper 内に閉じ込め、
import yoshidacomp が torch を引かないようにした。"
```

---

### Task 9: 生成パイプラインの抽出

**Files:**
- Create: `packages/core/yoshidacomp/compensation/pipeline.py`
- Modify: `examples/python/sample.py`, `examples/python/rebuild_compensation_with_map.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 3・4・6・8 のすべて
- Produces: `yoshidacomp.compensation.pipeline` に
  - `srgb_to_linear(a) -> ndarray`, `linear_to_srgb(a) -> ndarray`
  - `capture_to_srgb_uint8(image) -> ndarray`
  - `load_target_images(target_dir) -> list[tuple[str, ndarray]]` — `(filename, (H,W,3) uint8 RGB)`
  - `decode_target(target_uint8, *, mode: Literal["none","srgb","gamma"], gamma: float = 2.2) -> ndarray`
  - `resolve_warp_settings(paths) -> tuple[str, str]`, `resolve_target_image_space(paths) -> str`, `compensation_space_from_warp_method(m) -> str`
  - `Linearizer` プロトコルと `LINEARIZERS: dict[str, Linearizer]`（キー `"none"`, `"gamma"`, `"inverse_lut"`）
  - `generate_projection_image(target_uint8, warped_matrix, *, projector, active_region, linearizer, lut=None, target_decode="srgb", target_gamma=2.2) -> tuple[ndarray, ndarray]` — `(線形補償画像 uint8, 投影用コード値 uint8)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_pipeline.py`:

```python
"""Hardware-free generation pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_srgb_roundtrip() -> None:
    from yoshidacomp.compensation.pipeline import linear_to_srgb, srgb_to_linear

    values = np.array([0.0, 0.02, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-5)


def test_decode_target_modes() -> None:
    from yoshidacomp.compensation.pipeline import decode_target

    codes = np.full((1, 1, 3), 128, dtype=np.uint8)
    assert np.allclose(decode_target(codes, mode="none"), 128 / 255.0, atol=1e-6)
    assert decode_target(codes, mode="srgb")[0, 0, 0] < 128 / 255.0
    assert np.allclose(
        decode_target(codes, mode="gamma", gamma=2.0), (128 / 255.0) ** 2.0, atol=1e-6
    )
    with pytest.raises(ValueError):
        decode_target(codes, mode="gamma", gamma=0.0)


def test_linearizers_registered() -> None:
    from yoshidacomp.compensation.pipeline import LINEARIZERS

    assert set(LINEARIZERS) == {"none", "gamma", "inverse_lut"}


def test_generate_projection_image_identity_matrix() -> None:
    """恒等行列 + linearizer=none なら、投影画像は線形化したターゲットに一致する。"""
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 1.0

    target = np.full((4, 4, 3), 255, dtype=np.uint8)
    linear, projected = generate_projection_image(
        target,
        matrices,
        projector=Projector(),
        active_region=(0, 0, 4, 4),
        linearizer="none",
        target_decode="srgb",
    )
    assert linear.shape == (4, 4, 3)
    assert projected.shape == (4, 4, 3)
    # 白のsRGBは線形1.0 → 恒等行列 → 255
    assert int(linear[0, 0, 0]) == 255
    assert np.array_equal(linear, projected)  # linearizer="none" は素通し
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp.compensation.pipeline'`

- [ ] **Step 3: pipeline.py を作る**

`examples/python/sample.py` から `srgb_to_linear`(389), `linear_to_srgb`(414),
`capture_to_srgb_uint8`(443), `_load_target_images`(300), `resolve_warp_settings`(80),
`resolve_target_image_space`(94), `compensation_space_from_warp_method`(104),
`prepare_target_image_for_compensation`(899) を移す。
`examples/python/rebuild_compensation_with_map.py` の `_decode_target`(67) を
`decode_target` として移す。

線形化レジストリを追加する:

```python
def _linearize_none(image: ndarray, *, projector, lut=None) -> ndarray:
    """Pass the compensation codes through unchanged.

    補償コード値をそのまま通す。
    """
    return image


def _linearize_gamma(image: ndarray, *, projector, lut=None) -> ndarray:
    """Apply the analytic inverse gamma of the projector.

    プロジェクタの解析逆ガンマを適用する。

    Note: apply_inverse_gamma_correction() uses the given exponent directly,
    so brightening requires 1 / projector.gamma.
    注: apply_inverse_gamma_correction() は与えた指数をそのまま使うため、
    明るくするには 1 / projector.gamma を渡す必要がある。
    """
    return apply_inverse_gamma_correction(image, gamma=1.0 / projector.gamma)


def _linearize_inverse_lut(image: ndarray, *, projector, lut=None) -> ndarray:
    """Apply a measured per-channel inverse LUT.

    実測のチャンネル別逆LUTを適用する。
    """
    if lut is None:
        raise ValueError("linearizer='inverse_lut' requires an InverseLut")
    return apply_inverse_lut(image, lut, dtype=np.uint8)


LINEARIZERS = {
    "none": _linearize_none,
    "gamma": _linearize_gamma,
    "inverse_lut": _linearize_inverse_lut,
}
```

`generate_projection_image` は仕様書 §10 の順序で実装する:
線形化 → `embed_in_active_region` → `calc_compensation_image(dtype=np.uint8)` →
`mask_to_active_region` → linearizer 適用。

- [ ] **Step 4: 2つのCLIを pipeline を使う形に書き換える**

`examples/python/rebuild_compensation_with_map.py` の `_decode_target` を削除し、
`from yoshidacomp.compensation.pipeline import decode_target, generate_projection_image` を
使う。`sample.py` も同様に移した関数を import する。

`importlib.util.spec_from_file_location` による `sample.py` の直接ロードは削除する。

- [ ] **Step 5: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 86 passed（82 + 新規4）

- [ ] **Step 6: 既存CLIとの出力一致を実データで確認**

保存済みの行列とターゲットで `rebuild_compensation_with_map.py` を実行し、
`v1.0-hdr-lut` 時点の出力と画素一致することを確認する:

先に基準となる出力を `v1.0-hdr-lut` で作る（作業ツリーを汚さないよう worktree を使う）:

```bash
git worktree add ../ycomp-baseline v1.0-hdr-lut
cd ../ycomp-baseline && git submodule update --init
.venv/Scripts/python.exe examples/python/rebuild_compensation_with_map.py \
  --config config.toml \
  --matrix data/linear_pipeline_1080/matrices_linear_gamma.npy \
  --linearizer gamma \
  --target-dir data/comp_test_targets_square \
  --output-dir ../rebuild_baseline
```

`v1.0-hdr-lut` の worktree には `data/` の実体が無いため、`--matrix` と
`--target-dir` は元リポジトリの絶対パスを指定する。次に現在のブランチで同じ入力を通す:

```bash
.venv/Scripts/python.exe examples/python/rebuild_compensation_with_map.py \
  --config config.toml \
  --matrix data/linear_pipeline_1080/matrices_linear_gamma.npy \
  --linearizer gamma \
  --target-dir data/comp_test_targets_square \
  --output-dir ../rebuild_after
```

両者を比較する。差分が無いこと:

```bash
diff -r ../rebuild_baseline/compensation_projector ../rebuild_after/compensation_projector
diff -r ../rebuild_baseline/compensation_linear    ../rebuild_after/compensation_linear
```

確認後に worktree を片付ける:

```bash
git worktree remove ../ycomp-baseline
```

- [ ] **Step 7: コミット**

```bash
git add -A
git commit -m "refactor: 生成パイプラインを yoshidacomp.compensation.pipeline へ抽出

線形化を none/gamma/inverse_lut のレジストリにした。
rebuild_compensation_with_map.py が importlib で sample.py を
直接ロードしていた回避策を解消した。"
```

---

### Task 10: 境界テストと最終確認

**Files:**
- Create: `tests/test_package_boundaries.py`
- Modify: `README.md`
- Test: `tests/test_package_boundaries.py`

**Interfaces:**
- Consumes: Task 1〜9 のすべて
- Produces: なし（保護のみ）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_package_boundaries.py`:

```python
"""Layering guards for the yoshidacomp workspace.

core は Web層に依存しない。core はカメラSDKなしで import できる。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_DIR = _REPO_ROOT / "packages" / "core" / "yoshidacomp"
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(yoshidacomp_web)\b", re.MULTILINE)


def test_core_does_not_import_web() -> None:
    offenders = []
    for path in _CORE_DIR.rglob("*.py"):
        if _IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"core must not import the web layer: {offenders}"


def test_core_imports_without_camera_sdk() -> None:
    """edsdk / rawpy / fastapi を遮断しても core が import できること。"""
    script = """
import sys

BLOCKED = {"edsdk", "rawpy", "fastapi", "uvicorn"}


class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in BLOCKED else None

    def load_module(self, name):
        raise ImportError(f"blocked: {name}")


sys.meta_path.insert(0, Blocker())

import yoshidacomp
import yoshidacomp.config
import yoshidacomp.capture
import yoshidacomp.compensation.pipeline
import yoshidacomp.compensation.inverse_lut
import yoshidacomp.geometry.warp

print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_package_boundaries.py -q`
Expected: `test_core_imports_without_camera_sdk` が FAIL する場合、core のどこかが
SDK を module scope で import している。該当箇所をメソッド内の遅延importへ直す。
両方すでに通る場合はこのタスクは保護テストの追加のみとなる。

- [ ] **Step 3: 違反があれば修正する**

`yoshidacomp/capture/canon_edsdk.py` の `edsdk` / `rawpy` import が
module scope にあればメソッド内へ移す。

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 88 passed（86 + 新規2）

- [ ] **Step 5: README を更新**

`README.md` のセットアップとインポート例を新レイアウトに合わせる。

```markdown
## パッケージ構成

- `packages/core`（配布名 `yoshidacomp`）— ハードウェア非依存の計算コア
- `examples/python/` — 撮影を含むCLI

```python
from yoshidacomp.config import get_config
from yoshidacomp.compensation.pipeline import generate_projection_image
```

撮影機能を使う場合は `uv sync --extra sample` を実行する。
```

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "test: パッケージ境界テストを追加しREADMEを新レイアウトへ更新

core が Web層を import しないことの静的スキャンと、
カメラSDK・FastAPIを遮断した subprocess での import 確認。"
```

---

## Self-Review

**仕様書との対応:**

| 仕様書の節 | 対応タスク |
| --- | --- |
| §3.1 パッケージ構成（core側） | Task 1 |
| §3.2 既存コードの移行 | Task 2〜7 |
| §3.2 pipeline.py / geometry/warp.py の抽出 | Task 8, 9 |
| §3.3 既存テストへの影響 | 全タスクの Step 5（70テスト緑の維持） |
| §8 線形化レジストリ | Task 9 Step 3 |
| §11 境界テスト | Task 10 |

**本計画の対象外（後続の計画Bで扱う）:** §4 バックエンドAPI、§5 ビューレジストリ、
§6 数値読み取り、§7 表示と再計算の分離、§9 行列読み込み、§10 投影画像生成のAPI化、
§11 のゴールデンテストとビューテスト。

**型の一貫性:** Task 8 が `warp_color_mixing_matrices_to_projector` を公開し、
Task 9 の `generate_projection_image` が第2引数 `warped_matrix` として受け取る。
Task 9 の `LINEARIZERS` のキーは仕様書 §8 の `none` / `gamma` / `inverse_lut` と一致する。
Task 10 の import 検査対象モジュール名は Task 1〜9 で作ったものと一致する。

**リスク:** Task 2 の `_PROJECT_ROOT` 階層変更を誤ると config が既定値にフォールバックし、
テストは通るのに実行時の挙動だけが変わる。Task 2 Step 5 で
`get_config().projector.active_width == 722` を確認すること。
