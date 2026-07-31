# Webビューア縦スライス 実装計画（計画B1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `packages/web`（配布名 `yoshidacomp-web`）を作り、プロジェクタ座標の色変換行列 `.npy` を読み込んで12係数タイルを WebGL2 で表示し、ホバーで値を読めるところまでを通す。

**Architecture:** FastAPI がコア（`yoshidacomp`）だけに依存し、検証済みの float32 をバイナリで返す。フロントはビルド工程なしの ES module で、WebGL2 のフラグメントシェーダが表示トーンを担う。参照実装は `C:\py_scripts\hyperspectral-color\packages\web`。

**Tech Stack:** Python 3.13, FastAPI, uvicorn, WebGL2, 素の ES module（Node/Vite 不要）

## Global Constraints

- `requires-python = ">=3.13"`
- **依存方向は `web → core` の一方向。** `yoshidacomp` は `yoshidacomp_web` を import しない。`yoshidacomp_web` は `fastapi` / `uvicorn` / `yoshidacomp` にのみ依存し、`matplotlib` も `PyQt` も引かない
- **画素データは JSON に載せない。** `application/octet-stream` の float32 リトルエンディアンで送り、形状は `X-Height` / `X-Width` / `X-Channels` ヘッダで伝える
- `/api/*` を先に登録し、`StaticFiles(html=True)` を**最後に** `/` へマウントする。同一オリジンなので CORS は不要で `X-*` ヘッダも読める
- **表示パラメータはサーバ往復なし。** 露出・γ・カラーマップ・スケール・pan/zoom・ディザは GL の uniform 更新のみ。再フェッチするのはビュー切替とビューのパラメータ変更だけ
- コミットメッセージは日本語。コード・コメント・docstring は英日併記スタイル
- 各タスク完了時に `.venv/Scripts/python.exe -m pytest -q` が緑（開始時点は 101 passed）
- `uv sync` は必ず `--extra sample` を付ける。素の `uv sync` は edsdk と rawpy をアンインストールし、手動配置した DLL ごと消す

## 本計画の範囲

**入る:** パッケージ骨組み、行列の読み込みとメタ取得、ビューレジストリと `coef_tiles` 1つ、WebGL2 フロントの縦スライス、ホバーの値読み取り、境界テスト、実データでの性能確認。

**入らない（計画B2）:** 残り3ビュー（`surface` / `compensation` / `gamut`）、クリックによる 4×3 数値の固定表示、ターゲット画像入力、線形化と LUT カーブ、投影画像生成、カメラ座標行列の P2C ワープ（バックグラウンドジョブ）。

**カメラ座標行列は本計画では拒否する。** メタで `space: "camera"` を返し、フロントは「計画B2でワープ対応」と表示する。プロジェクタ座標の 100 MB 行列（`data/linear_pipeline_1080/matrices_linear_*.npy`）で縦スライスを通し、1.56 GB のワープは性能を確かめてから着手する。

---

### Task 1: web パッケージの骨組み

**Files:**
- Create: `packages/web/pyproject.toml`
- Create: `packages/web/yoshidacomp_web/__init__.py`
- Create: `packages/web/yoshidacomp_web/app.py`
- Create: `packages/web/yoshidacomp_web/__main__.py`
- Create: `packages/web/yoshidacomp_web/static/index.html`
- Modify: `pyproject.toml`
- Test: `tests/test_web_app.py`

**Interfaces:**
- Consumes: `yoshidacomp` 0.1.0
- Produces: `yoshidacomp_web.app.create_app() -> FastAPI`、`GET /api/health -> {"status": "ok"}`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_app.py`:

```python
"""Web layer smoke tests (no browser, no hardware)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_static_index_is_served_at_root() -> None:
    """`/` は index.html を返し、/api/* より後にマウントされていること。"""
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_app.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp_web'`

- [ ] **Step 3: パッケージを作る**

`packages/web/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "yoshidacomp-web"
version = "0.1.0"
description = "Web viewer for projector-camera color mixing matrices"
requires-python = ">=3.13"
dependencies = [
    "yoshidacomp",
    "fastapi>=0.115",
    "uvicorn>=0.34",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["yoshidacomp_web*"]

[tool.setuptools.package-data]
yoshidacomp_web = ["static/*", "static/js/*"]
```

`packages/web/yoshidacomp_web/__init__.py`:

```python
"""Web viewer for projector-camera color mixing matrices.

色変換行列のWebビューア。FastAPI が検証済みの float32 をバイナリで返し、
ブラウザ側の WebGL2 が表示トーンのみを担う。
"""

__version__ = "0.1.0"
```

`packages/web/yoshidacomp_web/app.py`:

```python
"""FastAPI application factory.

API は /api 配下、WebGL2 フロントエンドは / にマウントする。同一オリジンで
配信するため CORS は不要で、バイナリ応答の ``X-*`` ヘッダをブラウザから直接読める。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Build the FastAPI app with all routers and the static mount."""
    app = FastAPI(title="Yoshida Compensation Matrix Viewer", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Mount the frontend last so /api/* wins. html=True serves index.html at /.
    # フロントは最後にマウントする。先にすると /api/* が奪われる。
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
```

`packages/web/yoshidacomp_web/__main__.py`:

```python
"""Entry point: ``python -m yoshidacomp_web``."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Serve the viewer on localhost:8000."""
    uvicorn.run("yoshidacomp_web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
```

`packages/web/yoshidacomp_web/static/index.html` — この時点では最小の骨組みでよい:

```html
<!doctype html>
<meta charset="utf-8">
<title>色変換行列ビューア</title>
<h1>色変換行列ビューア</h1>
<p>準備中</p>
```

- [ ] **Step 4: workspace へ追加**

ルート `pyproject.toml` の `[project.optional-dependencies]` に追加する。
`[tool.uv.workspace] members = ["packages/*"]` は既にあるので変更不要:

```toml
[project.optional-dependencies]
sample = [
    "yoshidacomp[camera]",
    # Plotting for validate_inverse_lut.py (linearity check plots).
    "matplotlib",
]
web = [
    "yoshidacomp-web",
]
```

`[tool.uv.sources]` に `yoshidacomp-web = { workspace = true }` を追加する。

- [ ] **Step 5: 同期してテストが通ることを確認**

Run: `uv sync --extra sample --extra web`
Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 103 passed（101 + 新規2）

同期後、edsdk と rawpy と DLL が残っていることを確認する:

```
.venv/Scripts/python.exe -c "import importlib.util as u; [print(m, 'OK' if u.find_spec(m) else 'MISSING') for m in ['edsdk','rawpy','fastapi','uvicorn']]"
ls .venv/Lib/site-packages/edsdk/*.dll
```

消えていたら `cp C:/py_scripts/edsdk-python/dependencies/EDSDK_64/Dll/*.dll .venv/Lib/site-packages/edsdk/` で戻す。

- [ ] **Step 6: コミット**

```bash
git add pyproject.toml uv.lock packages/web tests/test_web_app.py
git commit -m "build: Web層パッケージ yoshidacomp-web を追加

FastAPI の骨組みと静的配信のマウント順序だけを置く。
/api/* を先に登録し StaticFiles を最後にマウントする。"
```

---

### Task 2: 行列の読み込みとメタ取得

**Files:**
- Create: `packages/web/yoshidacomp_web/store.py`
- Create: `packages/web/yoshidacomp_web/matrix.py`
- Create: `packages/web/yoshidacomp_web/models.py`
- Create: `packages/web/yoshidacomp_web/deps.py`
- Create: `packages/web/yoshidacomp_web/routers/__init__.py`
- Create: `packages/web/yoshidacomp_web/routers/session.py`
- Modify: `packages/web/yoshidacomp_web/app.py`
- Test: `tests/test_web_matrix.py`

**Interfaces:**
- Consumes: Task 1 の `create_app`
- Produces:
  - `matrix.MatrixContext` — `array: np.ndarray (H,W,4,3) float32`, `space: Literal["projector","camera"]`, `path: Path`, `active_region: tuple[int,int,int,int] | None`
  - `matrix.load_matrix(path, projector) -> MatrixContext`
  - `store.MatrixStore` — `put(ctx) -> str`, `get(id) -> MatrixContext`, `clear()`、モジュール singleton `store`
  - `POST /api/session/matrix` body `{"path": str}` → `{"id": str, "meta": {...}}`
  - `GET /api/matrix/{id}/meta` → `{"id","shape","space","dtype","path","projector",...}`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_matrix.py`:

```python
"""Matrix loading and metadata for the web layer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402


def _write_matrix(path, height: int, width: int) -> None:
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    for channel in range(3):
        matrices[:, :, channel, channel] = 1.0
    np.save(path, matrices)


def test_projector_space_matrix_is_accepted(tmp_path) -> None:
    store.clear()
    path = tmp_path / "projector.npy"
    _write_matrix(path, 1080, 1920)

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["space"] == "projector"
    assert body["meta"]["shape"] == [1080, 1920, 4, 3]

    meta = client.get(f"/api/matrix/{body['id']}/meta")
    assert meta.status_code == 200
    assert meta.json()["space"] == "projector"


def test_camera_space_matrix_is_reported_not_rejected(tmp_path) -> None:
    """カメラ座標も読み込めるが space="camera" と申告する（ワープは計画B2）。"""
    store.clear()
    path = tmp_path / "camera.npy"
    _write_matrix(path, 60, 80)

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200
    assert response.json()["meta"]["space"] == "camera"


def test_wrong_shape_is_rejected(tmp_path) -> None:
    store.clear()
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((4, 4, 3), dtype=np.float32))

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 400
    assert "(H, W, 4, 3)" in response.json()["detail"]


def test_missing_file_is_404(tmp_path) -> None:
    store.clear()
    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(tmp_path / "nope.npy")})
    assert response.status_code == 404


def test_unknown_id_is_404() -> None:
    store.clear()
    client = TestClient(create_app())
    assert client.get("/api/matrix/nosuch/meta").status_code == 404
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_matrix.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp_web.store'`

- [ ] **Step 3: `store.py` を書く**

`hyperspectral-color` の `store.py` と同じ形にする。FastAPI 非依存でスレッドセーフ:

```python
"""In-memory matrix store (one process, thread-safe).

読み込み済みの行列をサーバのRAMに置き、不透明なIDで参照する。線を渡るのは
2次元のビュー結果だけで、(H, W, 4, 3) の行列そのものは決して送らない。
FastAPI に依存させないので単体でテストできる。
"""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matrix import MatrixContext


class MatrixStore:
    """Thread-safe id -> MatrixContext map."""

    def __init__(self) -> None:
        self._items: dict[str, MatrixContext] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def put(self, context: MatrixContext) -> str:
        """Store a matrix context and return its new id."""
        with self._lock:
            matrix_id = f"matrix{next(self._counter)}"
            self._items[matrix_id] = context
            return matrix_id

    def get(self, matrix_id: str) -> MatrixContext:
        """Return the context for ``matrix_id`` or raise ``KeyError``."""
        with self._lock:
            context = self._items.get(matrix_id)
        if context is None:
            raise KeyError(matrix_id)
        return context

    def clear(self) -> None:
        """Drop everything (used by tests)."""
        with self._lock:
            self._items.clear()


store = MatrixStore()
```

- [ ] **Step 4: `matrix.py` を書く**

```python
"""Color mixing matrix loading and coordinate-space classification.

色変換行列の読み込みと座標空間の判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from yoshidacomp.compensation.projection_region import resolve_active_region
from yoshidacomp.config import ProjectorConfig

MatrixSpace = Literal["projector", "camera"]


@dataclass(frozen=True)
class MatrixContext:
    """A loaded matrix plus what the viewer needs to interpret it.

    Attributes:
        array: Per-pixel affine matrices, shape ``(H, W, 4, 3)``, float32.
            画素ごとのアフィン行列。
        space: Which coordinate space ``array`` lives in.
            行列が住んでいる座標空間。
        path: Where it was loaded from.
        active_region: ``(x, y, w, h)`` of the valid projection area, or None
            when the matrix is not in projector space.
            有効投影領域。プロジェクタ座標でない場合は None。
    """

    array: np.ndarray
    space: MatrixSpace
    path: Path
    active_region: tuple[int, int, int, int] | None


def classify_space(height: int, width: int, projector: ProjectorConfig) -> MatrixSpace:
    """Return "projector" when the matrix matches the configured frame.

    行列の縦横がプロジェクタのフレームと一致すればプロジェクタ座標とみなす。
    一致しなければカメラ座標として扱う。カメラ解像度は機種依存で固定値を
    持たないため、こちらを既定にする。
    """
    if width == projector.width and height == projector.height:
        return "projector"
    return "camera"


def load_matrix(path: Path, projector: ProjectorConfig) -> MatrixContext:
    """Load a ``(H, W, 4, 3)`` float matrix and classify its space.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: the array is not shaped ``(H, W, 4, 3)``.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    array = np.load(path, mmap_mode="r")
    if array.ndim != 4 or array.shape[2] != 4 or array.shape[3] != 3:
        raise ValueError(
            f"expected a matrix shaped (H, W, 4, 3), got {tuple(array.shape)}"
        )

    height, width = int(array.shape[0]), int(array.shape[1])
    space = classify_space(height, width, projector)
    region = resolve_active_region(projector) if space == "projector" else None
    return MatrixContext(array=array, space=space, path=path, active_region=region)
```

`mmap_mode="r"` で開く。1.56 GB のカメラ座標行列を丸ごと RAM へ載せないため。

- [ ] **Step 5: `models.py` / `deps.py` / `routers/session.py` を書く**

`models.py`:

```python
"""Pydantic request bodies."""

from __future__ import annotations

from pydantic import BaseModel


class LoadMatrixRequest(BaseModel):
    """Body of ``POST /api/session/matrix``."""

    path: str
```

`deps.py`:

```python
"""Shared FastAPI dependencies / helpers."""

from __future__ import annotations

from fastapi import HTTPException

from .matrix import MatrixContext
from .store import store


def require_matrix(matrix_id: str) -> MatrixContext:
    """Return the stored matrix context or raise 404."""
    try:
        return store.get(matrix_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"matrix not found: {matrix_id}"
        ) from exc
```

`routers/session.py` は `POST /api/session/matrix` と `GET /api/matrix/{id}/meta` を実装する。
`FileNotFoundError` → 404、`ValueError` → 400 に写す。メタは次を返す:

```python
{
    "id": matrix_id,
    "shape": [h, w, 4, 3],
    "dtype": str(context.array.dtype),
    "space": context.space,
    "path": str(context.path),
    "active_region": list(context.active_region) if context.active_region else None,
    "projector": {"width": projector.width, "height": projector.height},
}
```

`app.py` に `app.include_router(session.router, prefix="/api", tags=["session"])` を
**StaticFiles のマウントより前に**追加する。

- [ ] **Step 6: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 108 passed（103 + 新規5）

- [ ] **Step 7: コミット**

```bash
git add -A
git commit -m "feat(web): 行列の読み込みとメタ取得を追加

mmap で開いて座標空間を形状から判定する。1.56GBのカメラ座標行列を
RAMへ丸ごと載せないため mmap_mode='r' を使う。"
```

---

### Task 3: ビューレジストリと coef_tiles

**Files:**
- Create: `packages/web/yoshidacomp_web/views/__init__.py`
- Create: `packages/web/yoshidacomp_web/views/base.py`
- Create: `packages/web/yoshidacomp_web/views/coef_tiles.py`
- Create: `packages/web/yoshidacomp_web/routers/view.py`
- Modify: `packages/web/yoshidacomp_web/app.py`
- Test: `tests/test_web_views.py`

**Interfaces:**
- Consumes: Task 2 の `MatrixContext` と `require_matrix`
- Produces:
  - `views.base.ViewSpec` — `key: str`, `label: str`, `kind: Literal["scalar","rgb","categorical"]`, `params: list[ParamSpec]`, `render: Callable[[MatrixContext, dict], ViewResult]`
  - `views.base.ViewResult` — `data: np.ndarray (H, W, C) float32`, `channels: int`, `vmin: float`, `vmax: float`, `labels: list[str]`
  - `views.base.REGISTRY: dict[str, ViewSpec]` と `register(spec)`
  - `GET /api/views` → ビュー一覧とパラメータ定義
  - `GET /api/matrix/{id}/view/{key}` → float32 バイナリ + `X-Height` / `X-Width` / `X-Channels` / `X-Vmin` / `X-Vmax` / `X-Kind` / `X-Labels-B64`（UTF-8 JSON配列のbase64）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_views.py`:

```python
"""View registry and the coefficient-tile view."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402


def _load(client, tmp_path, height=1080, width=1920):
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)[None, :]
    # 空間的に変化させる。一様だと座標の取り違えを検出できない。
    matrices[:, :, 0, 0] = xs
    matrices[:, :, 3, 1] = -1.0
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    response = client.post("/api/session/matrix", json={"path": str(path)})
    return response.json()["id"]


def test_views_endpoint_lists_coef_tiles() -> None:
    store.clear()
    client = TestClient(create_app())
    body = client.get("/api/views").json()
    keys = {view["key"] for view in body["views"]}
    assert "coef_tiles" in keys
    spec = next(v for v in body["views"] if v["key"] == "coef_tiles")
    assert spec["kind"] == "scalar"
    assert spec["channels"] == 12


def test_coef_tiles_returns_float32_binary(tmp_path) -> None:
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Height"] == "6"
    assert response.headers["X-Width"] == "8"
    assert response.headers["X-Channels"] == "12"
    assert response.headers["X-Kind"] == "scalar"

    array = np.frombuffer(response.content, dtype="<f4").reshape(6, 8, 12)
    # チャンネル順は (i, j) の行優先: 0=in_R->out_R, 1=in_R->out_G, ... 11=bias->out_B
    assert np.allclose(array[:, :, 0], np.arange(8, dtype=np.float32)[None, :])
    assert np.allclose(array[:, :, 10], -1.0)


def test_unknown_view_is_404(tmp_path) -> None:
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)
    assert client.get(f"/api/matrix/{matrix_id}/view/nosuch").status_code == 404
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_views.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp_web.views'`

- [ ] **Step 3: `views/base.py` を書く**

```python
"""View registry.

拡張の中心。ビューを1つ登録すればGUIへ現れ、``kind`` が標準的なものであれば
JavaScript の変更を要しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from ..matrix import MatrixContext

ViewKind = Literal["scalar", "rgb", "categorical"]


@dataclass(frozen=True)
class ParamSpec:
    """One GUI control, rendered automatically by the frontend.

    Attributes:
        name: Parameter key passed back as a query string.
        label: Human-readable label.
        type: "number" | "select" | "rgb".
        default: Default value.
        choices: Options when ``type == "select"``.
    """

    name: str
    label: str
    type: Literal["number", "select", "rgb"]
    default: object
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ViewResult:
    """What a view renders.

    Attributes:
        data: ``(H, W, C)`` float32. The wire format is exactly this, little-endian.
        vmin, vmax: Suggested display range (the frontend may override).
        labels: One label per channel, shown as tile captions.
    """

    data: np.ndarray
    vmin: float
    vmax: float
    labels: list[str]


@dataclass(frozen=True)
class ViewSpec:
    """A registered view."""

    key: str
    label: str
    kind: ViewKind
    channels: int
    params: list[ParamSpec]
    render: Callable[[MatrixContext, dict], ViewResult]


REGISTRY: dict[str, ViewSpec] = {}


def register(spec: ViewSpec) -> ViewSpec:
    """Add a view to the registry. Raises on a duplicate key."""
    if spec.key in REGISTRY:
        raise ValueError(f"duplicate view key: {spec.key}")
    REGISTRY[spec.key] = spec
    return spec
```

- [ ] **Step 4: `views/coef_tiles.py` を書く**

```python
"""The 12 affine coefficients laid out as the matrix itself.

行列の形（4行×3列）そのままに係数マップを並べるビュー。
"""

from __future__ import annotations

import numpy as np

from ..matrix import MatrixContext
from .base import ParamSpec, ViewResult, ViewSpec, register

_INPUT_LABELS = ("in_R", "in_G", "in_B", "bias")
_OUTPUT_LABELS = ("out_R", "out_G", "out_B")

# Channel order is row-major over (input, output): c = i * 3 + j.
# チャンネル順は (入力, 出力) の行優先。
LABELS = [f"{i} -> {j}" for i in _INPUT_LABELS for j in _OUTPUT_LABELS]


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Flatten (H, W, 4, 3) into (H, W, 12) and pick a symmetric range."""
    array = np.asarray(context.array, dtype=np.float32)
    height, width = array.shape[0], array.shape[1]
    data = array.reshape(height, width, 12).copy()

    if context.active_region is not None:
        x, y, w, h = context.active_region
        mask = np.zeros((height, width), dtype=bool)
        mask[y : y + h, x : x + w] = True
        # Outside the active region the fit is physically meaningless; NaN so
        # the shader paints it as a mask colour instead of a plausible value.
        # 有効領域外のfitは物理的に無意味なので NaN にしてシェーダでマスク色にする。
        data[~mask] = np.nan

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmax = 1.0
    elif params.get("normalize", "shared") == "per_tile":
        # Per-tile normalisation is applied client-side; report the global
        # range so the caller still has a sane fallback.
        vmax = float(np.percentile(np.abs(finite), 99))
    else:
        vmax = float(np.percentile(np.abs(finite), 99))
    vmax = max(vmax, 1e-6)

    # Symmetric range keeps zero at the white point of the diverging map.
    # 対称にすることで発散カラーマップの0が白に保たれる。
    return ViewResult(data=data, vmin=-vmax, vmax=vmax, labels=LABELS)


SPEC = register(
    ViewSpec(
        key="coef_tiles",
        label="12係数タイル",
        kind="scalar",
        channels=12,
        params=[
            ParamSpec(
                name="normalize",
                label="正規化",
                type="select",
                default="shared",
                choices=["shared", "per_tile"],
            )
        ],
        render=_render,
    )
)
```

`views/__init__.py` は登録を発火させるために import する:

```python
"""View registry and the built-in views."""

from __future__ import annotations

from . import coef_tiles  # noqa: F401  registers "coef_tiles"
from .base import REGISTRY, ParamSpec, ViewResult, ViewSpec, register

__all__ = ["REGISTRY", "ParamSpec", "ViewResult", "ViewSpec", "register"]
```

- [ ] **Step 5: `routers/view.py` を書く**

```python
"""View endpoints: the registry listing and the binary view payload."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response

from ..deps import require_matrix
from ..views import REGISTRY

router = APIRouter()


@router.get("/views")
def list_views() -> dict[str, Any]:
    """Return every registered view and its parameter declarations.

    フロントはこれを読んでコントロールを自動生成する。
    """
    return {
        "views": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "channels": spec.channels,
                "params": [
                    {
                        "name": p.name,
                        "label": p.label,
                        "type": p.type,
                        "default": p.default,
                        "choices": p.choices,
                    }
                    for p in spec.params
                ],
            }
            for spec in REGISTRY.values()
        ]
    }


@router.get("/matrix/{matrix_id}/view/{key}")
def render_view(matrix_id: str, key: str, request: Request) -> Response:
    """Render a view as a raw float32 binary body.

    Shape and display range travel in ``X-*`` headers, readable because the
    frontend is served same-origin. Pixel data is never JSON-encoded.
    形状と表示レンジは ``X-*`` ヘッダで返す。画素データはJSONに載せない。
    """
    context = require_matrix(matrix_id)
    spec = REGISTRY.get(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown view: {key}")

    params: dict[str, Any] = {p.name: p.default for p in spec.params}
    for p in spec.params:
        raw = request.query_params.get(p.name)
        if raw is None:
            continue
        if p.type == "number":
            params[p.name] = float(raw)
        elif p.type == "select":
            if p.choices and raw not in p.choices:
                raise HTTPException(
                    status_code=400,
                    detail=f"{p.name} must be one of {p.choices}, got {raw!r}",
                )
            params[p.name] = raw
        else:
            params[p.name] = raw

    result = spec.render(context, params)
    body = np.ascontiguousarray(result.data, dtype="<f4").tobytes()
    height, width = result.data.shape[0], result.data.shape[1]
    channels = result.data.shape[2] if result.data.ndim == 3 else 1
    headers = {
        "X-Height": str(height),
        "X-Width": str(width),
        "X-Channels": str(channels),
        "X-Kind": spec.kind,
        "X-Vmin": repr(float(result.vmin)),
        "X-Vmax": repr(float(result.vmax)),
        "X-Labels-B64": base64.b64encode(json.dumps(result.labels).encode("utf-8")).decode("ascii"),
    }
    return Response(
        content=body, media_type="application/octet-stream", headers=headers
    )
```

`app.py` に `app.include_router(view.router, prefix="/api", tags=["view"])` を
StaticFiles のマウントより前に追加する。

- [ ] **Step 6: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 111 passed（108 + 新規3）

- [ ] **Step 7: コミット**

```bash
git add -A
git commit -m "feat(web): ビューレジストリと12係数タイルを追加

kind によってフロントの描画方法が決まるため、標準的な種類のビューは
Python側の登録だけでGUIへ現れる。"
```

---

### Task 4: WebGL2 フロントエンドの縦スライス

**Files:**
- Create: `packages/web/yoshidacomp_web/static/js/api.js`
- Create: `packages/web/yoshidacomp_web/static/js/shaders.js`
- Create: `packages/web/yoshidacomp_web/static/js/glview.js`
- Create: `packages/web/yoshidacomp_web/static/js/main.js`
- Create: `packages/web/yoshidacomp_web/static/style.css`
- Modify: `packages/web/yoshidacomp_web/static/index.html`
- Test: 手動スモーク（下記 Step 5）

**Interfaces:**
- Consumes: Task 3 の `/api/views` と `/api/matrix/{id}/view/{key}`
- Produces: `GLView` クラス（`setImage` / `setParams` / `screenToImage` / `imageToScreen` / `onPixel` / `onHover` / `onViewChange`）

- [ ] **Step 1: `glview.js` を参照実装から移植する**

`C:\py_scripts\hyperspectral-color\packages\web\hyperspectral_color_web\static\js\glview.js`（240行）を
コピーし、次の点だけ変える。それ以外はそのまま使う — 座標変換・入力処理・
DPR 対応・float テクスチャの扱いはすべて転用できる。

| 変更点 | 内容 |
| --- | --- |
| ROI 関連を削除 | `onRoiPreview` / `onRoiCommit` / `roiing` / `_bounds` / Shift+ドラッグ分岐。計画B1では使わない |
| 単チャンネル取り出し | `setImage` に `channel` を渡し、`(H, W, 12)` から1枚を切り出して `R32F` で載せる。12枚同時にGPUへ置かず、タイルごとに `GLView` を持つ |
| uniform の入替 | `u_bandLo` / `u_bandHi` を捨て、`u_vmin` / `u_vmax` / `u_colormap` を足す |
| NaN マスク色 | `u_nanColor` はそのまま使う。有効領域外を示す |

**バッキングストアの DPR 対応（`resize()` の
`clientWidth * devicePixelRatio`）は必ず残すこと。** これが参照実装が
PyQt6 から移った理由そのものである。

- [ ] **Step 2: `shaders.js` を書く**

頂点シェーダは参照実装のまま。フラグメントシェーダの本体は次のとおり。
`R32F` テクスチャの `.r` を読み、NaN をマスク色に、それ以外をカラーマップして
sRGB OETF とディザを掛ける。

```glsl
#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform float u_vmin;
uniform float u_vmax;
uniform int   u_colormap;   // 0 = diverging, 1 = grayscale
uniform float u_exposure;   // EV
uniform float u_gamma;
uniform float u_dither;
uniform vec3  u_nanColor;

// Blue -> white -> red. Zero sits at white when the range is symmetric.
// 青→白→赤。レンジが対称なら0が白に来る。
vec3 diverging(float t) {          // t in [-1, 1]
    float a = clamp(abs(t), 0.0, 1.0);
    vec3 cold = vec3(0.23, 0.30, 0.75);
    vec3 warm = vec3(0.71, 0.02, 0.15);
    vec3 tip  = (t < 0.0) ? cold : warm;
    return mix(vec3(1.0), tip, a);
}

// Standard sRGB OETF. Applied once, immediately before quantisation.
// 標準sRGB OETF。量子化の直前に一度だけ掛ける。
vec3 srgbEncode(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float v = texture(u_tex, v_uv).r;
    if (isnan(v)) { fragColor = vec4(u_nanColor, 1.0); return; }

    float span = max(u_vmax - u_vmin, 1e-9);
    float t = (u_colormap == 0)
        ? clamp((2.0 * (v - u_vmin) / span) - 1.0, -1.0, 1.0)   // [-1, 1]
        : clamp((v - u_vmin) / span, 0.0, 1.0);                 // [0, 1]

    vec3 rgb = (u_colormap == 0) ? diverging(t) : vec3(t);
    rgb *= exp2(u_exposure);
    rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)));
    rgb = srgbEncode(rgb);

    // Dither before the driver quantises to 8 bit, so gradients do not band.
    // ドライバが8bitへ量子化する前にディザを入れ、階調のバンディングを防ぐ。
    rgb += (hash12(gl_FragCoord.xy) - 0.5) * (u_dither / 255.0);
    fragColor = vec4(rgb, 1.0);
}
```

- [ ] **Step 3: `api.js` を書く**

参照実装の `api.js`（130行）と同じ形。`fetch` の薄いラッパで、バイナリ応答は
`ArrayBuffer` と `X-*` ヘッダを組にして返す:

```javascript
export async function fetchView(matrixId, key, params) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`/api/matrix/${matrixId}/view/${key}?${qs}`);
  if (!res.ok) throw new Error(await res.text());
  const buf = await res.arrayBuffer();
  return {
    data: new Float32Array(buf),
    height: Number(res.headers.get("X-Height")),
    width: Number(res.headers.get("X-Width")),
    channels: Number(res.headers.get("X-Channels")),
    kind: res.headers.get("X-Kind"),
    vmin: Number(res.headers.get("X-Vmin")),
    vmax: Number(res.headers.get("X-Vmax")),
    labels: JSON.parse(atob(res.headers.get("X-Labels-B64") || "W10=")),
  };
}
```

- [ ] **Step 4: `main.js` と `index.html` を書く**

画面構成:

- 上部にパス入力欄と「読み込み」ボタン、読み込んだ行列のメタ表示
- `space === "camera"` なら「カメラ座標の行列です。プロジェクタ座標へのワープは
  未実装のため、係数はカメラ画素のまま表示しています」と注記を出す
- 本体は 4行×3列のタイルグリッド。各タイルが `<canvas>` を1つ持ち、
  行ラベル（in_R / in_G / in_B / bias）と列ラベル（out_R / out_G / out_B）を付ける
- 右側に露出・γ・カラーマップ・正規化のコントロール
- 下部にホバー読み取り欄

**表示と再計算の分離を守ること。** 露出・γ・カラーマップは全 `GLView` の
`setParams` を呼ぶだけでサーバへ行かない。正規化方法の変更だけが再フェッチを起こす。

pan/zoom は12タイル連動にする。1つの `GLView` の `onViewChange` で他11枚の
`view` を同期し `draw()` する。再帰しないようフラグでガードすること。

ホバーは `onHover` で全12チャンネルの値を読む。`fetchView` で受けた
`Float32Array` を CPU 側に保持しておき、`data[(row * width + col) * 12 + c]` で
引く。**サーバ往復は行わない。**

- [ ] **Step 5: 実機スモーク**

```
uv run --no-sync python -m yoshidacomp_web
```

ブラウザで `http://127.0.0.1:8000` を開き、次を確認する:

1. `data/linear_pipeline_1080/matrices_linear_gamma.npy` を読み込み、12タイルが出る
2. 対角（in_R→out_R など）が非対角より濃い
3. pan/zoom が12枚連動し、60fps で動く
4. 露出とγを動かしてもネットワークタブに新しいリクエストが出ない
5. ホバーで12個の値が出て、タイルの見た目と符号が一致する
6. Windows の表示スケーリング **150%** で画像と文字が鮮明
7. 有効領域の外がマスク色で塗られる

**性能の確認が本タスクの主目的である。** 1920×1080×12 の float32 は約 100 MB。
これが実用的な速度で転送・描画できないなら、計画B2 のカメラ座標ワープへ進む前に
設計を見直す。転送時間とタイル描画時間を報告すること。

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "feat(web): WebGL2フロントの縦スライスを追加

glview.js は参照実装から移植し、ROIを外してスカラー単チャンネル表示と
発散カラーマップに置き換えた。DPR対応のバッキングストアは維持している。
表示パラメータはuniform更新のみでサーバ往復を起こさない。"
```

---

### Task 5: ゴールデンテストと境界テスト

**Files:**
- Create: `tests/test_web_golden.py`
- Modify: `tests/test_package_boundaries.py`
- Test: 上記2ファイル

**Interfaces:**
- Consumes: Task 3 のバイナリ応答
- Produces: なし（保護のみ）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_golden.py` — バックエンドの float 出力に、シェーダ相当の処理を
numpy で適用した結果が、期待値と画素単位で一致することを確認する。FastAPI の
`TestClient` だけで完結し、ブラウザを必要としない。

具体的には、既知の係数を持つ小さな行列を作り、`/view/coef_tiles` の応答を
`np.frombuffer` で読み、発散カラーマップと sRGB OETF を numpy で再現して、
手計算した RGB と一致することを見る。ディザは検証対象外（乱数のため）。

`tests/test_package_boundaries.py` に追記:

```python
def test_core_does_not_import_web() -> None:
    """既存のテストを web パッケージにも広げる。"""
    offenders = []
    for path in _CORE_DIR.rglob("*.py"):
        if _IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"core must not import the web layer: {offenders}"


def test_web_does_not_import_matplotlib_or_qt() -> None:
    """Web層は描画ライブラリを引かない（描画はブラウザ側）。"""
    import re

    web_dir = _REPO_ROOT / "packages" / "web" / "yoshidacomp_web"
    pattern = re.compile(r"^\s*(?:from|import)\s+(matplotlib|PyQt6|PyQt5)\b", re.MULTILINE)
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in web_dir.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"web must not import a plotting stack: {offenders}"
```

既存の `_IMPORT_RE` は `yoshidacomp_web` を対象にしているので、Task 1 で
パッケージが実在するようになった今、初めて意味を持つ。

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_golden.py -q`
Expected: FAIL — ゴールデンテストが未実装

- [ ] **Step 3: 実装して通す**

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 114 passed（111 + 新規3）

- [ ] **Step 5: README を更新**

`README.md` に Web ビューアの起動方法を追記する:

```markdown
## 色変換行列ビューア

```bash
uv sync --extra sample --extra web
uv run --no-sync python -m yoshidacomp_web
```

`http://127.0.0.1:8000` を開き、行列の `.npy` パスを入力する。
```

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "test(web): ゴールデンテストと境界テストを追加

バックエンドのfloat出力にシェーダ相当の処理をnumpyで適用した結果が
手計算と一致することを確認する。ブラウザ不要で通常のテストゲートに乗る。"
```

---

## Self-Review

**仕様書との対応:**

| 仕様書の節 | 対応タスク |
| --- | --- |
| §3.1 パッケージ構成（web側） | Task 1 |
| §4 バックエンドAPI（`session/matrix`, `matrix/{id}/meta`, `views`, `view/{key}`） | Task 2, 3 |
| §5 ビューレジストリ、§5.1 の `coef_tiles` | Task 3 |
| §6 数値の読み取り（ホバーのみ） | Task 4 |
| §7 表示と再計算の分離 | Task 4 |
| §9 行列の読み込みと座標空間（判定のみ、ワープなし） | Task 2 |
| §11 ゴールデンテスト・境界テスト | Task 5 |

**本計画の対象外（計画B2）:** §5.1 の `surface` / `compensation` / `gamut`、
§6 のクリックによる 4×3 固定表示（`/api/matrix/{id}/probe`）、§8 の線形化と
LUTカーブ、§9 のカメラ座標ワープとジョブ、§10 の投影画像生成。

**型の一貫性:** Task 2 が `MatrixContext` を公開し、Task 3 の `ViewSpec.render` が
第1引数として受け取る。Task 3 の `ViewResult.data` は `(H, W, C) float32` で、
Task 4 の `fetchView` が `X-Channels` とともに読む。ヘッダ名は Task 3 と Task 4 で一致。

**リスク:** Task 4 Step 5 の性能確認が本計画最大の未知数である。100 MB の
float32 転送とタイル描画が実用速度に乗らない場合、ダウンサンプリングした
プレビューを返す設計へ切り替える必要がある。カメラ座標の 1.56 GB ワープ
（計画B2）へ進む前にここで判断する。
