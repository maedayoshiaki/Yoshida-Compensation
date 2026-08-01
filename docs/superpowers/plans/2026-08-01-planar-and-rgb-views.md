# planar転送形式と rgb ビュー 実装計画（計画B2）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ビューの転送形式を planar `(C, H, W)` へ変え、フロントエンドに `rgb` 種別の描画を実装し、行列から投影面の色を求める `surface` ビューを追加する。

**Architecture:** 既存の `web → core` 一方向依存とバイナリ転送規約はそのまま。変えるのは (1) 配列の並び順と1つのヘッダ、(2) シェーダのモード分岐と `GLView` のテクスチャ形式、(3) `views/` へのビュー1つの登録。

**Tech Stack:** Python 3.13, FastAPI, WebGL2, 素の ES module, pytest + node ハーネス

## Global Constraints

- 依存方向は `web → core` の一方向。`yoshidacomp_web` は `fastapi` / `uvicorn` / `numpy` / `pydantic` / `yoshidacomp` にのみ依存し、`matplotlib` も `PyQt` も引かない
- **`yoshidacomp.compensation.color_mixing_matrix` を web から import しないこと。** module scope で `torch` と `psutil` を引くため、サーバプロセスへ torch が載る
- 画素データは JSON に載せない。`application/octet-stream` の float32 リトルエンディアン
- 表示パラメータ（露出・γ・カラーマップ・ディザ・pan/zoom）はサーバ往復なし
- コミットメッセージは日本語。コード・コメント・docstring は英日併記スタイル
- 各タスク完了時に `.venv/Scripts/python.exe -m pytest -q` が緑（開始時点は 132 passed）
- `uv sync` は必ず `--extra sample --extra web`

## `main.js` の変異テストアンカー（触ると壊れる）

`tests/test_web_frontend_js.py::test_harness_catches_the_regressions_it_exists_for` は
以下の文字列を完全一致で探して差し替える。**表記を変えてはならない。**
変える必要が生じたら、テスト側の `_MUTATIONS` も同じコミットで更新すること。

1. `  if (isStale(req)) return;   // a newer request owns the screen now\n`
2. `    if (isStale(req)) return;\n    state.matrixId = res.id;`
3. `setLoadBusy(true)` と `setLoadBusy(false)`
4. `  if (kind !== SUPPORTED_KIND) {`
5. `    buildViewSelect(specs);\n`

本計画では 4 を意図的に変更する（Task 3）。その際 `_MUTATIONS` の
`unsupported_kind_renders_as_scalar` を同時に更新する。

---

### Task 1: 転送形式を planar `(C, H, W)` にする

**Files:**
- Modify: `packages/web/yoshidacomp_web/routers/view.py`
- Modify: `packages/web/yoshidacomp_web/static/js/api.js`
- Modify: `packages/web/yoshidacomp_web/static/js/glview.js`
- Modify: `packages/web/yoshidacomp_web/static/js/main.js`
- Modify: `tests/test_web_views.py`, `tests/test_web_golden.py`, `tests/js/frontend_harness.mjs`
- Test: 上記

**Interfaces:**
- Consumes: 既存の `ViewResult`（`data` は `(H, W, C)` のまま。Python 側の自然な形を保つ）
- Produces:
  - 新ヘッダ `X-Layout: planar`
  - 本体は `(C, H, W)` の float32 リトルエンディアン
  - `fetchView` の戻り値に `layout: string` が加わる
  - `GLView.setImage({data, width, height, channels, channel, layout})` — `layout === "planar"` のとき `data.subarray()` でゼロコピー

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_views.py` に追記:

```python
def test_view_body_is_planar_and_declares_its_layout(tmp_path, stand_in_projector_config) -> None:
    """本体は (C, H, W)。ヘッダ X-Layout がそれを申告する。

    インタリーブ (H, W, C) だとクライアントがチャンネルを取り出すのに
    ストライドコピーが要る。実測ではGPUアップロード時間の93%がこの
    デインターリーブだった。planar なら subarray でゼロコピーになる。
    """
    import numpy as np

    stand_in_projector_config(width=8, height=6)
    client = TestClient(create_app())

    matrices = np.zeros((6, 8, 4, 3), dtype=np.float32)
    xs = np.arange(8, dtype=np.float32)[None, :]
    ys = np.arange(6, dtype=np.float32)[:, None]
    matrices[:, :, 0, 0] = xs     # channel 0
    matrices[:, :, 3, 1] = ys     # channel 10
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    matrix_id = client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200
    assert response.headers["X-Layout"] == "planar"
    assert response.headers["X-Channels"] == "12"

    planar = np.frombuffer(response.content, dtype="<f4").reshape(12, 6, 8)
    np.testing.assert_allclose(planar[0], np.broadcast_to(xs, (6, 8)))
    np.testing.assert_allclose(planar[10], np.broadcast_to(ys, (6, 8)))
    # 転置していないことの確認: 幅8・高さ6の非正方なので取り違えれば形が合わない
    assert planar.shape == (12, 6, 8)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_views.py -q -k planar`
Expected: FAIL — `KeyError: 'x-layout'`

- [ ] **Step 3: サーバ側を planar にする**

`routers/view.py` の本体生成を差し替える:

```python
    # Serialise as planar (C, H, W). The client uploads one channel per texture,
    # so a planar body lets it take a zero-copy subarray instead of walking the
    # array with a stride. Measured: the strided de-interleave was 93% of the
    # client's "GPU upload" time for a 95 MB payload.
    # planar (C, H, W) で直列化する。クライアントはチャンネルごとにテクスチャを
    # 作るので、planar ならストライド走査ではなく subarray のゼロコピーで済む。
    # 実測では 95MB のペイロードでストライド走査がアップロード時間の93%を占めた。
    body = np.ascontiguousarray(np.moveaxis(result.data, 2, 0), dtype="<f4").tobytes()
```

`headers` に `"X-Layout": "planar"` を追加する。

**`np.moveaxis` は view を返すので、`ascontiguousarray` が実際のコピーを行う。**
`(H, W, C)` → `(C, H, W)` の並べ替えはサーバ側で一度だけ起きる。クライアント側で
12回のストライド走査をするより安い。

- [ ] **Step 4: クライアント側を追随させる**

`api.js` の `fetchView` の戻り値に追加:

```js
    layout: res.headers.get("X-Layout") || "interleaved",
```

`glview.js` の `setImage` のチャンネル抽出を差し替える。**既存のインタリーブ経路は
残すこと** — ハーネスのフィクスチャや将来の単一チャンネル応答が使う:

```js
    let plane;
    if (channels === 1) {
      plane = img.data;
    } else if (img.layout === "planar") {
      // Planar (C, H, W): the channel is already contiguous. subarray shares
      // the buffer, so this costs nothing.
      // planar (C, H, W) ならチャンネルは既に連続しており、subarray は
      // バッファを共有するのでコストがかからない。
      const n = width * height;
      plane = img.data.subarray(channel * n, (channel + 1) * n);
    } else {
      const n = width * height;
      plane = new Float32Array(n);
      for (let i = 0, k = channel; i < n; i++, k += channels) plane[i] = img.data[k];
    }
```

`main.js` の `showProbe` のインデックス計算を layout で分岐させる。
**この関数は変異テストアンカーではないので編集して安全。**

```js
  const planar = p.layout === "planar";
  const stride = p.width * p.height;
  for (let c = 0; c < probeCells.length && c < p.channels; c++) {
    const v = planar
      ? p.data[c * stride + hit.row * p.width + hit.col]
      : p.data[(hit.row * p.width + hit.col) * p.channels + c];
    ...
  }
```

`main.js` の `percentileAbsPerChannel(data, channels, pct)` も layout を要する。
シグネチャへ `layout` を足し、planar のときはチャンネル `c` の走査を
`data[c * stride + i]`（連続）にする。`computeRanges` の呼び出しも更新する。

`setImage` の呼び出しへ `layout: payload.layout` を渡す。

- [ ] **Step 5: ハーネスとゴールデンテストを追随させる**

`tests/js/frontend_harness.mjs` の偽サーバは `data[i] = ((i % channels) + 1) * tag`
でインタリーブを作っている。planar へ変え、`X-Layout: "planar"` を返すこと。
**`payloadTag()` が `state.payload.data[0]` を読んでいる点に注意** — planar では
`data[0]` はチャンネル0の先頭画素なので、チャンネル0に `1 * tag` が入る並びを保てば
既存のアサーションはそのまま通る。

`tests/test_web_golden.py` は `(H, W, C)` を前提に本体を読んでいる。planar として
読み直す。シェーダのミラーそのものは変わらない（シェーダは1チャンネルしか見ない）。

- [ ] **Step 6: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 133 passed（132 + 新規1）

- [ ] **Step 7: コミット**

```bash
git add packages/web tests
git commit -m "perf(web): ビューの転送形式を planar (C, H, W) にする

クライアントはチャンネルごとにテクスチャを作るため、planar なら
subarray のゼロコピーで済む。実測では 95MB のペイロードでストライド
走査がGPUアップロード時間の93%を占めていた。

X-Layout ヘッダで申告し、インタリーブ経路も残す。"
```

---

### Task 2: surface ビュー（投影面の色）

**Files:**
- Create: `packages/web/yoshidacomp_web/views/surface.py`
- Modify: `packages/web/yoshidacomp_web/views/__init__.py`
- Test: `tests/test_web_surface_view.py`

**Interfaces:**
- Consumes: Task 1 の planar 転送、`MatrixContext`
- Produces: `views.surface.SPEC`（`key="surface"`, `kind="rgb"`, `channels=3`,
  params `p_r` / `p_g` / `p_b`（number, 既定 1.0））

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_surface_view.py`:

```python
"""The forward-model view: what the camera sees for a given projector command."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402


def _load(client, tmp_path, matrices):
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    return client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]


def test_surface_inverts_a_known_affine(tmp_path, stand_in_projector_config) -> None:
    """M が [C,1]M = P を満たすとき、surface は C = (P - b)A^-1 を返す。

    A = 2I, b = (0.1, 0.2, 0.3) を置くと、P=(1,1,1) に対する答えは
    ((1-0.1)/2, (1-0.2)/2, (1-0.3)/2) = (0.45, 0.40, 0.35) と手計算できる。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 200
    assert response.headers["X-Kind"] == "rgb"
    assert response.headers["X-Channels"] == "3"
    assert response.headers["X-Layout"] == "planar"

    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    np.testing.assert_allclose(planar[0], 0.45, atol=1e-5)
    np.testing.assert_allclose(planar[1], 0.40, atol=1e-5)
    np.testing.assert_allclose(planar[2], 0.35, atol=1e-5)


def test_surface_honours_the_projection_command(tmp_path, stand_in_projector_config) -> None:
    """P=(0,0,0) は環境光のみの応答 -b·A^-1 を返す。"""
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(
        f"/api/matrix/{matrix_id}/view/surface?p_r=0&p_g=0&p_b=0"
    )
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    np.testing.assert_allclose(planar[0], -0.05, atol=1e-5)
    np.testing.assert_allclose(planar[1], -0.10, atol=1e-5)
    np.testing.assert_allclose(planar[2], -0.15, atol=1e-5)


def test_surface_marks_singular_pixels_nan(tmp_path, stand_in_projector_config) -> None:
    """A が特異な画素は逆行列を持たないので NaN にする。

    もっともらしい値を描くとユーザーを誤らせる。NaN ならシェーダが
    マスク色で塗る。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[2, 3, :3, :] = 0.0          # この画素だけ A = 0 で特異
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    assert np.isnan(planar[:, 2, 3]).all()
    assert np.isfinite(planar[:, 0, 0]).all()


def test_surface_is_listed_with_its_params() -> None:
    store.clear()
    client = TestClient(create_app())
    views = client.get("/api/views").json()["views"]
    spec = next(v for v in views if v["key"] == "surface")
    assert spec["kind"] == "rgb"
    assert spec["channels"] == 3
    assert {p["name"] for p in spec["params"]} == {"p_r", "p_g", "p_b"}
    assert all(p["type"] == "number" for p in spec["params"])
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_surface_view.py -q`
Expected: FAIL — `404 unknown view: surface`

- [ ] **Step 3: `views/surface.py` を書く**

```python
"""The forward model: what the camera sees for a given projector command.

順方向モデル。行列 M は [C_r, C_g, C_b, 1] @ M = P を満たすので、
M = [A; b]（A は 3x3、b は 1x3）と書けば C·A + b = P。したがって
任意の投影コマンド P に対するカメラ応答は C = (P - b)·A^-1 である。

P = (1,1,1) が投影面の色、P = (0,0,0) が環境光のみの応答になる。
"""

from __future__ import annotations

import numpy as np

from ..matrix import MatrixContext
from .base import ParamSpec, ViewResult, ViewSpec, register

LABELS = ["R", "G", "B"]

# Below this |det(A)| the affine has no usable inverse. The value is a floor on
# float32 round-off, not a physical threshold.
# これ未満の |det(A)| では逆行列が使えない。物理的な閾値ではなく float32 の
# 丸め誤差に対する下限。
_DET_EPS = 1e-12


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Solve C = (P - b) A^-1 per pixel, NaN where A is singular."""
    array = np.asarray(context.array, dtype=np.float32)
    height, width = array.shape[0], array.shape[1]

    a_mat = np.asarray(array[:, :, :3, :], dtype=np.float64)
    b_vec = np.asarray(array[:, :, 3, :], dtype=np.float64)

    command = np.array(
        [
            float(params.get("p_r", 1.0)),
            float(params.get("p_g", 1.0)),
            float(params.get("p_b", 1.0)),
        ],
        dtype=np.float64,
    )

    flat = a_mat.reshape(-1, 3, 3)
    invertible = np.abs(np.linalg.det(flat)) > _DET_EPS
    inverse = np.zeros_like(flat)
    if invertible.any():
        inverse[invertible] = np.linalg.inv(flat[invertible])
    inverse = inverse.reshape(height, width, 3, 3)

    response = np.einsum("hwk,hwkc->hwc", command[None, None, :] - b_vec, inverse)

    # A singular pixel has no answer. Painting a plausible number there would
    # mislead; NaN makes the shader draw the mask colour instead.
    # 特異な画素には答えが無い。もっともらしい数字を描くと誤解を招くので
    # NaN にしてシェーダにマスク色を塗らせる。
    usable = invertible.reshape(height, width)
    if context.active_region is not None:
        x, y, w, h = context.active_region
        inside = np.zeros((height, width), dtype=bool)
        inside[y : y + h, x : x + w] = True
        usable &= inside
    response[~usable] = np.nan

    data = np.ascontiguousarray(response, dtype=np.float32)
    finite = data[np.isfinite(data)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    return ViewResult(data=data, vmin=0.0, vmax=max(vmax, 1e-6), labels=LABELS)


def _command_param(name: str, label: str) -> ParamSpec:
    return ParamSpec(name=name, label=label, type="number", default=1.0)


SPEC = register(
    ViewSpec(
        key="surface",
        label="投影面の色",
        kind="rgb",
        channels=3,
        params=[
            _command_param("p_r", "投影 R"),
            _command_param("p_g", "投影 G"),
            _command_param("p_b", "投影 B"),
        ],
        render=_render,
    )
)
```

`views/__init__.py` に `from . import surface  # noqa: F401  registers "surface"` を追加する。
**`coef_tiles` の後に置くこと** — `REGISTRY` は挿入順で、フロントは既定で先頭を選ぶ。

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 137 passed（133 + 新規4）

この時点で `surface` は API から取得できるが、フロントは `kind="rgb"` を
拒否するため画面には出せない。Task 3 で対応する。

- [ ] **Step 5: 実データで確かめる**

```bash
.venv/Scripts/python.exe -c "
from fastapi.testclient import TestClient
from yoshidacomp_web.app import create_app
import numpy as np
c = TestClient(create_app())
mid = c.post('/api/session/matrix', json={'path': 'data/linear_pipeline_1080/matrices_linear_gamma.npy'}).json()['id']
r = c.get(f'/api/matrix/{mid}/view/surface')
a = np.frombuffer(r.content, dtype='<f4').reshape(3, 1080, 1920)
m = np.nanmedian(a.reshape(3, -1), axis=1)
print('白投影時のカメラ応答の中央値 RGB =', np.round(m, 4))
"
```

設計時の実測では `(0.526, 0.245, 0.436)` だった。緑が赤の半分以下の
マゼンタ寄りの面で、`data/linear_pipeline_1080/README.md` の当日の観察
「白投影で中央=マゼンタ、Gの反射が弱い」と一致する。**大きく外れたら
実装を疑うこと。**

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/views tests/test_web_surface_view.py
git commit -m "feat(web): 投影面の色を出す surface ビューを追加

行列を順方向に解いて C = (P - b)A^-1 を求める。P=(1,1,1) が投影面の色、
P=(0,0,0) が環境光のみの応答になる。A が特異な画素は答えを持たないので
NaN にし、もっともらしい値を描かない。

kind='rgb' なのでフロントはまだ表示できない（次タスクで対応）。"
```

---

### Task 3: フロントエンドの rgb 描画

**Files:**
- Modify: `packages/web/yoshidacomp_web/static/js/shaders.js`
- Modify: `packages/web/yoshidacomp_web/static/js/glview.js`
- Modify: `packages/web/yoshidacomp_web/static/js/main.js`
- Modify: `tests/test_web_golden.py`, `tests/test_web_frontend_js.py`
- Test: 上記

**Interfaces:**
- Consumes: Task 1 の planar、Task 2 の `surface`
- Produces: `u_mode`（0=scalar、1=rgb）を持つシェーダ、`GLView.setImage` の
  RGB32F 経路、`SUPPORTED_KINDS = ["scalar", "rgb"]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_golden.py` に、rgb 経路のトーンチェーンを numpy で再現して
手計算値と一致することを確認するテストを追記する。scalar 経路の既存テストと
同じ形にすること。rgb はカラーマップを通らず、線形RGBへ直接
露出 → γ → sRGB OETF を掛ける。

`tests/test_web_frontend_js.py` の `_MUTATIONS` に、rgb 経路を壊すと
落ちることを確認する項目を追加する。

- [ ] **Step 2: テストが失敗することを確認**

- [ ] **Step 3: シェーダに rgb モードを足す**

`FRAG_SRC` に `uniform int u_mode;` を追加し、`main()` を分岐させる。
**scalar 経路の記述は一字も変えないこと** — `tests/test_web_golden.py` の
`test_numpy_mirror_still_matches_the_shipped_shader` が
`cold` / `warm` / `0.0031308` / `12.92 * c` / `isNaNBits` / `highp uint` /
`0x7F800000u` と**トーンチェーンの順序**を文字列位置で照合している。

rgb 経路は `texture(u_tex, v_uv).rgb` を読み、いずれかの成分が NaN なら
マスク色を返し、そうでなければカラーマップを通さず
露出 → γ → sRGB OETF → ディザを掛ける。

- [ ] **Step 4: `GLView` に RGB32F 経路を足す**

`setImage` に `mode`（`"scalar"` | `"rgb"`）を受け付けさせる。rgb のときは
planar の3チャンネルを1つの `RGB32F` テクスチャへ詰める。**planar から
インタリーブへの詰め替えが1回だけ必要**になる（WebGL のテクスチャは
インタリーブしか受け付けない）。3チャンネルなので12チャンネルの
デインターリーブより遥かに安い。

`NEAREST` の min/mag は維持すること。`LINEAR` にすると補間が NaN を
有限の隣接画素へにじませ、マスク境界が汚染される。

- [ ] **Step 5: `main.js` を対応させる**

`SUPPORTED_KIND` を `SUPPORTED_KINDS = ["scalar", "rgb"]` に変え、
`requireSupportedKind` を配列判定にする。**変異テストアンカー4
（`  if (kind !== SUPPORTED_KIND) {`）を変更するので、
`tests/test_web_frontend_js.py` の `_MUTATIONS` の
`unsupported_kind_renders_as_scalar` を同じコミットで更新すること。**

`rgb` のときのタイル配置を決める。`channels === 3` かつ `kind === "rgb"` なら
グリッドは1枚。`buildGrid` に単一タイル経路を足すか、`axesFromLabels` が
`null` を返したときの `fallbackAxes` を利用する。**どちらを選んだか報告すること。**

ホバー読み取りは3チャンネル分になる。`buildProbeGrid` も追随させる。

- [ ] **Step 6: テストが通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`

- [ ] **Step 7: コミット**

---

### Task 4: ハーネスの拡張と性能の再計測

**Files:**
- Modify: `tests/js/frontend_harness.mjs`
- Modify: `docs/web_viewer_performance_20260731.md`
- Test: `tests/test_web_frontend_js.py`

- [ ] **Step 1: ハーネスに rgb シナリオを足す**

`surface` 相当の `kind="rgb"`, `channels=3` の偽ビューを `VIEWS` に足し、
セレクタから選べること・タイルが1枚になること・ホバーが3値を返すことを
確認するシナリオを追加する。

- [ ] **Step 2: planar の効果を実測する**

`window.__perf` の `uploadMs` を planar 前後で比較する。Task 1 のコミットの
前後で同じ行列（`data/linear_pipeline_1080/matrices_linear_gamma.npy`）を
読み込み、`fetchMs` と `uploadMs` を記録する。

**予測は「uploadMs が大きく下がり fetchMs は変わらない」。** 外れたら
デインターリーブが93%という前提が誤っていたことになるので、
そのまま報告すること。数字を合わせにいかないこと。

- [ ] **Step 3: 性能ドキュメントを更新する**

`docs/web_viewer_performance_20260731.md` に planar 後の数値を追記し、
インタリーブ時の数値は削除せず併記する。B2 の判断根拠が追えるようにする。

- [ ] **Step 4: コミット**

---

## Self-Review

**仕様書との対応:**

| 仕様書の節 | 対応タスク |
| --- | --- |
| §5.1 `surface` ビュー | Task 2 |
| §5.2 順方向モデルの導出 | Task 2 Step 3 |
| §5 `kind` による描画分岐（`rgb`） | Task 3 |
| §7 表示と再計算の分離（維持） | Task 1, 3 |

**本計画の対象外（計画B3以降）:** `compensation` / `gamut` ビュー、
ターゲット画像入力、線形化と LUT カーブ、投影画像生成、
カメラ座標行列の P2C ワープ、サーバ側 probe エンドポイント。

**型の一貫性:** Task 1 が `X-Layout` と `fetchView().layout` を導入し、
Task 3 の `setImage` がそれを読む。Task 2 の `ViewResult` は `(H, W, 3)` で、
Task 1 のルータが `(3, H, W)` へ並べ替える。`LABELS = ["R", "G", "B"]` は
`axesFromLabels` が `"->"` を含まないため `null` を返し、`fallbackAxes` 経路に落ちる
— Task 3 Step 5 はこれを利用してよい。

**リスク:** Task 3 が変異テストアンカー4を変更する。`_MUTATIONS` の更新を
同じコミットに含めないと、次のタスクで「mutation anchor no longer present」
という紛らわしい失敗になる。
