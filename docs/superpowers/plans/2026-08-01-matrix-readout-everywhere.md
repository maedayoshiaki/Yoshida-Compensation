# 行列読み取りの常時表示とレイアウト圧縮 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** どのビューを表示していてもホバーした画素の 4×3 行列が読めるようにし、読み取り欄を圧縮して他の情報を同時に表示する。

**Architecture:** 行列の値はクライアントに持たせず、サーバへの点クエリ（`GET /api/matrix/{id}/probe`）で12個の float だけ取る。`surface` のような非係数ビューではクライアントに行列が無く、2つ目の99MBを転送すると計画B2で半減させたヒープが元に戻るため。画面下の全幅読み取り帯を、行列テーブル・ビュー値・表示レンジ・計測を横に並べたダッシュボードに置き換える。

**Tech Stack:** FastAPI / numpy memmap / 素の ES モジュール / CSS grid + flex

## Global Constraints

- 行列そのもの `(H, W, 4, 3)` をレスポンスに乗せない。点クエリが返すのは1画素分の12個だけ。
- **非有限係数は明示的に `null` に変換すること。** 行列は有効投影領域の外に NaN を
  持ちうる。今日この文脈では2つのフレームワーク層がたまたま同じ結果を作っている
  ――返り値が戻り値注釈つきの `dict` のときは Pydantic の `TypeAdapter` が既定で
  NaN/Inf を `null` にし、それをすり抜けた場合も Starlette の `JSONResponse` は
  `json.dumps(allow_nan=False)` を使うため不正な `NaN` リテラルではなく **500** に
  なる（`JSON.parse` が壊れた本文を受け取ることはない）。しかしどちらもこの
  エンドポイント自身の契約ではなく、戻り値注釈を外す・`Response` を直接返す等の
  変更で消える。変換を怠ると、NaNを持つ行列の該当画素をホバーしただけで 500 に
  なり読み取り欄が空白になる。**この計画が当初「実機行列では全画素の75%が対象」
  と書いていたのは誤り**で、`data/linear_pipeline_1080/matrices_linear_gamma.npy`
  の非有限値は0個である（有効領域の外は全画素の74.86%あるが、そのすべてが有限。
  NaNは `views/coef_tiles.py` が描画時に書いている）。変換はNaNを持ちうる行列の
  ための契約であって、この行列の説明ではない。明示的に変換して
  `null` をこのエンドポイントの契約にする。
- `yoshidacomp.compensation.color_mixing_matrix` を web 層から import しない
  （module scope で torch と psutil を引く）。
- 数値の桁揃えは既存の `fmt()`（`packages/web/yoshidacomp_web/static/js/main.js`）を使う。
- テストは `node` が無ければ skip する既存の仕組みを壊さない。
- 行列の行は `["in_R", "in_G", "in_B", "bias"]`、列は `["out_R", "out_G", "out_B"]`。
  この順序は `views/coef_tiles.py` のチャンネル順 `c = i * 3 + j` と一致する。

---

## File Structure

| ファイル | 責任 |
| --- | --- |
| `packages/web/yoshidacomp_web/routers/session.py` | 点クエリ `GET /api/matrix/{id}/probe` を追加 |
| `packages/web/yoshidacomp_web/static/js/api.js` | `probeMatrix()` を追加 |
| `packages/web/yoshidacomp_web/static/js/main.js` | 読み取りを点クエリ化・合流・ビュー値パネル |
| `packages/web/yoshidacomp_web/static/index.html` | 下部をダッシュボード化、サイドバーから計測を移動 |
| `packages/web/yoshidacomp_web/static/style.css` | 行列テーブルを固定幅に、下部を横並びに |
| `tests/test_web_matrix_probe.py` | 点クエリのテスト（新規） |
| `tests/js/frontend_harness.mjs` | 合流・スタール・描画のチェック追加 |
| `tests/test_web_frontend_js.py` | 変異エントリ追加 |

レビュー対応で以下も触った（当初この表に無かったもの）。ビュー値パネルを
「描いているものが行列でないときだけ出す」に変えるため、判断をビュー仕様の
宣言 `values_are_matrix` としてサーバから運ぶことにしたためである。

| ファイル | 責任 |
| --- | --- |
| `packages/web/yoshidacomp_web/views/base.py` | `ViewSpec.values_are_matrix`（既定は「出す」側の `False`） |
| `packages/web/yoshidacomp_web/views/coef_tiles.py` | 自分の12枚は行列そのものだと宣言する |
| `packages/web/yoshidacomp_web/routers/view.py` | `/api/views` でその宣言を運ぶ |
| `tests/test_web_views.py` | `coef_tiles` の宣言を固定 |
| `tests/test_web_surface_view.py` | `surface` の宣言を固定 |

---

### Task 1: 点クエリのエンドポイント

**Files:**
- Modify: `packages/web/yoshidacomp_web/routers/session.py`
- Test: `tests/test_web_matrix_probe.py`（新規）

**Interfaces:**
- Consumes: `..deps.require_matrix(matrix_id) -> MatrixContext`（存在しないIDで 404 を送る既存関数）、
  `MatrixContext.array` は `(H, W, 4, 3)` の `mmap_mode="r"` 配列、
  `MatrixContext.active_region` は `(x, y, w, h)` または `None`。
- Produces: `GET /api/matrix/{matrix_id}/probe?row=<int>&col=<int>` が
  ```json
  {"row": 12, "col": 34,
   "values": [[0.9,0.0,0.0],[0.0,0.9,0.0],[0.0,0.0,0.9],[0.01,0.01,0.01]],
   "inside_active_region": true}
  ```
  を返す。`values` は行優先の 4×3、非有限値は `null`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_matrix_probe.py` を新規作成する。既存の web テストが
`tests/test_web_session.py` でどうクライアントと一時 `.npy` を用意しているかを
先に読み、同じ作法（同じフィクスチャ名・同じ `store.clear()` の扱い）に合わせること。

```python
"""点クエリ GET /api/matrix/{id}/probe のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from yoshidacomp_web.app import app
from yoshidacomp_web.store import store


@pytest.fixture(autouse=True)
def _clear_store():
    store.clear()
    yield
    store.clear()


def _load(client: TestClient, path) -> str:
    res = client.post("/api/session/matrix", json={"path": str(path)})
    assert res.status_code == 200, res.text
    return res.json()["id"]


def _write_matrix(tmp_path, array) -> str:
    path = tmp_path / "m.npy"
    np.save(path, array.astype(np.float32))
    return path


def test_probe_returns_the_four_by_three_at_that_pixel(tmp_path):
    # 全画素で違う値にする。定数行列だと row/col の取り違えが見えない。
    array = np.arange(4 * 5 * 4 * 3, dtype=np.float32).reshape(4, 5, 4, 3)
    path = _write_matrix(tmp_path, array)
    with TestClient(app) as client:
        matrix_id = _load(client, path)
        res = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 2, "col": 3})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["row"] == 2
    assert body["col"] == 3
    assert body["values"] == array[2, 3].tolist()


def test_probe_does_not_transpose_row_and_col(tmp_path):
    # 正方形でない形にして、取り違えたら 400 か別の値になるようにする。
    array = np.arange(3 * 7 * 4 * 3, dtype=np.float32).reshape(3, 7, 4, 3)
    path = _write_matrix(tmp_path, array)
    with TestClient(app) as client:
        matrix_id = _load(client, path)
        res = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 1, "col": 6})
    assert res.status_code == 200, res.text
    assert res.json()["values"] == array[1, 6].tolist()


def test_probe_turns_non_finite_into_null(tmp_path):
    # Python の json は NaN をそのまま書き、JSON.parse が落ちる。
    array = np.zeros((2, 2, 4, 3), dtype=np.float32)
    array[0, 0, 1, 2] = np.nan
    array[0, 0, 2, 0] = np.inf
    array[0, 0, 2, 1] = -np.inf
    path = _write_matrix(tmp_path, array)
    with TestClient(app) as client:
        matrix_id = _load(client, path)
        res = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 0, "col": 0})
    assert res.status_code == 200, res.text
    # 本文がそのまま JSON として読めることを、生テキストからも確かめる。
    import json as _json
    body = _json.loads(res.text)
    assert body["values"][1][2] is None
    assert body["values"][2][0] is None
    assert body["values"][2][1] is None
    assert body["values"][0][0] == 0.0
    assert "NaN" not in res.text
    assert "Infinity" not in res.text


@pytest.mark.parametrize(
    ("row", "col"),
    [(-1, 0), (0, -1), (2, 0), (0, 2), (99, 99)],
)
def test_probe_refuses_out_of_bounds(tmp_path, row, col):
    array = np.zeros((2, 2, 4, 3), dtype=np.float32)
    path = _write_matrix(tmp_path, array)
    with TestClient(app) as client:
        matrix_id = _load(client, path)
        res = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": row, "col": col})
    assert res.status_code == 400, res.text


def test_probe_on_unknown_matrix_is_404():
    with TestClient(app) as client:
        res = client.get("/api/matrix/nope/probe", params={"row": 0, "col": 0})
    assert res.status_code == 404


def test_probe_reports_whether_the_pixel_is_in_the_active_region(tmp_path):
    # active_region は config 由来。ここでは meta から実際の領域を読み、
    # その内側と外側を1点ずつ問い合わせる。領域が無い行列なら常に true。
    array = np.zeros((8, 8, 4, 3), dtype=np.float32)
    path = _write_matrix(tmp_path, array)
    with TestClient(app) as client:
        matrix_id = _load(client, path)
        meta = client.get(f"/api/matrix/{matrix_id}/meta").json()
        region = meta["active_region"]
        if region is None:
            res = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 0, "col": 0})
            assert res.json()["inside_active_region"] is True
            return
        x, y, w, h = region
        inside = client.get(
            f"/api/matrix/{matrix_id}/probe",
            params={"row": min(y + h - 1, 7), "col": min(x + w - 1, 7)},
        )
        assert inside.json()["inside_active_region"] is True
```

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_matrix_probe.py -v`
Expected: FAIL（404 — ルートが無い）

- [ ] **Step 3: 実装**

`routers/session.py` に追記する。`Query` を `fastapi` から import すること。

```python
def _finite_or_none(value: float) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not finite.

    Python の ``json`` は NaN と Infinity を JSON として不正なリテラルで
    書き出し、ブラウザの ``JSON.parse`` はそれを受け取れない。行列は有効投影
    領域の外に NaN を持ちうるので、線に乗せる前に ``null`` へ落とす。
    """
    v = float(value)
    return v if math.isfinite(v) else None


@router.get("/matrix/{matrix_id}/probe")
def probe_matrix(
    matrix_id: str,
    row: int = Query(..., description="行（0 起点、画像の上から）"),
    col: int = Query(..., description="列（0 起点、画像の左から）"),
) -> dict[str, Any]:
    """Return the 4x3 coefficients at one pixel.

    ビューが何であっても行列そのものを読めるようにするための点クエリ。
    ``surface`` のような非係数ビューではクライアントに行列が無く、
    99MB の2つ目のコピーを送るとヒープが計画B2以前に戻ってしまう。
    mmap からの1画素読みなので転送量は12個の float に収まる。
    """
    context = require_matrix(matrix_id)
    height, width = context.array.shape[0], context.array.shape[1]
    if not (0 <= row < height and 0 <= col < width):
        raise HTTPException(
            status_code=400,
            detail=(
                f"pixel (row={row}, col={col}) is outside the matrix; "
                f"expected 0 <= row < {height} and 0 <= col < {width}"
            ),
        )
    block = context.array[row, col]
    return {
        "row": row,
        "col": col,
        "values": [[_finite_or_none(v) for v in coefficient_row] for coefficient_row in block],
        "inside_active_region": _inside_active_region(context, row, col),
    }
```

`_inside_active_region` は `active_region` が `None` なら `True`、そうでなければ
`(x, y, w, h)` に対して `x <= col < x + w and y <= row < y + h` を返す小さな関数として
同じファイルに書く。`math` を import すること。

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_matrix_probe.py -v`
Expected: PASS（8件）

- [ ] **Step 5: 判別力の確認（変異試験）**

以下を一時的に入れて、それぞれ**必ずどれかのテストが落ちる**ことを確かめ、
落ちたテスト名を報告に書いてから元に戻すこと。1つでも「全部通る」ものがあれば
そのテストは判別力を持っていないので、テストを直してから次へ進む。

1. `context.array[row, col]` を `context.array[col, row]` にする
2. `_finite_or_none` を `float(value)` を返すだけにする
3. 範囲チェックの `<` を `<=` にする

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/routers/session.py tests/test_web_matrix_probe.py
git commit -m "feat(web): 1画素の4x3係数を返す点クエリを足す"
```

---

### Task 2: クライアントを点クエリに切り替える

**Files:**
- Modify: `packages/web/yoshidacomp_web/static/js/api.js`
- Modify: `packages/web/yoshidacomp_web/static/js/main.js`
- Test: `tests/js/frontend_harness.mjs`

**Interfaces:**
- Consumes: Task 1 の `GET /api/matrix/{id}/probe?row=&col=`。
- Produces: `probeMatrix(matrixId, row, col) -> Promise<{row, col, values, inside_active_region}>`
  を `api.js` が export する。`main.js` の `showProbe` が点クエリ経由になる。

**背景（この変更が消す複雑さ）**

いまの `showProbe` は `state.payload` を読む。つまり**表示中のビューの値**であって
行列ではない。`surface` を見ているときにホバーしても出るのは投影面の RGB 3値で、
行列の係数は見えない。`layoutFor` が `probeCols` という別の軸を持ち、
「タイルの並びと読み取りの並びが食い違うのはここだけ」という注釈が付いているのは
このためである。読み取りが常に行列を出すようになれば `probeCols` は不要になるので、
**`layoutFor` の戻り値と `DEFAULT_LAYOUT` から `probeCols` を削除すること。**
`buildProbeGrid` は常に `DEFAULT_ROWS` × `DEFAULT_COLS` で組む。

- [ ] **Step 1: 失敗するテストを書く**

`tests/js/frontend_harness.mjs` に追加する。既存のスタブ（`fetchStub` の登録方法、
`check(...)` の書式、ホバーの起こし方）を先に読み、同じ作法に合わせること。
以下は**追加するチェックの内容**であり、書式は既存に合わせて書き直してよい。

1. `ホバーすると行列の点クエリが飛ぶ`
   `surface` を表示した状態で `(col=3, row=2)` をホバーし、
   `/api/matrix/<id>/probe` への fetch が `row=2` と `col=3` を持って発生すること。
2. `読み取りは12セルすべてを行列の値で埋める`
   応答 `values` が `[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]` のとき、
   12セルのテキストが行優先でその順に並ぶこと（`fmt` 適用後）。
3. `連続ホバー中に飛ぶ点クエリは1つまで`
   応答を保留したまま5回ホバーし、fetch 回数が1であること。
   保留を解いたあと、最後の位置でもう1回だけ飛ぶこと（計2回）。
4. `古い点クエリの応答は捨てる`
   1回目を保留、2回目を先に解決させ、そのあと1回目を解決させると、
   セルの中身が**2回目の値のまま**であること。
5. `null は NaN として表示する`
   `values` に `null` を含めると、そのセルが `probe-cell nan` になること。
6. `ビュー値パネルはペイロードのチャンネルを出す`
   `surface`（3チャンネル、ラベル `["R","G","B"]`）で、パネルが3項目になり
   ラベルと値が対応すること。
7. `行列が未読み込みのときはホバーしても点クエリを投げない`

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_frontend_js.py -v`
Expected: FAIL（新しいチェックが失敗する）

- [ ] **Step 3: 実装**

`api.js` に追加：

```js
// One pixel's 4x3 coefficients. The matrix itself never crosses the wire --
// see routers/session.py::probe_matrix for why this is a point query.
// 1画素分の4x3係数だけを取る。行列本体は線に乗せない。
export async function probeMatrix(matrixId, row, col) {
  const res = await fetch(
    `/api/matrix/${encodeURIComponent(matrixId)}/probe?row=${row}&col=${col}`,
  );
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}
```

`main.js` の変更点：

- `state` に読み取り用の合流フィールドを足す。**表示中のビューのフェッチとは別の
  系列にすること**（`state.reqId` を使い回すと、ホバーのたびにビューのフェッチが
  スタール扱いになる）。

```js
  // Hover readout. Fetches are coalesced rather than queued: a mousemove can
  // fire dozens of times a second and every one of them would otherwise be a
  // round trip. At most one request is in flight; the newest position waits
  // for it and fires once, so a fast drag costs two requests, not fifty.
  // ホバーの点クエリは1つずつに合流させる。飛んでいる間の移動は最新の位置
  // だけを残し、戻ってきたら1回だけ追加で投げる。
  probeSeq: 0,        // monotonic; a response older than this is dropped
  probeInFlight: false,
  probeWanted: null,  // { col, row } requested while a fetch was in flight
```

- `showProbe(hit)` は同期的に「位置の表示」と「ビュー値パネルの更新」を行い、
  行列テーブルは `requestMatrixProbe(hit)` に任せる。
- `requestMatrixProbe` は上のフィールドで合流させ、`probeSeq` で古い応答を捨てる。
  失敗したときはセルを `—` に戻し、`status()` にエラーを出す（ホバーのたびに
  ステータス行が点滅しないよう、同じメッセージの連投は避けてよい）。
- `values` の `null` は `NaN` として描く（`fmt(null)` は `—` を返すので、
  `null` を `NaN` に読み替えてから `fmt` に渡すか、セル側で分岐すること。
  **どちらにするか決めて、その理由をコメントに書くこと。**）
- ビュー値パネル `#view-values` は `state.payload` と `payload.labels` から
  `ラベル: 値` の並びを作る。planar/インタリーブの分岐は既存の `showProbe` に
  あるものをそのまま使う（**この分岐を消さないこと**。ここを取り違えると
  絵は正しいまま数値だけ狂う）。
- `layoutFor` と `DEFAULT_LAYOUT` から `probeCols` を削除し、`buildProbeGrid` は
  引数なしで `DEFAULT_ROWS` × `DEFAULT_COLS` を組む。`layoutFor` の
  `probeCols` に関する注釈も削除する（説明する対象が無くなるため）。
- `fallbackAxes` が `probeCols` のためだけに呼ばれている箇所があれば、
  その呼び出しも消す。他所で使っているなら残す。

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_frontend_js.py -v`
Expected: PASS

- [ ] **Step 5: 変異エントリを足す**

`tests/test_web_frontend_js.py` の `_MUTATIONS` に2件足す。
既存の書式（対象ファイル・完全一致文字列・期待される失敗チェック名の集合）に合わせること。

1. 合流を壊す（`probeInFlight` の判定を無視して毎回投げる）→
   「点クエリは1つまで」が落ちること
2. スタール判定を壊す（`probeSeq` の比較をやめる）→
   「古い応答は捨てる」が落ちること

**双方向の固定集合**（`missing` と `unexpected` の両方）を既存エントリと同じ形で書き、
それぞれ**ちょうど1つのチェックだけ**が落ちることを確かめる。

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/static/js/api.js packages/web/yoshidacomp_web/static/js/main.js tests/js/frontend_harness.mjs tests/test_web_frontend_js.py
git commit -m "feat(web): どのビューでもホバーで行列を読めるようにする"
```

---

### Task 3: 読み取り欄を圧縮して情報を横に並べる

**Files:**
- Modify: `packages/web/yoshidacomp_web/static/index.html`
- Modify: `packages/web/yoshidacomp_web/static/style.css`
- Modify: `packages/web/yoshidacomp_web/static/js/main.js`（要素IDが動く分だけ）
- Test: `tests/js/frontend_harness.mjs`

**Interfaces:**
- Consumes: Task 2 の `#probe-grid` / `#view-values`。
- Produces: なし（最後のタスク）。

**いまの何が無駄か**

`.probe-grid` は `grid-template-columns: var(--rowhead) repeat(3, minmax(0, 1fr))` で、
12個の数値をウィンドウ全幅に引き伸ばしている。1600px の画面で1セルが 500px あり、
中身は `-0.1234` の7文字しかない。同時に表示レンジと計測がサイドバーの下端にあり、
サイドバーは縦に溢れてスクロールが要る。

**やること**

1. `.probe-grid` の列を**固定幅**にする（`grid-template-columns: 38px repeat(3, 66px)`）。
   これでテーブルは約 240px に収まる。
2. 画面下の `.readout` を横並びのダッシュボードにする。パネルは4つ:
   - **行列** — `#probe-pos`・`#probe-pin` を見出しに置き、`#probe-grid` を下に
   - **ビュー値** — 見出しに現在のビュー名、下に `#view-values`
   - **表示レンジ** — `#range-info`
   - **計測** — `#perf-fetch` / `#perf-upload` / `#perf-heap`
   `display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap;` で並べる。
3. サイドバーから `.readouts` ブロック（表示レンジ・転送・GPU転送・JSヒープ）を
   **削除**し、上のダッシュボードへ移す。サイドバーには操作系と凡例だけが残る。
4. `#view-values` は `display: flex; flex-wrap: wrap; gap: 2px 12px;` にして、
   3項目なら1行、12項目でも3行程度に収まるようにする。
5. 既存の `.hint`（「ドラッグ=パン…」）はタイル直下に残す。

**注意**

- 要素の **id は変えないこと**。`main.js` と既存のハーネスが `$("range-info")`
  などで引いている。移動するのは DOM 上の位置と CSS だけ。
- サイドバーの `.readouts` を消すことで `main.js` 側に**参照が切れる箇所が出ないか**
  確認すること。`$("perf-fetch")` などが `null` を引くと `textContent` の代入で
  例外になり、ホバーのたびに落ちる。
- 狭い画面（`@media (max-width: 860px)`）でダッシュボードが崩れないこと。

- [ ] **Step 1: 失敗するテストを書く**

ハーネスに2件足す。

1. `計測の表示先はダッシュボードにある`
   `#perf-fetch` / `#perf-upload` / `#perf-heap` / `#range-info` が
   `.readout` の中に存在すること（`.sidebar` の中ではないこと）。
2. `読み取りテーブルの列は固定幅である`
   `#probe-grid` の `style.gridTemplateColumns` が `1fr` を含まないこと。

ハーネスの DOM スタブが `.readout` の入れ子を表現できない場合は、
**スタブを直して表現できるようにすること**。表現できないからと言って
チェックを弱めない（それは「落ちないテスト」を1件増やすことになる）。
スタブの改修が Task の範囲を超えると判断したら BLOCKED で報告すること。

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_frontend_js.py -v`
Expected: FAIL

- [ ] **Step 3: 実装**

`index.html` の `<section class="readout">` を差し替え、サイドバーの
`.readouts` ブロックを削除する。`style.css` に `.rd-panel` / `.rd-title` /
`.view-values` を足す。

**実装時に1点変更した**: `.probe-grid` の列幅は `style.css` ではなく
`main.js`（`PROBE_HEAD_PX` / `PROBE_CELL_PX`）に置いた。`buildProbeGrid` が
テンプレートをインラインで書き、インラインはシートに勝つので、両方に書けば
シート側は「生きているように見える死んだ記述」になる。シートからは列指定を
削除し、その理由をコメントで残してある。

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全件 PASS

- [ ] **Step 5: 実物で見る**

`uv run uvicorn yoshidacomp_web.app:app --port 8123` を起動し、
`data/linear_pipeline_1080/matrices_linear_gamma.npy` を読み込んで、
**headless Chrome のスクリーンショットで**以下を確認して報告に貼ること
（スタブは見た目を検証しない）:

- `coef_tiles` と `surface` の**両方**でホバーすると行列テーブルが同じ12個を出すこと
- ダッシュボードが1行に収まり、タイルの高さが以前より増えていること
- 1280px 幅でも崩れないこと

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/static/ tests/js/frontend_harness.mjs
git commit -m "refactor(web): 読み取りを圧縮し計測と同時に見えるようにする"
```

---

## Self-Review

**Spec coverage:** ユーザの3要求 — (1) どのビューでも行列が見える = Task 1+2、
(2) コンパクトに = Task 3 の固定幅、(3) 同時に他の情報 = Task 3 のダッシュボード。

**Placeholder scan:** Task 2 の Step 1 はチェックの**内容**を指定し書式は既存に
合わせさせている。これは既存ハーネスの書式を計画側が知らないための意図的な委譲で、
何を検証するかは7件すべて具体的に決まっている。

**Type consistency:** `probeMatrix(matrixId, row, col)` の引数順は Task 1 の
クエリ文字列 `row=&col=` と一致。`values` は行優先 4×3。`DEFAULT_ROWS` /
`DEFAULT_COLS` は既存の定数をそのまま使う。
