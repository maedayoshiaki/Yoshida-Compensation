# ターゲット画像入力・線形化・LUTカーブ 実装計画（計画B3）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 画像を受け付け、行列を適用した補償結果をGUIで見られるようにし、線形化方式（解析ガンマ / 実測逆LUT）を切り替えて比較できるようにする。LUTカーブも表示する。

**Architecture:** 計算は `yoshidacomp.compensation.pipeline` の既存関数を通す。CLIと同じ経路にすることが要点で、web層に補償の数式を再実装しない。ターゲット画像とLUTは行列と同じく `store.py` にIDで保持する。LUTカーブは画素データではないので JSON で返し、2D canvas（`plot.js`）で描く。

**Tech Stack:** FastAPI / numpy / OpenCV / 素の ES モジュール / Canvas 2D

## Global Constraints

- **web層は `yoshidacomp.compensation.color_mixing_matrix` を import しない。**
  module scope で `torch` と `psutil` を引く。
  **[Task 1完了により解消済み・以下は経緯として残す]** Task 1 実行前は
  `pipeline.py` がこの `color_mixing_matrix` を module scope で import しており、
  web層は `pipeline.py` も import できなかった。Task 1
  （`.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`）が
  `_linearize_gamma` を numpy 実装（`_apply_inverse_gamma_numpy`）へ切り替え、
  `calc_compensation_image`（`photometric_compensation.py`）も同様に numpy 化した
  ことで、`pipeline.py` は `generate_projection_image` の呼び出しグラフ全体で
  torch に依存しなくなった。web層は `pipeline.py` を import・使用してよい。
  ただし `pipeline.prepare_target_image_for_compensation`（P2C/C2Pワープ経由で
  `geometry.warp` を呼ぶ）は依然として初回呼び出しで torch を引き約7秒かかる
  （本物のGPUワープのため意図的に未対応。計画B4がこの経路をweb層へ配線する
  場合は事前ウォームアップか停止許容かを判断すること）。
  **上記「`color_mixing_matrix` を import しない」という制約自体は今も有効**
  であり、web層が直接 `color_mixing_matrix.apply_inverse_gamma_correction` を
  import・呼び出すことは変わらず禁止（下記の訂正も参照）。
- 画素データは `application/octet-stream` の float32 リトルエンディアンで送る。
  JSON に載せない。LUTカーブは画素データではないので JSON でよい。
- ビューの本体は **planar `(C, H, W)`**。`X-Layout: planar` を必ず送る。
  クライアントはヘッダが無ければ拒否する（計画Cで必須化済み）。
- ラベルは `X-Labels-B64`（UTF-8 JSON配列のbase64）。HTTPヘッダは latin-1 なので
  和文ラベルを生で置くと 500 になる。
- ビューが入力を扱えない場合は `ViewNotApplicable` を投げる。router が 400 に写す。
  numpy の `ValueError` は 500 のままにする（計画Cで分離済み）。
- 新しい `ViewSpec` には `values_are_matrix` を宣言する。既定は *表示*。
  読み取り欄の「ビュー値」パネルの出し分けに使う。
- **`apply_inverse_gamma_correction`（および Task 1 が追加した numpy版
  `pipeline._apply_inverse_gamma_numpy`）は渡した指数をそのまま `pow` に使う。**
  明るくするには `gamma=1/projector.gamma`（≈0.4545）を渡す。
  **訂正（測定した）: 「省略すると 2.2 が適用され逆方向に暗くなる」は偽。**
  `gamma` は既定値を持たない必須引数で、省略は `TypeError:
  … missing 1 required positional argument: 'gamma'` である（両実装で
  `inspect.signature` の default が `inspect._empty` であることと、実際の
  省略呼び出しが送出することを実測）。暗黙既定は commit `9dca136` が
  意図的に除去済みで、本計画の分岐点 d682917 より前である。
  **`gamma = gamma or 2.2` のような防御的既定を足さないこと**（`9dca136` が
  取り除いたものを戻す変更になる）。実在する罠は反転し忘れだけ。
  **web層からの呼び出しは
  `pipeline.LINEARIZERS["gamma"]` 経由とし、`color_mixing_matrix.
  apply_inverse_gamma_correction` を直接importしないこと**（Global Constraint
  1つ目を参照）。この罠自体はどちらの実装でも同じ。
- 実測LUTの終端は 1.0 とは限らない（ISO400実測でチャンネルランプが
  約 0.876/0.876/0.688）。縦軸を 1.0 で固定しないこと。
- テストは `node` が無ければ skip する既存の仕組みを壊さない。
- ハーネスは**出荷される** `main.js`/`api.js`/`glview.js` を駆動する。写しを作らない。

## テストについて（このリポジトリ固有の事情）

**「通るが落ちることができないテスト」がこれまでに12件出ている。** 直近4件は
直前の計画Cで出たもので、内訳は「有効領域の境界テストが幅と高さの入れ替えでも通る」
「太字で指定した唯一の性質を何も検証していない」「コメントで明言した性質を
何も検証していない」「レビュー指摘に対応して追加しコメント3つで守ったフラグの
読み方を何も検証していない」。

共通の形は**フィクスチャが検証対象の次元について対称**であること、および
**性質を散文で自信を持って述べたことで検証済みに見える**ことである。

各タスクで、書いた各テストについて「どの誤実装なら通ってしまうか」を問うこと。
Step の変異試験は形式ではない。実際に壊して、名前の付いたテストが落ちることを
確認し、落ちなければテストの方を直すこと。

---

## File Structure

**Task 1 実績（当初の計画は下の2行のみを挙げていたが、実際は測定の結果として
4本の本番ファイル・3本のテストファイルに拡大した。詳細は
`.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`）:**

| ファイル | 責任 |
| --- | --- |
| `packages/core/yoshidacomp/compensation/pipeline.py` | ガンマ線形化から torch を外す（当初計画） |
| `tests/test_pipeline_gamma_numpy.py` | numpy版とtorch版の一致検証（新規、当初計画） |
| `packages/core/yoshidacomp/geometry/warp.py` | torch importを関数内へ遅延化（測定後追加） |
| `packages/core/yoshidacomp/compensation/inverse_lut.py` | scipy importを関数内へ遅延化（測定後追加） |
| `packages/core/yoshidacomp/compensation/photometric_compensation.py` | calc_compensation_imageをtorchからnumpyへ切り替え（測定後追加） |
| `tests/test_package_boundaries.py` | import/呼び出し境界テストを追加（測定後追加） |
| `tests/test_photometric_compensation_numpy.py` | calc_compensation_imageのnumpy版とtorch版の一致検証（新規、測定後追加） |
| `tests/_torch_reference.py` | 複数テストファイルが共有するtorch参照実装（新規、レビュー指摘後追加） |

**Task 2 以降のファイル一覧（当初計画のまま）:**

| ファイル | 責任 |
| --- | --- |
| `packages/web/yoshidacomp_web/target.py` | ターゲット画像の読み込みと保持（新規） |
| `packages/web/yoshidacomp_web/routers/target.py` | `POST /api/target`（新規） |
| `packages/web/yoshidacomp_web/lut.py` | 逆LUTの読み込みとカーブ抽出（新規） |
| `packages/web/yoshidacomp_web/routers/lut.py` | `POST /api/lut` / `GET /api/lut/{id}/curve`（新規） |
| `packages/web/yoshidacomp_web/views/compensation.py` | `compensation` ビュー（新規） |
| `packages/web/yoshidacomp_web/store.py` | ターゲットとLUTのストアを追加 |
| `packages/web/yoshidacomp_web/static/js/plot.js` | LUTカーブの2D canvas（新規） |
| `packages/web/yoshidacomp_web/static/js/main.js` | ターゲット/LUT の選択UIとカーブ表示 |
| `packages/web/yoshidacomp_web/static/index.html` | 入力欄とカーブ領域 |

---

### Task 1: ガンマ線形化から torch を外す

**Files:**
- Modify: `packages/core/yoshidacomp/compensation/pipeline.py`
- Test: `tests/test_pipeline_gamma_numpy.py`（新規）

**Interfaces:**
- Consumes: `yoshidacomp.compensation.color_mixing_matrix.apply_inverse_gamma_correction(image, gamma, device)` —
  torch実装。`clamp(in ** gamma, 0, 1)` を正規化値に対して計算し、入力の dtype を保つ。
- Produces: `pipeline.py` が module scope で `color_mixing_matrix` を import しなくなる。
  `LINEARIZERS["gamma"]` の呼び出し規約（引数・戻り値の dtype と形状）は不変。

**なぜこれが最初か**

`pipeline.py` は `apply_inverse_gamma_correction` を module scope で import しており、
それが `torch` と `psutil` を引く。web層は `color_mixing_matrix` を import しない
規約なので、この連鎖のせいで web層は `pipeline.py` 全体を使えない。以降の全タスクが
`pipeline.py` の関数を必要とする。

加えて、関数の中身は正規化値に対する `clamp(x ** gamma, 0, 1)` であり、
1920×1080×3 の冪乗のために CUDA を初期化するのは対話的GUIでは割に合わない
（初回に数秒止まり、プロジェクタを駆動している同じGPUのメモリを取る）。

**この Task は「numpy に置き換える」ではなく「一致するかを測ってから決める」ものである。**

- [ ] **Step 1: 一致を測るテストを書く**

`tests/test_pipeline_gamma_numpy.py` を新規作成する。既存のテストの作法
（フィクスチャ名・`stand_in_projector_config` のような既存ヘルパの有無）を先に確認し、
それに合わせること。

検証すること:

1. `uint8` の全256値に対し、numpy版と torch版が**バイト単位で一致する**こと。
   `np.arange(256, dtype=np.uint8).reshape(...)` を使い、丸めの境界を全部踏むこと。
2. `float32` について、`[0, 1]` を細かく刻んだ値と、範囲外の値
   （負値、1超）でも一致すること。**clamp の効き方が違うと危ない箇所である。**
3. `uint16` を扱う経路があるなら同様に確認する（無ければテストも書かない）。
4. `gamma` に `1/2.2` と `2.2` の両方を渡し、**方向が逆になること**を確認する。
   これは規約の罠（指数をそのまま使う）を固定するためのテストであり、
   一致検証とは別の目的を持つ。

torch が入っていない環境では 1〜3 を skip すること（`pytest.importorskip("torch")`)。
4 は numpy 版だけで成立するので skip しない。

**判別力について**: 全部を `[0,1]` の内側の float だけで測ると、clamp の違いも
uint8 の丸めの違いも見えない。範囲外と整数境界を必ず含めること。

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_gamma_numpy.py -v`
Expected: FAIL（numpy版がまだ無い）

- [ ] **Step 3: numpy 版を書き、一致を実測する**

`pipeline.py` に純numpyの実装を足す。名前は `_apply_inverse_gamma_numpy` とし、
module private にする（公開APIを増やさない）。

```python
def _apply_inverse_gamma_numpy(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Raise a normalized RGB image to *gamma*, clamped to [0, 1].

    ``color_mixing_matrix.apply_inverse_gamma_correction`` の numpy 版。
    *gamma* はそのまま指数として使うので、明るくするには 1/projector.gamma を
    渡す。入力の dtype を保つ。

    torch 版は 600 万画素の冪乗のために CUDA を初期化する。対話的GUIでは
    初回に数秒止まり、プロジェクタを駆動しているのと同じGPUのメモリを取るため
    割に合わない。加えて torch は module scope で psutil ごと web 層へ漏れる。
    """
```

正規化のスケール（uint8 なら 255、uint16 なら 65535、float はそのまま）と、
戻すときの丸めを **torch 版の実装を読んで合わせること**。推測しない。

**測った結果を報告に必ず書くこと。**

- バイト単位で一致した場合 → `_linearize_gamma` を numpy 版へ切り替え、
  `color_mixing_matrix` の module-scope import を削除する。
- 一致しなかった場合 → **切り替えない。** `_linearize_gamma` は torch 版のまま
  遅延 import にして module scope から外し、差の内訳（最大絶対差・
  何個の値が違うか・どの入力で）を報告に書く。numpy 版は web 層専用として残し、
  差があることをコメントに明記する。

どちらの結末になったかで後続タスクの前提が変わるので、**status に明示すること。**

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_gamma_numpy.py -v`
Expected: PASS

- [ ] **Step 5: 境界テスト**

`pipeline.py` を import しても `torch` が `sys.modules` に入らないことを確認する
テストを足す。既に `tests/` にパッケージ境界のテストがあればそこへ、無ければ
このテストファイルに書く。

```python
def test_importing_pipeline_does_not_pull_torch():
    import subprocess, sys
    code = (
        "import sys; import yoshidacomp.compensation.pipeline; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", out.stdout
```

**サブプロセスで確認すること。** 同一プロセスでは他のテストが先に torch を
import している可能性があり、そのとき何も検証しないテストになる。

- [ ] **Step 6: 既存CLIの回帰を確認**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 196 passed（増分は新規テスト分のみ）

- [ ] **Step 7: 変異試験**

1. `_apply_inverse_gamma_numpy` の clamp を外す → 範囲外のテストが落ちること
2. 正規化のスケールを 255 から 256 にする → uint8 のテストが落ちること
3. `1.0 / projector.gamma` を `projector.gamma` にする → 方向のテストが落ちること

それぞれ落ちたテスト名を報告に書くこと。

- [ ] **Step 8: コミット**

```bash
git add packages/core/yoshidacomp/compensation/pipeline.py tests/test_pipeline_gamma_numpy.py
git commit -m "refactor(core): ガンマ線形化から torch の連鎖を切る"
```

---

### Task 2: ターゲット画像の入力

**Files:**
- Create: `packages/web/yoshidacomp_web/target.py`
- Create: `packages/web/yoshidacomp_web/routers/target.py`
- Modify: `packages/web/yoshidacomp_web/store.py`, `app.py`
- Test: `tests/test_web_target.py`（新規）

**Interfaces:**
- Consumes: `yoshidacomp.compensation.pipeline.decode_target(image, mode, gamma)` と
  `load_target_images(dir)`。**実装前に両方の実シグネチャを読むこと**（この計画の
  記述は要約であり、引数名と既定値は実物が正しい）。
- Produces:
  - `POST /api/target` — body `{"path": "<画像ファイルまたはディレクトリ>"}` →
    `{"id": "target1", "meta": {...}}`
  - `GET /api/target/{id}/meta` → 形状・元パス・デコード方式
  - `store.targets` に `TargetContext` を保持する（`MatrixStore` と同じ作法）。

**設計判断**

アップロードではなく**パス指定のみ**とする。このGUIはオフライン専用でローカルの
`data/` を直接読む前提であり、99GB のツリーが手元にある。アップロード経路は
一時ファイルの寿命管理を増やす割に使われない。

ディレクトリを渡された場合は `load_target_images` で全画像を読み、
`meta` に一覧を返して**先頭を既定**にする。個別の選択は `index` パラメータで行う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_target.py` を新規作成する。既存の `tests/test_web_matrix.py` の
作法（クライアントの作り方、`store.clear()` の扱い、一時ファイルの置き方）に
合わせること。

検証すること:

1. 単一の画像ファイルを渡すと id が返り、`meta` の `shape` が実際の画像と一致する
2. ディレクトリを渡すと `meta` に画像名の一覧が入り、件数が正しい
3. 存在しないパスは 404
4. 空文字は 400（`Path("")` は `Path(".")` になるので先に弾く。
   `routers/session.py` に同じ扱いの前例がある）
5. 画像でないファイル（テキスト等）は 400 で、理由が読める文言であること
6. `GET /api/target/{id}/meta` が `POST` の `meta` と一致する
7. 未知の id は 404

**判別力について**: 正方形の画像を使うと縦横の取り違えが見えない。
**縦横が違うサイズ**（例 5×3）の画像を使うこと。1件目の欠陥はまさにこの形で出た。

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_target.py -v`
Expected: FAIL

- [ ] **Step 3: 実装**

`target.py` に `TargetContext` を dataclass で定義する。最低限持つもの:
画像本体（`(H, W, 3) uint8`）、元パス、ディレクトリだった場合の一覧と選択中の index。

`store.py` には `MatrixStore` と同じ形の `TargetStore` を足す。既存の
`MatrixStore` をコピーせず、**共通部分を切り出して両方から使う形にすること**
（id採番・ロック・`get`/`put`/`clear` は完全に同じ）。

`app.py` に router を登録する。`/api/*` を先に、`StaticFiles` を最後に
マウントする既存の順序を崩さないこと。

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_target.py -v`
Expected: PASS

- [ ] **Step 5: 変異試験**

1. `shape` の H と W を入れ替えて返す → 縦横違いのテストが落ちること
2. ディレクトリの一覧をソートしない → 順序を検証しているテストがあれば落ちること
   （落ちなければ順序を検証していないので、決定的な順序を仕様にしてテストを足す）
3. 空文字のガードを外す → 400 のテストが落ちること

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/target.py packages/web/yoshidacomp_web/routers/target.py packages/web/yoshidacomp_web/store.py packages/web/yoshidacomp_web/app.py tests/test_web_target.py
git commit -m "feat(web): ターゲット画像をパス指定で読み込む"
```

---

### Task 3: `compensation` ビュー

**Files:**
- Create: `packages/web/yoshidacomp_web/views/compensation.py`
- Modify: `packages/web/yoshidacomp_web/views/__init__.py`
- Test: `tests/test_web_compensation_view.py`（新規）

**Interfaces:**
- Consumes: Task 2 の `store.targets`、`pipeline.generate_projection_image()`、
  `views/base.py` の `ParamSpec` / `ViewResult` / `ViewSpec` / `register` /
  `ViewNotApplicable`。`views/surface.py` を**先に読むこと** — カメラ座標の
  拒否・コスト上限・`vmin`/`vmax` の決め方に前例がある。
- Produces: `REGISTRY["compensation"]`。`kind="rgb"`、`channels=3`、
  `values_are_matrix=False`。

**パラメータ**

| name | type | 既定 | 内容 |
| --- | --- | --- | --- |
| `target` | select | 読み込み済みターゲットのid | どの画像を補償するか |
| `linearizer` | select | `none` | `none` / `gamma` / `inverse_lut` |
| `lut` | select | なし | `linearizer="inverse_lut"` のときのLUT id |
| `output` | select | `linear` | `linear`（補償画像）/ `code`（投影コード値） |

`ParamSpec` の `choices` は静的な `list[str]` である。ターゲットやLUTのidは
実行時に増えるので、**`/api/views` を叩いた時点の一覧を返す形にするか、
`type="select"` ではなく文字列入力にするかを決めること。** 前者ならレジストリの
`params` が動的になり、`ViewSpec` が frozen であることと衝突する。
**この衝突をどう解くかは実装者の判断とし、選んだ理由を報告に書くこと。**
最小の解は `choices` を空にして文字列として受け、不正なidを `ViewNotApplicable`
にすることである。

**拒否する条件**

- 行列がカメラ座標 → `ViewNotApplicable`（`surface` と同じ理由・同じ作法）
- `target` が未指定または未知のid → `ViewNotApplicable`
- `linearizer="inverse_lut"` かつ `lut` が未指定 → `ViewNotApplicable`

- [ ] **Step 1: 失敗するテストを書く**

検証すること:

1. **CLIと同じ結果になること。** 合成した小さな行列とターゲットで、
   ビューの出力が `generate_projection_image()` を直接呼んだ結果と一致する。
   **これがこのタスクの中心的な主張である。** 補償の数式をビュー側に
   再実装していないことを、この1件が担保する。
2. `linearizer` を `none` / `gamma` で切り替えると出力が**変わる**こと。
   同じになるなら線形化が効いていない。
3. `output` を `linear` / `code` で切り替えると出力が変わること。
4. カメラ座標の行列は `ViewNotApplicable`（router 経由で 400）。
5. 未知の target id は `ViewNotApplicable`。
6. `linearizer="inverse_lut"` で lut 未指定は `ViewNotApplicable`。
7. `ViewResult.data` が `(H, W, 3) float32` で、`labels` が3要素であること。

**判別力について（重要）**

- 行列を単位行列に近い対称なものにしないこと。**計画B2で、`A` が全て `2·I`
  だったために逆行列の転置が14テスト全部を通った。** 非対称な行列を使うこと。
- ターゲット画像を単色にしないこと。単色だと画素の入れ替えが見えない。
- ターゲットの縦横を違うサイズにすること。
- テスト1の期待値は**手計算の定数にしない。** `generate_projection_image()` を
  テスト内で直接呼んで突き合わせること。定数を書き間違える余地が無くなる。

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_compensation_view.py -v`
Expected: FAIL

- [ ] **Step 3: 実装**

`views/surface.py` の構造に倣う。`MatrixContext` から行列と有効領域を取り、
ターゲットを `store.targets` から引き、`generate_projection_image()` を呼ぶ。

**有効領域の外は NaN にすること。** `coef_tiles` と `surface` が同じ規約で
マスクしており、計画Cで読み取り欄がこの規約に合わせてある。ここだけ 0 で
埋めると、マスク色ではなく黒として描かれ、読み取り欄のマーカーとも食い違う。

`vmin`/`vmax` は `surface` の折衷（`min(0.0, nanpercentile(finite, 1))` と
`nanpercentile(finite, 99)`）を踏襲する。3チャンネルが1つのレンジを共有するため、
床を浮かせると R/G/B から同じ定数を引くことになり露出ではなく彩度低下になる。

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_compensation_view.py -v`
Expected: PASS

- [ ] **Step 5: 変異試験**

1. `generate_projection_image()` の戻り値2つ（linear と code）を入れ替える →
   `output` のテストが落ちること
2. 有効領域マスクを外す → マスクのテストが落ちること
3. `linearizer` パラメータを無視して常に `none` を渡す → テスト2が落ちること
4. ターゲット画像を転置して渡す → 縦横違いのテストが落ちること

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/views/compensation.py packages/web/yoshidacomp_web/views/__init__.py tests/test_web_compensation_view.py
git commit -m "feat(web): ターゲットに行列を適用する compensation ビュー"
```

---

### Task 4: ターゲットとLUTの読み込みUI、カーブ表示

**【計画の訂正】** 当初この Task は LUT のみを扱う手順で書かれていたが、
File Structure 表は `main.js` に「ターゲット/LUT の選択UI」を割り当てており、
**どの Task の手順もターゲット読み込みUIを作らなかった。** Task 3 のレビューが
独立にこれを指摘している（`compensation` ビューはブラウザから駆動できず、
`POST /api/target` に呼び出し側が存在しない）。計画の中心的な成果物が
curl 無しでは使えない状態になるため、ターゲットUIをこの Task に加える。

**追加要件（Step 5 の前に実施すること）**

- ツールバーかサイドバーに**ターゲット画像のパス入力と読み込みボタン**を置く。
  行列の `path-input` / `btn-load` と同じ作法にすること。
- 読み込むと `POST /api/target` を呼び、返った id を
  `compensation` ビューの `target` パラメータ欄へ**自動的に入れる**。
  手で貼らせない。
- ディレクトリを読み込んだ場合は `meta.images` の一覧を出し、選ぶと
  `index` が変わって再取得されること。
- 失敗（404/400）は既存の `errorMessage()` 経由で散文として出すこと。
- ハーネスに次を足す:
  1. 読み込み後に `target` 欄が id で埋まること
  2. 一覧から選ぶと `index` が変わって再取得が起きること
  3. 404 のとき散文が出て、UIが操作不能にならないこと



**Files:**
- Create: `packages/web/yoshidacomp_web/lut.py`, `routers/lut.py`
- Create: `packages/web/yoshidacomp_web/static/js/plot.js`
- Modify: `store.py`, `app.py`, `static/index.html`, `static/js/main.js`, `static/js/api.js`
- Test: `tests/test_web_lut.py`（新規）、`tests/js/frontend_harness.mjs`

**Interfaces:**
- Consumes: `yoshidacomp.compensation.inverse_lut.load_inverse_lut(path) -> InverseLut`。
  **実装前に `InverseLut` の実フィールドを読むこと。** 計画のこの記述は要約である。
  仕様書には `.npz` に `lut` `(3, 256) float32`、`t_grid` `(256,) float32`、
  `meta`（JSON文字列）とある。
- Produces:
  - `POST /api/lut` — body `{"path": "..."}` → `{"id": "lut1", "meta": {...}}`
  - `GET /api/lut/{id}/curve` → `{"t_grid": [...], "lut": [[...], [...], [...]], "meta": {...}}`
  - `plot.js` が `drawLutCurve(canvas, curve, options)` を export する。

**注意**

**[Task 1完了により解消済み]** この節は執筆時点（`inverse_lut.py` が
module scope で `scipy.interpolate.PchipInterpolator` を import していた頃）の
記述である。Task 1（`.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`、
コミット `9f73f71`）が測定した結果、`PchipInterpolator` の唯一の利用箇所は
`_build_channel_lut`（LUT*構築*時のみ。`pipeline.py` が使う `apply_inverse_lut`
のホットパスは `np.interp` のみ）だったため、import を `_build_channel_lut` の
関数内へ遅延化した。**Task 4 の実装者がこのimportコストを計測しても、scipyの
コストはもう存在しない** -- ここで計測して見つからなかった場合、それは計画が
誤っていたのではなく、Task 1 で既に解消済みだったということである（計測結果と
計画のどちらが誤りか判断できない、という事態を避けるためにここへ明記する）。
`cv2` は今も module scope で import されている（撮影用コードが直接使うため）。
`inverse_lut.py` は撮影用の巨大なコードを抱えている点は変わらない。

- [ ] **Step 1: 失敗するテストを書く（サーバ側）**

`tests/test_web_lut.py`。検証すること:

1. 実際の `.npz` を書いて読み込ませ、`curve` が `t_grid` と `lut` を
   保存した値のまま返すこと
2. `lut` が `(3, 256)` の入れ子リストであること（`[R行, G行, B行]` の順）
3. 存在しないパスは 404、空文字は 400、未知の id は 404
4. **チャンネルの順序が入れ替わっていないこと。** 3チャンネルに
   **明確に違う形**のカーブを入れること（例: R は直線、G は上に凸、
   B は終端が 0.688 で止まる）。3本とも同じ形だと入れ替えが見えない。
5. `meta` の JSON 文字列がパースされて辞書として返ること

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_lut.py -v`
Expected: FAIL

- [ ] **Step 3: サーバ側の実装**

`lut.py` に `LutContext`、`store.py` に `LutStore`（Task 2 で切り出した共通部分を使う）。

- [ ] **Step 4: 通ることを確認 → コミット**

```bash
git commit -m "feat(web): 逆LUTを読み込みカーブを返す"
```

- [ ] **Step 5: フロントの失敗するテストを書く**

ハーネスに追加する。既存の書式に合わせること。検証すること:

1. LUTを読み込むとカーブ領域に3本の線が描かれること
   （`plot.js` は canvas に描くので、**2Dコンテキストのスタブが
   `moveTo`/`lineTo`/`stroke` の呼び出しを記録できるようにすること。**
   スタブが記録できないなら、記録できるように直すこと。チェックを弱めない）
2. **縦軸が 1.0 に固定されていないこと。** 終端が 0.688 のカーブを与えたとき、
   縦軸の上限がデータ由来であること。これは仕様の明示要件である。
3. 解析ガンマを重ねたとき、**LUTのカーブとは別の線として**描かれること
4. DPR が 1 でないときにスケーリングが入ること
   （`devicePixelRatio` を 2 にして `canvas.width` が CSS 幅の2倍になること）

**判別力について**: 「3本描かれた」だけを数えると、3本とも同じ座標でも通る。
**各線の座標が入力と対応していること**を検証すること。

- [ ] **Step 6: 落ちることを確認 → 実装 → 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_frontend_js.py -v`

- [ ] **Step 7: 変異エントリを足す**

`_MUTATIONS` に2件。双方向の固定集合（`missing` と `unexpected`）を
既存エントリと同じ形で書くこと。

1. 縦軸を 1.0 固定にする → 上限のチェックが落ちること
2. チャンネルの描画順を入れ替える → 座標対応のチェックが落ちること

- [ ] **Step 8: 実物で見る**

`uv run uvicorn yoshidacomp_web.app:app --port 8123` を起動し、
実際の LUT（`data/` 配下の `.npz` を探すこと）を読み込んで
**headless Chrome のスクリーンショット**を撮り、報告に貼ること。

- 3本のカーブが区別できること
- 終端が 1.0 未満のチャンネルがそのまま見えること
- 解析ガンマと重ねて比較できること

- [ ] **Step 9: コミット**

```bash
git add packages/web/yoshidacomp_web/static/ tests/js/frontend_harness.mjs tests/test_web_frontend_js.py
git commit -m "feat(web): LUTカーブを解析ガンマと重ねて表示する"
```

---

## Self-Review

**Spec coverage:** 仕様書 §12 の項目7「ターゲット入力、線形化、LUTカーブ」を
Task 2/3/4 が覆う。Task 1 は §8 が指定していた `apply_inverse_gamma_correction`
の利用と Global Constraints の torch 禁止が衝突するために必要になった前提整備。
**[測定後の訂正]** Task 1 はこの衝突を §8 に従うことでではなく、§8 から
**離れる**ことで解消した（`gamma` linearizer を `pipeline.LINEARIZERS["gamma"]`
経由のnumpy実装へ切り替え、§8 が指定していた torch 関数を呼ばなくした）。
そのため §8 自体を訂正済み（`docs/superpowers/specs/2026-07-31-matrix-viewer
-gui-design.md` の該当箇所）。仕様が Global Constraint と矛盾していたのは
計画のミスであり、Task 1 が「§8を実装した」のではなく「§8の指示が誤りだったと
判明したので仕様側を直した」という順序である点を記録として残す。

**スコープ外（計画B4へ）:** `gamut` ビュー（`categorical` kind のフロント実装が
まるごと必要）、`POST /api/generate`（投影画像の保存とメタデータJSON）、
カメラ座標行列のP2Cワープ（バックグラウンドジョブ）。**このP2Cワープ経路は
`pipeline.prepare_target_image_for_compensation` または `geometry.warp` を
直接呼ぶことになるが、Task 1 はこの経路のtorch依存を解消していない**
（`invwarp_image`/`warp_image` はGrayCodeの `PixelMapWarperTorch` による
本物のGPU規模ワープで、意図的に対象外とした）。初回呼び出しで実測約7秒
かかりCUDAを初期化する。計画B4の実装時は、プロセス起動時にtorchを
事前ウォームアップするか、初回リクエストの約7秒停止を受け入れるかを
判断すること -- 「Task 1で解決済み」ではない。

**Placeholder scan:** Task 3 の `ParamSpec` の動的 choices 問題と、Task 1 の
一致/不一致の分岐は、**意図的に実装者の判断へ委ねている**。どちらも実測か
実物のシグネチャを見ないと決められず、計画側で決め打ちすると誤った前提を
固定することになる。判断の理由を報告に書くことを求めている。

**Type consistency:** `TargetContext` / `LutContext` / `MatrixContext` は
いずれも store に id で入る。共通のストア実装を Task 2 で切り出し、Task 4 が
それを使う。`ViewSpec` の `values_are_matrix` は計画Cで追加済みで、
`compensation` は `False` を宣言する。

---

## 完了記録と申し送り

**ブランチ**: `feature/target-and-lut`（`main` d682917 から21コミット）
**テスト**: 196 → **293 passed**（うちJSハーネス132チェック / 7シナリオ）

### ユーザの必要最低限要件が揃った

| 要件 | 実現 |
| --- | --- |
| 画像を受け付けて投影画像を生成 | Task 2（`POST /api/target`）+ Task 3（`compensation` ビュー） |
| ガンマ補正（LUT対応） | Task 1（線形化レジストリ）+ Task 4（`POST /api/lut`） |
| LUTカーブをGUIで見る | Task 4（`plot.js`、縦軸はデータ由来） |

補償ビューが**CLIと同じ関数を通る**ことはテスト1件が担保する。
`calc_compensation_image` のバイアス項を削除しても23件全部が緑のままであることを
レビューが確認しており、これは両辺が一緒に動く＝ビュー側に独立した補償演算が
残っていないことの証明である。

### Task 1 は前提整備のはずが性能の本題だった

| | 元 | 現在 |
| --- | ---: | ---: |
| `import pipeline` | 9.4〜13.0 s | **0.29 s** |
| `generate_projection_image` 初回 | 7.94 s | **0.015 s** |

torch を4箇所（`pipeline` / `photometric_compensation` / `warp` / `inverse_lut`の scipy）
から外し、`_linearize_gamma` と `calc_compensation_image` を numpy 化した。

**判断は測定で決めた。** 当初「numpy に置き換える」つもりだったが、float32 では
一致しない（CUDA 774/12003、CPU 222/12003、最大 1 ULP）。しかし
**torch自身の CPU と CUDA が同じ幅で食い違う**ので、保存すべき「torch の答え」は
存在しない。実データ622万画素の uint8 で:

| 比較 | 不一致 |
| --- | ---: |
| numpy vs torch-CPU | **0** |
| numpy vs torch-CUDA | 9 |
| torch-CUDA vs torch-CPU（ノイズ床） | **同じ9個、位置も大きさも一致** |

代償は1フレーム +0.15秒（numpy 0.22s vs CUDA 0.08s）。7.1秒の import を
1回避けるのに十分見合う。

### 残る torch 依存（計画B4への申し送り）

`pipeline.prepare_target_image_for_compensation` は `geometry.warp` 経由で
**初回呼び出しに約7秒**かかる。GrayCode の `PixelMapWarperTorch` による
2M点の散布ワープであり、12回の積和とは規模が違うので意図的に未対応。
**B4 がこの経路をweb層へ配線する際に、起動時の事前ウォームアップか
停止許容かを判断すること。**「もう解決済み」ではない。

### 「落ちないテスト」がこのブランチで7件（累計22件）

| # | 場所 | 通ってしまった誤実装 |
| --- | --- | --- |
| 13 | 報告が「CLI出力の保証」と名指ししたテスト | 参照側が numpy を呼ぶよう漂流。バイアス行削除で6/6緑 |
| 14-17 | Task 2 の4件 | NTFS が既にソート済みで返すためソート削除で11/11緑、ほか |
| 18 | Task 4 のガンマ | 実configのγが2.2なので2.2ハードコードで19/19緑（実装者が自分で発見） |
| 19 | ローダのフェッチ抑止 | 行列未読込のシナリオなので常に0。両ローダに再描画を足しても12/12緑 |
| 20 | `loadInverseLut` の順序保証 | 散文のみ。id を先に公開しても12/12緑 |
| 21 | 仕様 §8 のガンマ既定 | staleness を直した同じ段落の隣の文が偽のまま残った |
| 22 | 復号できないファイルの一覧残留 | 散文のみ。除外に戻しても292/292緑 |

**22件目が最も示唆的である。** I2（散文だけの保証）を直したのと**同じ修正ウェーブ**が、
構造的に同一の欠陥を再導入した。この欠陥は「最後に書き留めて信じたもの」に付く。
同じ種類の欠陥への修正であることも例外にならない。

18件目は実装者が自分で見つけた。**期待値が実環境の既定値と一致するときは要注意**
という形で、レビュー側にも「既定値と一致する期待値を特に探せ」と指示すべきである。

### 計画・仕様の記述の誤り（測って判明したもの）

1. 仕様 §8 が `color_mixing_matrix.apply_inverse_gamma_correction` を使えと指示。
   Global Constraints の torch 禁止と衝突しており、従うと web層へ torch が戻る。
2. 「gamma を省略すると2.2が適用され暗くなる」は偽。commit `9dca136` 以降
   必須引数で、省略は `TypeError`。**防御的な `gamma or 2.2` を足さないこと。**
3. 「有効領域外の75%は NaN」は偽（計画C由来）。非有限値は0個で、
   領域外74.86%は全て有限。NaN は `coef_tiles` が描画時に書いている。
4. `GET /api/target/{id}/meta` に `デコード方式` を含めた契約は実現不能。
   `decode_target` は `mode`/`gamma` を必要とし、それを選ぶのは Task 3 である。
5. ターゲット読み込みUIをどの Task の手順も作っていなかった。
   File Structure 表は `main.js` に割り当てていたが手順が欠けていた。

### 申し送り

- `plot.js` の `vMin` 下限は未検証（負値サンプルのLUTフィクスチャが要る）。
- `_IdStore` は設計上無制限。eviction を足すとスレッド安全性テストの
  「発行した id は必ず解決する」と衝突する。
- `index.html` の `path-input` はマシン固有パスのまま（本ブランチ以前から）。
  ファイル内コメントで「一貫性の回復」として戻さないよう警告してある。
- `examples/python/sample.py` は `LINEARIZERS` を通らず
  `color_mixing_matrix.apply_inverse_gamma_correction` を直接呼ぶ。
