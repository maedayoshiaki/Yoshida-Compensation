# 色域外ビューと投影画像の生成 実装計画（計画B4）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 補償が破綻している画素（クリップされる画素）をGUIで可視化し、CLIと同じ投影画像をGUIから保存できるようにする。

**Architecture:** `gamut` はクリップ**前**の補償値を4クラスに分類した1チャンネルのクラス番号マップで、フロントに `categorical` kind の描画経路を新設する。生成は `pipeline.generate_projection_image()` を通し、保存の並びは `rebuild_compensation_with_map.py` に合わせる。

**Tech Stack:** FastAPI / numpy / OpenCV / 素の ES モジュール / WebGL2

## Global Constraints

- **web層は `yoshidacomp.compensation.color_mixing_matrix` を import しない。**
  ガンマは `pipeline.LINEARIZERS["gamma"]` 経由で使う。
- **`pipeline.prepare_target_image_for_compensation` を呼ばない。** `geometry.warp`
  経由で torch を引き初回約7秒かかる。本計画では不要。
- ビューの本体は **planar `(C, H, W)`** float32。`X-Layout: planar` を必ず送る。
- ラベルは `X-Labels-B64`（UTF-8 JSON配列のbase64）。HTTPヘッダは latin-1 なので
  `gamut` の和文ラベルを生ヘッダに置くと 500 になる。
- 有効投影領域の外は **NaN**。`coef_tiles` / `surface` / `compensation` と同じ規約で、
  読み取り欄の領域外マーカーがこれに依存している。
- ビューが入力を扱えない場合は `ViewNotApplicable`（router が 400）。
  numpy の `ValueError` は 500 のまま。
- `ViewSpec` に `values_are_matrix` を宣言する。`gamut` は `False`。
- `/api/*` を先に登録し、`StaticFiles(html=True)` を最後にマウントする。
- テストは `node` が無ければ skip する既存の仕組みを壊さない。
- ハーネスは**出荷される** JS を駆動する。写しを作らない。

## テストについて（このリポジトリ固有の事情）

**「通るが落ちることができないテスト」がこれまでに22件出ている。** 直近の計画B3だけで7件。
共通の形は3つある。

1. **フィクスチャが検証対象の次元について対称。** 実データの行列は非対称
   （`max|offdiag|/max|diag| ≈ 0.25`）だが、テストが `2·I` を使っていたため
   逆行列の転置が14テスト全部を通った。
2. **性質を散文で自信を持って述べたことで検証済みに見える。** 「ディスク上でわざと
   アルファベット順を崩している」というコメントに反して NTFS はソート済みで返し、
   製品側のソートを消しても11件全部通った。
3. **期待値が実環境の既定値とたまたま一致する。** 実 `config.toml` のガンマが 2.2
   なので、期待値に 2.2 をハードコードしても19件全部通った。

**このうち3番目は本計画で特に危ない。** `gamut` の分類境界は 0 と 1 であり、
既定のターゲットもクリップも 0/1 に張り付いている。境界値そのものを踏むこと。

各タスクで、書いた各テストについて「どの誤実装なら通ってしまうか」を問うこと。
変異試験は形式ではない。実際に壊して、名前の付いたテストが落ちることを確認し、
落ちなければテストの方を直すこと。

---

## File Structure

| ファイル | 責任 |
| --- | --- |
| `packages/web/yoshidacomp_web/views/gamut.py` | `gamut` ビュー（新規） |
| `packages/web/yoshidacomp_web/views/__init__.py` | 登録 |
| `packages/web/yoshidacomp_web/static/js/contract.js` | `categorical` の traits |
| `packages/web/yoshidacomp_web/static/js/shaders.js` | 離散パレットの分岐 |
| `packages/web/yoshidacomp_web/static/js/glview.js` | `categorical` のアップロード |
| `packages/web/yoshidacomp_web/static/js/main.js` | 凡例・生成UI |
| `packages/web/yoshidacomp_web/static/index.html` | 凡例領域・生成欄 |
| `packages/web/yoshidacomp_web/static/style.css` | 凡例・生成欄 |
| `packages/web/yoshidacomp_web/generate.py` | 保存とメタデータ（新規） |
| `packages/web/yoshidacomp_web/routers/generate.py` | `POST /api/generate`（新規） |
| `tests/test_web_gamut_view.py` | 新規 |
| `tests/test_web_generate.py` | 新規 |
| `tests/js/frontend_harness.mjs` | `categorical` と生成のチェック |
| `tests/test_web_frontend_js.py` | 変異エントリ |

---

### Task 1: `gamut` ビュー（サーバ側）

**Files:**
- Create: `packages/web/yoshidacomp_web/views/gamut.py`
- Modify: `packages/web/yoshidacomp_web/views/__init__.py`
- Test: `tests/test_web_gamut_view.py`（新規）

**Interfaces:**
- Consumes: `views/compensation.py` を**先に読むこと。** ターゲットの解決、
  カメラ座標の拒否、コスト上限、有効領域マスク、パラメータの持ち方に
  すべて前例がある。`gamut` はその双子であり、違うのは最後の分類だけである。
  `pipeline.generate_projection_image(..., dtype=np.float32, clip=False)` で
  **クリップ前**の線形補償値を得る。
- Produces: `REGISTRY["gamut"]`。`kind="categorical"`、`channels=1`、
  `values_are_matrix=False`。

**分類**

クリップ前の線形補償値 `c` の3チャンネルに対し、画素ごとに:

| クラス | 条件 | ラベル |
| ---: | --- | --- |
| 0 | どのチャンネルも `0 <= c <= 1` | `範囲内` |
| 1 | いずれかが `c < 0`、かつ 1超は無い | `負クリップ` |
| 2 | いずれかが `c > 1`、かつ 負は無い | `1超クリップ` |
| 3 | 負と1超の両方がある | `両方` |

**境界は閉区間である。** ちょうど 0 とちょうど 1 は `範囲内` に入る。
`<` と `<=` を取り違えると、飽和した画素が大量に誤分類される
（実データでは白飛び付近が 1.0 ちょうどに張り付く）。

有効投影領域の外は **NaN**。クラス番号ではない。

**パラメータ**

`compensation` と同じ `target` / `target_decode` / `target_gamma` を持つ。
`linearizer` と `lut` と `output` は**持たない** — 色域外かどうかは線形補償値の
性質であり、その後の線形化とは無関係である。持たせると「LUTを変えると色域外が
変わる」という誤った印象を与える。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_gamut_view.py` を新規作成する。`tests/test_web_compensation_view.py`
の作法（フィクスチャ、クライアントの作り方、`ViewNotApplicable` の確認方法）に
合わせること。

検証すること:

1. **4クラスすべてが1枚の中に現れるフィクスチャで、各画素のクラスが正しいこと。**
   行列とターゲットを設計して、範囲内・負のみ・1超のみ・両方の画素を
   意図的に作ること。**期待値は `generate_projection_image(dtype=np.float32,
   clip=False)` をテスト内で直接呼び、そこから分類を再計算して導くこと。**
   手書きの定数にしない。
2. **境界がちょうど 0 と 1 の画素が `範囲内` になること。** 正確に 0.0 と 1.0 を
   出す行列とターゲットの組を作る。`<` と `<=` の取り違えを検出する唯一の手段である。
3. 有効領域の外が NaN であること（クラス番号 0 ではない）。
4. `labels` が4要素で、順序がクラス番号と一致すること。
5. カメラ座標の行列は `ViewNotApplicable`。
6. 未知の target id は `ViewNotApplicable`。
7. `ViewResult.data` が `(H, W, 1) float32` であること。
8. `vmin`/`vmax` が 0 と 3 であること（クラス番号の範囲）。

**判別力について**

- 行列を対称にしないこと。実データは非対称（`max|offdiag|/max|diag| ≈ 0.25`）。
- ターゲットを単色にしないこと。
- 縦横を違うサイズにすること。
- **クラス1と2が入れ替わっても通るテストになっていないか確認すること。**
  負クリップと1超クリップの画素数が同じフィクスチャだと入れ替えが見えない。

- [ ] **Step 2: 落ちることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_gamut_view.py -v`
Expected: FAIL

- [ ] **Step 3: 実装**

`views/compensation.py` の構造に倣う。分類は numpy のブールマスクで書く。

```python
below = (linear < 0.0).any(axis=2)
above = (linear > 1.0).any(axis=2)
classes = np.where(below & above, 3.0, np.where(above, 2.0, np.where(below, 1.0, 0.0)))
```

有効領域の外は分類の**後**に NaN で上書きする。先にやると `< 0` の比較で
NaN が False になり、領域外が黙って「範囲内」になる。

- [ ] **Step 4: 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_gamut_view.py -v`
Expected: PASS

- [ ] **Step 5: 変異試験**

1. `<= 0` にする → 境界のテストが落ちること
2. `>= 1` にする → 境界のテストが落ちること
3. クラス1と2を入れ替える → 分類のテストが落ちること
4. NaN の上書きを分類の前に移す → 領域外のテストが落ちること
5. `clip=False` を `clip=True` にする → 4クラスのテストが落ちること
   （クリップされたら1超も負も出なくなる）

それぞれ落ちたテスト名を報告に書くこと。**変異5が落ちないなら、
そのテストは色域外を検出していない。**

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/views/gamut.py packages/web/yoshidacomp_web/views/__init__.py tests/test_web_gamut_view.py
git commit -m "feat(web): クリップ前の補償値を4クラスに分ける gamut ビュー"
```

---

### Task 2: フロントの `categorical` 描画

**Files:**
- Modify: `static/js/contract.js`, `shaders.js`, `glview.js`, `main.js`
- Modify: `static/index.html`, `static/style.css`
- Test: `tests/js/frontend_harness.mjs`, `tests/test_web_frontend_js.py`

**Interfaces:**
- Consumes: Task 1 の `kind="categorical"`、`channels=1`、`labels` 4要素、
  `vmin=0` / `vmax=3`。
- Produces: `KIND_TRAITS.categorical`。`gamut` が画面に出る。

**`contract.js` が要求していること**

`KIND_TRAITS` の冒頭コメントは、kind を足すとは5つの判断すべてを書き下すこと
だと明示している。**一覧に足すだけにしないこと。** 埋める行:

| 項目 | `categorical` で何にすべきか |
| --- | --- |
| `singleTile` | 1枚（クラス番号は1チャンネル1枚の絵） |
| `sharedRange` | レンジ自体が意味を持たない（0〜3の固定） |
| `colormap` | 発散カラーマップは**無効**。離散パレットを使う |
| `packed` | 1プレーン |
| `texture` | R32F / RED |
| `uMode` | 新しい分岐番号（`shaders.js` に離散パレットを足す） |

**表示コントロールの扱い**

露出・γ・カラーマップはクラス番号に対して**意味を持たない**。
`0.5 EV` 明るいクラス番号 1 は存在しない。これらのコントロールを
そのまま効かせると、パレットの色が動いて凡例と食い違う。
**`categorical` のときは無効化するか、シェーダ側で無視すること。**
どちらにするか決めて理由をコメントに書くこと。

**NaN**

領域外は他のビューと同じマスク色にすること。`shaders.js` の
`isNaNBits`（IEEEビットパターン判定）を使う。**`isnan()` も `v != v` も
ANGLE/D3D11 では畳み込まれて消える**ため、素直に書くと領域外が
「もっともらしいクラス」として描かれる。

**凡例**

4色とラベルの対応を画面に出す。ラベルは `payload.labels` から取ること。
パレットをJSに、ラベルをHTMLに書くと2つの真実源になる。

- [ ] **Step 1: 失敗するテストを書く**

ハーネスに追加する。既存の書式に合わせること。検証すること:

1. `categorical` のペイロードが1枚のタイルとして描かれること
2. **アップロードが R32F / RED で行われること**（RGB32F だとクラス番号が
   赤成分だけの色になり、それらしく描けてしまう）
3. `uMode` が `categorical` 用の値で渡ること
4. 凡例が4項目になり、**ラベルが `payload.labels` 由来**であること
   （ハードコードした和文だと、サーバがラベルを変えても追随しない）
5. **露出とγを動かしても `categorical` の描画が変わらないこと**
   （あるいは無効化されていること。実装した方に合わせる）
6. 未知の kind は従来どおり明示的に失敗すること（既存の性質を壊さない）

- [ ] **Step 2: 落ちることを確認 → 実装 → 通ることを確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_frontend_js.py -v`

- [ ] **Step 3: 変異エントリを足す**

`_MUTATIONS` に3件。双方向の固定集合（`missing` と `unexpected`）を
既存エントリと同じ形で書き、**それぞれちょうど1つのチェックだけ**が落ちることを
確かめること。

1. `categorical` の `texture` を RGB32F にする
2. 凡例のラベルをハードコードする
3. `colormap: true` にする

- [ ] **Step 4: 実GPUで確認**

**スタブは見た目を検証しない。** headless Chrome（ANGLE/D3D11）で
実際にコンパイル・描画し、`gl.readPixels` で読み戻して
numpy ミラーと一致することを確認すること。特に:

- 4クラスが4つの異なる色になっていること（1つでも同じ色なら区別できない）
- 領域外がマスク色であること（`isNaNBits` が効いていること）

計画B1で、`isnan()` の畳み込みにより領域外が「もっともらしい強い負の係数」として
描かれていた不具合を、まさにこの方法で発見している。ブラウザを見るだけでは
気づけない。

- [ ] **Step 5: スクリーンショット**

実データで `gamut` を表示し、スクリーンショットを報告に貼ること。

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/static/ tests/js/frontend_harness.mjs tests/test_web_frontend_js.py
git commit -m "feat(web): 離散クラスを凡例つきで描く categorical kind"
```

---

### Task 3: `POST /api/generate`（サーバ側）

**Files:**
- Create: `packages/web/yoshidacomp_web/generate.py`, `routers/generate.py`
- Modify: `app.py`
- Test: `tests/test_web_generate.py`（新規）

**Interfaces:**
- Consumes: `pipeline.generate_projection_image()`、`store` の matrix / target / lut。
  **`examples/python/rebuild_compensation_with_map.py` の保存部分（130行目付近）と
  メタデータ（160行目付近）を先に読むこと。** 並びと項目をそれに合わせる。
- Produces: `POST /api/generate` — body に matrix_id / target_id / linearizer /
  lut_id / target_decode / target_gamma / output_dir。
  保存したパスの一覧とメタデータを返す。

**保存の並び**（CLI に合わせる）

```
<output_dir>/
├── linear/compensation_image_00.png       線形補償画像
├── projector/<suffix>_comp_image_00.png   投影画像（suffix は linearizer 由来）
└── generate_metadata.json
```

`suffix` は `{"none": "no_gamma", "gamma": "inv_gamma", "inverse_lut": "inv_lut"}`。
**この辞書は CLI 側にも同じものがある。** 片方だけ変えると出力名が食い違うので、
どちらか一方を輸入するか、**同じであることをテストで固定すること。**

**ディレクトリを渡された target の扱い**

`TargetContext` はディレクトリの場合に一覧を持つ。**全部生成するのか
選択中の1枚だけかを決めること。** CLI は全部生成する。GUIから叩くときに
100枚が走ると止まらないので、**上限を設けるか、選択中の1枚だけにするか、
理由とともに決めること。**

**上書きについて**

`output_dir` に既にファイルがある場合、**黙って上書きしないこと。**
既存を検出したら 409 を返し、`overwrite: true` を明示されたときだけ上書きする。
このGUIはユーザのローカルの `data/`（99GB、実測データ）を直接読み書きする。
取り違えたパスに書いて既存の計測結果を壊すのは回復できない。

- [ ] **Step 1: 失敗するテストを書く**

検証すること:

1. **保存された投影画像が `generate_projection_image()` の戻り値と
   バイト単位で一致すること。** 読み戻して比較する。
   期待値は関数を直接呼んで得ること。
2. 保存の並びとファイル名が上記のとおりであること
3. `linearizer` を変えると `suffix` が変わること
4. メタデータJSONに `linearizer` / `target_decode` / `active_region` /
   `matrix_shape` が入り、値が実際の入力と一致すること
5. **既存ファイルがあると 409 で、何も書かれていないこと。**
   `overwrite: true` なら上書きされること
6. 未知の matrix_id / target_id は 404
7. `linearizer="inverse_lut"` で lut_id 未指定は 400
8. **CLI の suffix 辞書と同じであること**

**判別力について**

- 画像を単色にしないこと。単色だと BGR/RGB の取り違えが見えない
  （`_save_rgb` は `cv2.cvtColor(..., COLOR_RGB2BGR)` を通す）。
  **R と B が違う値を持つ画像を使うこと。**
- 縦横を違うサイズにすること。
- 409 のテストは、**書かれていないこと**まで確認すること。
  ステータスコードだけ見ると、書いた上で 409 を返す実装でも通る。

- [ ] **Step 2: 落ちることを確認 → 実装 → 通ることを確認**

- [ ] **Step 3: 変異試験**

1. `cv2.cvtColor` を外す → RGB/BGR のテストが落ちること
2. 409 を返す前に書く → 「何も書かれていない」テストが落ちること
3. `suffix` の辞書の値を1つ変える → CLI一致のテストが落ちること
4. メタデータの `active_region` を実際と違う値にする → 4番が落ちること

- [ ] **Step 4: コミット**

```bash
git add packages/web/yoshidacomp_web/generate.py packages/web/yoshidacomp_web/routers/generate.py packages/web/yoshidacomp_web/app.py tests/test_web_generate.py
git commit -m "feat(web): 投影画像とメタデータをCLIと同じ並びで保存する"
```

---

### Task 4: 生成のUI

**Files:**
- Modify: `static/index.html`, `static/style.css`, `static/js/main.js`, `static/js/api.js`
- Test: `tests/js/frontend_harness.mjs`, `tests/test_web_frontend_js.py`

**Interfaces:**
- Consumes: Task 3 の `POST /api/generate`。
- Produces: なし（最後のタスク）。

**やること**

- サイドバーに出力先ディレクトリの入力欄と「生成」ボタンを置く。
- 押すと現在の行列・ターゲット・LUT・線形化方式で `POST /api/generate` を呼ぶ。
  **画面に出ているものと同じ設定で生成されること。** ここが食い違うと
  「見たものと違うものが保存される」という、この計画群が一貫して潰してきた
  種類の不具合になる。
- 生成中はボタンを無効化し、終わったら保存先のパスを表示する。
- 409 のときは「既に存在します」と出し、**上書きするかどうかを明示的に選ばせる**。
  黙って上書きしない。
- 失敗は既存の `errorMessage()` 経由で散文として出す。

**注意**

- ビューが `gamut` のときは `linearizer` パラメータが無い。生成は
  `compensation` の設定で行うのか、独立した選択を持つのか決めること。
  **決めた理由を報告に書くこと。**
- 生成は数秒かかる。ボタンの無効化が効いていないと二重送信になる。

- [ ] **Step 1: 失敗するテストを書く**

ハーネスに追加する。検証すること:

1. 生成ボタンが画面の設定を送ること（ターゲットid・LUTid・線形化方式）
2. 生成中はボタンが無効で、二重送信できないこと
3. 409 のとき上書きの確認が出て、**確認するまで再送しないこと**
4. 失敗が散文で出て、UIが操作不能にならないこと

**判別力について**: 「ボタンを押すと fetch が飛ぶ」だけでは、
**画面と違う設定を送る実装でも通る**。送られた body の中身を検証すること。

- [ ] **Step 2: 落ちることを確認 → 実装 → 通ることを確認**

- [ ] **Step 3: 変異エントリを足す**

`_MUTATIONS` に2件。双方向の固定集合で、ちょうど1つずつ落ちること。

1. 二重送信のガードを外す
2. 送る body のターゲットidを固定値にする

- [ ] **Step 4: 実物で確認**

実データで生成し、**保存されたファイルを CLI の出力と比較すること。**
`rebuild_compensation_with_map.py` を同じ入力で走らせ、
投影画像がバイト単位で一致することを確認して報告に書く。
一致しない場合は差の内訳（何画素・最大差・原因）を報告すること。

**注意**: `examples/python/sample.py` とは一致しない。`main()` が
`generate_projection_image()` を通らず独自の補償ループを持っているためで、
これは既知であり本計画の対象外である。

- [ ] **Step 5: スクリーンショット → コミット**

```bash
git add packages/web/yoshidacomp_web/static/ tests/js/frontend_harness.mjs tests/test_web_frontend_js.py
git commit -m "feat(web): 画面の設定のまま投影画像を生成する"
```

---

## Self-Review

**Spec coverage:** 仕様書 §5.1 の `gamut` を Task 1・2 が、§10「投影画像の生成」を
Task 3・4 が覆う。§12 の実装順序 8 に相当する。

**スコープ外（計画B5へ）:** カメラ座標行列のP2Cワープ（バックグラウンドジョブ、
`POST /api/matrix/{id}/warp`、`GET /api/job/{job_id}`）。1.56GB の行列と
`(12, 4660, 6984)` のテンソルを扱い、`prepare_target_image_for_compensation` が
初回に torch を引いて約7秒かかる問題（事前ウォームアップか停止許容かの判断）も
そちらで扱う。

**Placeholder scan:** Task 3 の「ディレクトリの target を全部生成するか1枚か」と
Task 4 の「生成の設定をどこから取るか」は意図的に実装者の判断へ委ねている。
どちらも実際のUIの形を見ないと決められず、計画側で決め打ちすると
誤った前提を固定する。理由を報告に書くことを求めている。

**Type consistency:** `gamut` は `channels=1`、`kind="categorical"`、
`vmin=0`/`vmax=3`、`labels` 4要素。`KIND_TRAITS.categorical` の
`texture` は R32F/RED で、`channels=1` と整合する。
`generate.py` の `suffix` 辞書は `rebuild_compensation_with_map.py` と同一。
