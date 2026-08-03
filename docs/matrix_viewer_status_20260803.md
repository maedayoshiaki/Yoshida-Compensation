# 色変換行列ビューア 現況と再開の手引き（2026-08-03）

`main` = `6ed2ed1`（`origin/main` と同期済み）/ **394 passed**

仕様書 [`docs/superpowers/specs/2026-07-31-matrix-viewer-gui-design.md`](superpowers/specs/2026-07-31-matrix-viewer-gui-design.md)
の §12「実装順序」のうち、**残っているのは項目9（カメラ座標行列のワープ）だけ**である。

---

## 1. すぐ動かす

```bash
uv sync --extra sample --extra web        # --extra を必ず付ける（後述の罠）
uv run --no-sync python -m uvicorn yoshidacomp_web.app:app --host 127.0.0.1 --port 8123
```

ブラウザで `http://127.0.0.1:8123`。行列のパス欄に例えば
`data/linear_pipeline_1080/matrices_linear_gamma.npy` を入れて「読み込み」。

ビュー選択で4つが切り替わる。`補償画像` と `色域外` はサイドバーの
「ターゲット画像」欄に画像かディレクトリのパスを入れて「読み込み」すると動く
（例: `data/comp_test_targets`）。逆LUTは `.npz` のパスを同様に
（例: `data/linear_pipeline_1080_prev_target/inverse_lut.npz`）。

| 画面 | ファイル |
| --- | --- |
| 12係数タイル | ![](images/matrix_viewer_coef_tiles_20260802.png) |
| 色域外 | ![](images/gamut_view_20260803.png) |
| 生成 | ![](images/generate_from_gui_20260803.png) |

---

## 2. いま何があるか

### パッケージ

```
packages/core/yoshidacomp/     計算（GUI非依存、CLIと共有）
packages/web/yoshidacomp_web/  FastAPI + 素のESモジュール + WebGL2
examples/python/               薄いCLI
```

`yoshidacomp` は `yoshidacomp_web` を import しない（`tests/test_package_boundaries.py` が固定）。

### エンドポイント

| メソッド | パス | 内容 |
| --- | --- | --- |
| POST | `/api/session/matrix` | `.npy` を読み込み行列IDを発行 |
| GET | `/api/matrix/{id}/meta` | 形状・座標空間・有効領域 |
| GET | `/api/matrix/{id}/probe` | 1画素の4×3係数（JSON、12個のfloatのみ） |
| GET | `/api/views` | ビュー一覧とパラメータ定義 |
| GET | `/api/matrix/{id}/view/{key}` | ビューのfloat32バイナリ |
| POST | `/api/target` | ターゲット画像（パス指定、アップロード無し） |
| GET | `/api/target/{id}/meta` | 形状・一覧・選択中index |
| POST | `/api/lut` | 逆LUT `.npz` |
| GET | `/api/lut/{id}/curve` | LUTカーブ `(3, 256)` と `t_grid` |
| POST | `/api/generate` | 投影画像とメタデータJSONを保存 |

**未実装（計画B5）**: `POST /api/matrix/{id}/warp`、`GET /api/job/{job_id}`。

### ビュー

| key | kind | 内容 | 主なパラメータ |
| --- | --- | --- | --- |
| `coef_tiles` | scalar×12 | 4行(in_R/in_G/in_B/bias)×3列(out_R/out_G/out_B) | 正規化方法 |
| `surface` | rgb | 投影面の色 `C = (P − b)·A⁻¹` | 投影コマンド `P` |
| `compensation` | rgb | ターゲットに行列を適用した結果 | target / linearizer / lut / output / decode |
| `gamut` | categorical | 色域外の4クラス | target / decode |

ビューの追加は `packages/web/yoshidacomp_web/views/` に1ファイル置いて `register` するだけ。
`kind` がフロント実装済みなら JavaScript の変更は不要。

### フロントの `kind`

`static/js/contract.js` の `KIND_TRAITS` に `scalar` / `rgb` / `categorical` の3行。
**未登録の `kind` は scalar として描かれず、明示的に失敗する。**
kind を足すとは、この表の全列（`singleTile` / `sharedRange` / `colormap` /
`toneChain` / `packed` / `texture` / `uMode` / `labelsAreClasses` ほか）を
書き下すことである。冒頭のコメントがその理由を説明している。

---

## 3. 計画ごとの成果（すべて `main` にマージ済み）

| 計画 | 内容 | テスト | 計画書 |
| --- | --- | ---: | --- |
| A | パッケージ分割 | 132 | [package-split](superpowers/plans/2026-07-31-package-split.md) |
| B1 | WebGL2の縦スライス | 132 | [web-viewer-slice](superpowers/plans/2026-07-31-web-viewer-slice.md) |
| B2 | planar転送・`surface` | 162 | [planar-and-rgb-views](superpowers/plans/2026-08-01-planar-and-rgb-views.md) |
| C | 点クエリの読み取り・レイアウト圧縮 | 196 | [matrix-readout-everywhere](superpowers/plans/2026-08-01-matrix-readout-everywhere.md) |
| B3 | ターゲット・`compensation`・LUT | 293 | [target-and-lut](superpowers/plans/2026-08-02-target-and-lut.md) |
| B4 | `gamut`・生成 | **394** | [gamut-and-generate](superpowers/plans/2026-08-02-gamut-and-generate.md) |

各計画書の末尾に「完了記録と申し送り」がある。**次の作業を始める前に、
直前の計画の申し送りを読むこと。**

---

## 4. 次の作業（計画B5）

**カメラ座標行列のP2Cワープをバックグラウンドジョブとして実装する。**
これで `data/color_mixing_matrices.npy`（1.56GB、カメラ座標）を
プロジェクタ座標で表示できるようになる。

### 着手前に知っておくべきこと

**1. `prepare_target_image_for_compensation` は初回呼び出しで約7秒かかる。**
`geometry.warp` 経由で torch を引く。GrayCode の `PixelMapWarperTorch` による
2M点の散布ワープで、計画B3で numpy 化した「12回の積和」とは規模が違うため
意図的に torch のまま残してある。**web層へ配線する際に、起動時の事前
ウォームアップにするか初回の停止を許容するかを判断すること。**「もう解決済み」ではない。

現在の状態（実測）:

| | 値 |
| --- | ---: |
| `import yoshidacomp.compensation.pipeline` | 0.29 s（torch/psutil/scipy を引かない） |
| `generate_projection_image` 初回 | 0.015 s |
| `prepare_target_image_for_compensation` 初回 | **約7 s**（torch を引く） |

**2. メモリに余裕が無い。** `(12, 4660, 6984)` のテンソルを扱う。
計画B1の実測では 1.56GB の行列を読むだけで 4.19GB ヒープ中 2.98GB を使った。
ダウンサンプリングかタイル単位の遅延取得を検討すること。

**3. GUI↔CLI のバイト一致を取り直すこと。** 計画B4で確認した一致は、
GUI がワープしないため **CLI 内部のワープ済み行列を手で食わせた**比較である。
ブラウザ→body→ディスクの区間とファイル番号は証明しているが、ワープ区間は
証明していない。ワープを配線したら本物の比較を取ること。

**4. 検証用の中間ファイルは `data/` に置かない。** 計画B4の実装者が
`data/task4_warped_matrix.npy`（95MB）などを作った。`data/` は99GBの実測データで
再生成できない。スクラッチパッドへ書くこと。

**5. 現在カメラ座標の行列は明示的に断られる。** `views/surface.py` /
`views/compensation.py` / `views/gamut.py` / `routers/generate.py` がいずれも
400 を返し、拒否文が「B5で対応予定」と述べている。B5 で配線したらその文言も直すこと。

---

## 5. 環境の罠（毎回踏むので必ず読む）

**`uv sync` は必ず `--extra sample --extra web` を付ける。**
素の `uv sync` は `edsdk` と `rawpy` をアンインストールし、手で置いた
`EDSDK.dll` / `EdsImage.dll` を `.venv/Lib/site-packages/edsdk/` から削除する。
壊した場合の復旧:

```bash
cp C:/py_scripts/edsdk-python/dependencies/EDSDK_64/Dll/*.dll .venv/Lib/site-packages/edsdk/
```

**`.venv` を消さないこと。** 復旧できない場合は作業を止めて報告すること。

**ポート8000は別プロセスが使用中。** `python -m yoshidacomp_web` はポートを
8000 に固定しているので、uvicorn を直接呼んで `--port 8123` を使う。
**古いサーバが残っていないか確認すること** — 計画B4で、前セッションの
サーバが修正前のコードを配信していて計測が狂った事例がある。

**`.pytest_cache/v/cache/` はOSレベルで書き込めない。** 警告が出るが無害。

**変異試験の復元に `git checkout --` を使わないこと。** 未コミットの作業があると
巻き添えで消える。計画B4で実際に起きた。ファイルバックアップを使う。

---

## 6. テストについて（このリポジトリ固有の事情）

**「通るが落ちることができないテスト」がこれまでに26件出ている。**
緑であることは、テストが何かを守っている証拠にならない。

### 繰り返し現れる3つの形

1. **フィクスチャが検証対象の次元について対称。**
   実データの行列は非対称（`max|offdiag|/max|diag| ≈ 0.25`）だが、テストが `2·I` を
   使っていたため逆行列の転置が14テスト全部を通った。
2. **性質を散文で自信を持って述べたことで検証済みに見える。**
   「ディスク上でわざとアルファベット順を崩している」というコメントに反して
   NTFS はソート済みで返し、製品側のソートを消しても11件全部通った。
3. **期待値が実環境の既定値とたまたま一致する。**
   実 `config.toml` のガンマが 2.2 なので、2.2 をハードコードしても19件通った。

### 特に高くついた2件

**シェーダの `categorical` 分岐（計画B4、Critical）。** `isNaNBits` を消しても、
トーンチェーンを足しても、`u_mode == 2` の分岐を削除しても324件が緑だった。
実装者は実機GPUから画素を読み戻して `isNaNBits` が必要であることを**証明した**が、
その実行は手作業でCIには乗っていなかった。**「証明されている」と「守られている」は
別の状態で、その差は緑の結果に現れない。** 消すと実フレームの75%（未照明の周囲）が
「範囲内」と報告される。

**`isnan()` と `v != v` は ANGLE/D3D11 で両方とも畳み込まれて消える。**
このマシン（RTX 3060 Ti）で3回再現している。NaN判定は必ず IEEE ビットパターン
（`isNaNBits`、`highp uint` 必須）で書くこと。`shaders.js` にある。

### 運用

- 各テストについて「どの誤実装なら通ってしまうか」を問う。
- 変異試験は形式ではない。実際に壊して、名前の付いたテストが落ちることを確認する。
  落ちなければテストの方を直す。
- `tests/test_web_frontend_js.py` の `_MUTATIONS` は、各変異が
  **ちょうど1つのチェックだけ**を落とすことをCIで強制する（双方向の固定集合）。
  Python 側には同等の仕組みが無く、変異はすべて手動である（未対処）。
- レビューには「N+1件目があると仮定して探せ。生き残るのは主題が自明すぎて
  確認されないもの」と指示すると実際に見つかる。

---

## 7. 実データについて測って分かったこと

| 事実 | 値 |
| --- | --- |
| 実フレームの色域外率 | **46.5〜85.0%**（実行列4×実写3の12組） |
| `matrices_linear_gamma.npy` の非有限値 | **0個**（領域外74.86%も全て有限） |
| クリップ前にちょうど0.0/1.0になる値 | 3317万中 **0個と2個**（clip=Trueなら786万/845万） |
| numpy vs torch-CPU（uint8、622万画素） | **0個の差** |
| numpy vs torch-CUDA | 9個（CUDA自身がCPUと食い違う位置・大きさと完全一致） |
| 逆LUTの終端（ISO400） | 約 0.876 / 0.876 / 0.688（**1.0ではない**） |

**マスク色（マゼンタ）は NaN を意味する。** `coef_tiles` / `surface` /
`compensation` / `gamut` すべてが有効投影領域の外を NaN にする規約で、
読み取り欄の「有効投影領域の外」マーカーもこれに依存している。

`gamut` の画面で領域外がマゼンタなのはこの規約によるもので、
「クラス0＝範囲内」ではない。

---

## 8. 関連ドキュメント

- [`docs/web_viewer_performance_20260731.md`](web_viewer_performance_20260731.md) — 計画B1/B2の性能実測
- [`docs/current_color_compensation_pipeline.md`](current_color_compensation_pipeline.md) — 補償パイプラインそのものの説明
- [`docs/inverse_lut_説明.md`](inverse_lut_説明.md) — 逆LUTの原理
- `README.md` の「色変換行列ビューア」節 — 最短の起動手順
