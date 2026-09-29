# AGENTS.md — 作業を引き継ぐエージェントへ

最終更新: **2026-08-05** / git `main` = `4d2b438`（`origin/main` と同期済み）

この文書はリポジトリの入口である。**作業を始める前に第1章から第4章までを読むこと。**
詳細は各章末のリンク先にある。リンク先の方が新しい場合は**リンク先が正本**である。

> **いま working tree は clean ではない。** 2026-08-04〜05 の実機作業が
> 6ファイル分の未コミット変更として残っている。第6章を読まずに
> `git checkout --` / `git stash` / `git clean` を実行しないこと。

---

## 1. このリポジトリは何か

プロジェクタ-カメラ系の**測光補償（photometric compensation）**の研究ワークスペース。

プロジェクタから既知のパターンを投影し、カメラで撮影し、**画素ごとの 4×3 カラーミキシング
行列**（入力 R/G/B の3行 + バイアス1行 × 出力 R/G/B の3列）を推定する。この行列を使って、
目標画像 `T` が投影面に現れるような投影画像 `P` を逆算する。順方向モデルは

```
C = (P − b) · A⁻¹        （C = 投影面に見える色、P = 投影コード値、b = 環境光バイアス）
```

主要な構成要素は4つ。

| 要素 | 内容 |
| --- | --- |
| 幾何 | GrayCode 由来の P2C / C2P 対応マップ。カメラ座標 ⇔ プロジェクタ座標 |
| 測光 | 64色（`num_divisions=4`）の校正パターン撮影 → 画素ごとの行列 fit |
| 線形化 | プロジェクタのガンマを打ち消す4方式（下表） |
| 評価 | PSNR / SSIM / ΔE76 による条件間比較 |

線形化の4方式（**混在不可、1回の実行で1方式**）:

1. なし（`comp_no_gamma`）
2. 解析逆ガンマ（`comp_gamma`、`projector.gamma = 2.2`）
3. 実測逆LUT・チャンネルランプ（`comp_inverse_lut`、`inverse_lut.source = "channel_ramp"`）
4. 実測逆LUT・グレー投影 fit（`comp_inverse_lut_gray`、`source = "gray"`）

パイプライン全体の挙動は **[`docs/current_color_compensation_pipeline.md`](docs/current_color_compensation_pipeline.md)**
に節ごとに書いてある。実装を読む前にこれを読むこと。

---

## 2. 現在地 — 2つの作業ストリーム

このリポジトリには**独立した2本の作業**が並行している。混同しないこと。

### ストリーム A: 色変換行列ビューア（GUI）

**状態: `main` にすべてマージ済み。残るは計画B5（カメラ座標行列のP2Cワープ）だけ。**

ブラウザで色変換行列を見るオフライン専用ビューア。FastAPI + 素のESモジュール + WebGL2。
計画A / B1 / B2 / C / B3 / B4 が完了し、テストは 394 件。→ **第5章**

正本: [`docs/matrix_viewer_status_20260803.md`](docs/matrix_viewer_status_20260803.md)

### ストリーム B: 実機の再校正と色補償の再評価（2026-08-04〜05）

**状態: 未コミット。進行中。この会話の直前（本日 13:32）に最新の実行が終わっている。**

非HDR（単一露光）での校正へ切り替え、EDSDK のRAW現像クラッシュを回避し、
幾何を再校正し、最後に **GrayCode の復号結果から投影ROIを自動導出**するところまで来ている。
導出した新ROI `(580, 217, 784, 784)` は**まだどの config でも使われていない**。→ **第6章**

正本: この文書の第6章（他に記録が無い）

---

## 3. 環境と実行 — 毎回踏むので必ず読む

- Windows 11 Pro / PowerShell（Bash も可、構文が別） / Python 3.13 / uv ワークスペース
- CI は無い。`.github/` にあるのはコミットメッセージ規約だけ。**検証はすべてローカル。**

### 3.1 `uv sync` は必ず `--extra` を付ける

```bash
uv sync --extra sample --extra web
```

**素の `uv sync` は `edsdk` と `rawpy` をアンインストールし、手で置いた
`EDSDK.dll` / `EdsImage.dll` を `.venv/Lib/site-packages/edsdk/` から削除する。**
壊した場合の復旧:

```bash
cp C:/py_scripts/edsdk-python/dependencies/EDSDK_64/Dll/*.dll .venv/Lib/site-packages/edsdk/
```

**`.venv` を消さないこと。** 復旧できない場合は作業を止めて報告すること。

依存を変更しないなら `uv run --no-sync ...` を使うと sync 自体を回避できる。

### 3.2 テスト

```bash
uv run --no-sync python -m pytest -q
```

`tests/` のみが対象（`pyproject.toml` の `testpaths`）。JS フロントの検査は
`tests/test_web_frontend_js.py` が `tests/js/frontend_harness.mjs` を呼ぶ形で Python から走る。

**2026-08-05 に実測: 394 passed / 1 warning / 215.89 s（約3分36秒）。**
第6章の未コミット変更を含んだ状態で緑である。これは変更が安全だという意味ではなく、
**その変更にテストが1件も無い**という意味である（6.4参照）。

`.pytest_cache/v/cache/` はこの環境ではOSレベルで書き込めない。警告が出るが無害。

### 3.3 ビューアの起動

```bash
uv run --no-sync python -m uvicorn yoshidacomp_web.app:app --host 127.0.0.1 --port 8123
```

**ポート 8000 は別プロセスが使用中。** `python -m yoshidacomp_web` はポートを 8000 に
固定しているので使えない（`[WinError 10013]` で落ちる）。uvicorn を直接呼ぶこと。
使用中のプロセスは `Get-NetTCPConnection -LocalPort 8000` で確認できる。

**古いサーバが残っていないか確認すること。** 計画B4で、前セッションのサーバが修正前の
コードを配信していて計測が狂った事例がある。

### 3.4 実機撮影（Canon EDSDK）

`config.toml` の `camera.edsdk_dll_dir` に DLL のディレクトリを書く（現在は
`C:/py_scripts/edsdk-python/dependencies/EDSDK_64/Dll`）。環境変数 `EDSDK_PYTHON_DLL_DIR` でも可。

撮影が失敗したら、まずカメラの電源を入れ直し、EOS Utility など他の Canon アプリを閉じ、
モードダイヤルを Manual/Av/Tv にして再試行する。

`half_size` は**必ず false**。半解像度RAWはバックエンドでも拒否する。

---

## 4. 構造

```
packages/core/yoshidacomp/     計算コア（GUI非依存、CLIと共有）
packages/web/yoshidacomp_web/  FastAPI + 素のESモジュール + WebGL2
examples/python/               薄いCLI（実機撮影を含む）
tests/                         pytest
external/GrayCode/             サブモジュール（幾何復号）
data/                          入力・生成物・実験記録（約99GB、大半はgit管理外）
docs/                          報告書・仕様・計画
```

`yoshidacomp` は `yoshidacomp_web` を import しない。`tests/test_package_boundaries.py`
がこの境界を固定している。`src/python/` は廃止済みで存在しない。

### 4.1 コアのモジュール

| モジュール | 役割 |
| --- | --- |
| `config.py` | `config.toml` のローダー。`get_config()` / `reload_config(path)` |
| `capture/` | カメラ抽象層（`base` / `factory`）と実装（`canon_edsdk` / `opencv`） |
| `geometry/warp.py` | 幾何ワープ（c2p / p2c）。GrayCode の `PixelMapWarperTorch` を使う |
| `geometry/graycode_path.py` | サブモジュールへのパス解決 |
| `compensation/color_mixing_matrix.py` | パターン生成・逆ガンマ・行列 fit |
| `compensation/photometric_compensation.py` | 補償画像の計算 |
| `compensation/projection_region.py` | `resolve_active_region` / `mask_to_active_region` |
| `compensation/inverse_lut.py` | 実測逆LUTの計測・構築・適用 |
| `compensation/pipeline.py` | 生成パイプライン一式（`generate_projection_image` ほか） |
| `calibration/` | チェッカーボード撮影・PnP |

### 4.2 CLI（`examples/python/`）

| スクリプト | 用途 |
| --- | --- |
| `sample.py` | 補償パイプラインの通し実行（解析逆ガンマ） |
| `sample_inverse_lut.py` | 同上の実測逆LUT版 |
| `capture.py` | 手動キャプチャ＋ワープ |
| `build_inverse_lut.py` | 実測逆LUTの構築 |
| `validate_inverse_lut.py` | 逆LUTが実際に線形化しているかの検証（matplotlib が要る） |
| `calibrate_graycode_geometry.py` | GrayCode で密なP2Cマップを作る |
| `calibrate_color_matrix.py` | 校正撮影 → 画素ごとの色行列 fit |
| `rebuild_compensation_with_map.py` | **幾何だけ**変えたときに再撮影せず補償画像を作り直す |
| `evaluate_compensation_conditions.py` | 4条件または5条件の実写比較（PSNR/SSIM/ΔE） |
| `evaluate_recalibrated_compensation.py` | 補償なし vs 新規校正1条件の比較 |
| `check_projection_exposure.py` | 黒/白を投影して線形RAWの露出を測る |
| `capture_grayscale_levels.py` | 4段階グレーをフル解像度RAWで撮る |
| `verify_warp_geometry.py` | チェッカーを投影してP2C/C2Pの整合を検証 |
| `verify_hdr_tv.py` | Canon の Tv 切替が HDR ブラケットで通るかの実機確認 |
| `project_and_capture.py` | 固定画像を投影して任意タイミングで撮影 |
| `edsdk_error.py` | EDSDK エラーの再現用 |

すべて `--config <path>`（`-c`）で設定ファイルを差し替えられる。

### 4.3 `data/` の約束

**`data/` に検証用の中間ファイルを置かないこと。** 実測99GBが入っており再生成できない。
一時ファイルはスクラッチパッドへ書く。計画B4の実装者が `data/task4_warped_matrix.npy`（95MB）
などを作ってしまった前例がある。

ディレクトリごとの索引は [`data/README.md`](data/README.md) にある。
**2026-08-04/05 に増えた5ディレクトリはまだこの索引に載っていない**（第6章に一覧がある）。

---

## 5. ストリーム A: 色変換行列ビューア

**正本は [`docs/matrix_viewer_status_20260803.md`](docs/matrix_viewer_status_20260803.md)。
B5に着手するならその第4章「次の作業」を必ず読むこと。** ここでは要約だけ置く。

### 5.1 済んでいること

| 計画 | 内容 | テスト |
| --- | --- | ---: |
| A | パッケージ分割 | 132 |
| B1 | WebGL2の縦スライス（`coef_tiles`） | 132 |
| B2 | planar転送・`surface` | 162 |
| C | 点クエリの読み取り・レイアウト圧縮 | 196 |
| B3 | ターゲット・`compensation`・LUT | 293 |
| B4 | `gamut`・投影画像の生成 | 394 |

各計画書は `docs/superpowers/plans/` にあり、**末尾に「完了記録と申し送り」がある。
次の作業を始める前に直前の計画の申し送りを読むこと。**

### 5.2 エンドポイント

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

**未実装（B5）**: `POST /api/matrix/{id}/warp`、`GET /api/job/{job_id}`。

### 5.3 ビューと拡張の作法

| key | kind | 内容 |
| --- | --- | --- |
| `coef_tiles` | scalar×12 | 4行(in_R/in_G/in_B/bias)×3列(out_R/out_G/out_B) |
| `surface` | rgb | 投影面の色 `C = (P − b)·A⁻¹` |
| `compensation` | rgb | ターゲットに行列を適用した結果 |
| `gamut` | categorical | クリップ前の値を4クラスに分けた色域外表示 |

ビューの追加は `packages/web/yoshidacomp_web/views/` に1ファイル置いて `register` するだけ。
`kind` がフロント実装済みなら JavaScript の変更は不要。

`static/js/contract.js` の `KIND_TRAITS` に `scalar` / `rgb` / `categorical` の3行がある。
**未登録の `kind` は scalar として描かれず、明示的に失敗する。** kind を足すとは、この表の
全列（`singleTile` / `sharedRange` / `colormap` / `toneChain` / `packed` / `texture` /
`uMode` / `labelsAreClasses` ほか）を書き下すことである。

### 5.4 次の作業（B5）に入る前に知っておくべきこと

1. **`prepare_target_image_for_compensation` は初回呼び出しで約7秒かかる。**
   `geometry.warp` 経由で torch を引く（GrayCode の `PixelMapWarperTorch` による2M点の
   散布ワープ）。**web層へ配線する際に、起動時ウォームアップにするか初回の停止を許容するかを
   判断すること。「もう解決済み」ではない。**

   | | 値 |
   | --- | ---: |
   | `import yoshidacomp.compensation.pipeline` | 0.29 s |
   | `generate_projection_image` 初回 | 0.015 s |
   | `prepare_target_image_for_compensation` 初回 | **約7 s** |

2. **メモリに余裕が無い。** `(12, 4660, 6984)` を扱う。1.56GB の行列を読むだけで
   4.19GBヒープ中 2.98GB を使った（B1実測、読み込み18.2秒）。ダウンサンプリングか
   タイル単位の遅延取得を検討すること。

3. **GUI↔CLI のバイト一致を取り直すこと。** B4で確認した一致は、GUI がワープしないため
   **CLI 内部のワープ済み行列を手で食わせた**比較である。ワープ区間は証明していない。

4. **現在カメラ座標の行列は明示的に断られる。** `views/surface.py` / `views/compensation.py` /
   `views/gamut.py` / `routers/generate.py` が 400 を返し、拒否文が「B5で対応予定」と述べている。
   配線したらその文言も直すこと。

仕様書は [`docs/superpowers/specs/2026-07-31-matrix-viewer-gui-design.md`](docs/superpowers/specs/2026-07-31-matrix-viewer-gui-design.md)。
§12 が実装順序（項目9だけが未着手）、§13 が未決事項（生成先の既定値、複数行列の同時比較）。

性能実測は [`docs/web_viewer_performance_20260731.md`](docs/web_viewer_performance_20260731.md)。

---

## 6. ストリーム B: 実機の再校正（未コミット・進行中）

**この章がこの作業の唯一の記録である。** `docs/` にも計画書にも書かれていない。

### 6.1 未コミットの変更（`git status`）

```
 M examples/python/calibrate_color_matrix.py
 M examples/python/calibrate_graycode_geometry.py
 M examples/python/capture_grayscale_levels.py
 M examples/python/check_projection_exposure.py
 M packages/core/yoshidacomp/capture/canon_edsdk.py
 M packages/core/yoshidacomp/compensation/inverse_lut.py
?? config.iso100_nonhdr_20260804.toml
?? config.iso160_nonhdr_20260804.toml
?? config.iso160_nonhdr_20260805_geom.toml
?? data/iso100_nonhdr_20260804/            （計測JSON・調査スクリプト3本）
?? data/iso160_nonhdr_20260804/            （計測JSON）
?? data/iso160_nonhdr_20260805_geom/       （計測JSON）
?? data/graycode_recalibration_20260805_window4/
?? data/graycode_recalibration_20260805_window4_roi_redetermined/
```

`data/` 配下の `.npy` / `.png` は `.gitignore` により追跡されない。上に出るのは JSON と
スクリプトだけである。

### 6.2 変更の内容と理由（ファイル単位）

#### `examples/python/calibrate_color_matrix.py`（08-04 14:18）
- `cfg.camera.hdr_enabled` が true でなければ落とす検査を**削除**。非HDR（単一露光）での
  校正を許可した。
- 保存名を `hdr_capture_XX.png` → `capture_XX.png` に変更。docstring とエラー文からも
  "HDR" を除いた。

#### `examples/python/capture_grayscale_levels.py`、`check_projection_exposure.py`（08-04 14:16 / 14:34）
- **バグ修正**: `cfg.inverse_lut.hdr_trust_low` / `hdr_trust_high` を参照していたが、
  設定側の属性名は `trust_low` / `trust_high` である。`AttributeError` になっていた。
- **注意**: 出力JSONのキー名は `hdr_trust_low` / `hdr_trust_high` のまま。値だけ正しくなった。
  過去の計測JSONを比較するときはこのキー名が両方の時代にまたがることを踏まえること。
- `check_projection_exposure.py` に `--full-frame` を追加。`active_region` でマスクせず
  全画面白を投影する。

#### `packages/core/yoshidacomp/capture/canon_edsdk.py`（08-04 14:38）
2つの回避策が入っている。**どちらもテストが無い。**

1. **スプール先の環境変数上書き**: `YOSHIDA_EDSDK_TRIGGER_SPOOL_DIR` を設定すると、
   `tempfile.TemporaryDirectory` を使わずそのディレクトリへ直接書く。ファイル名には
   `{pid}-{id(self):x}-` のプレフィックスが付き、`__exit__` でグロブして削除する。
   理由: 管理されたWindows環境では、プロセスtempルート自体は書けても、その配下に
   新規作成したディレクトリへの書き込みが拒否されることがある。
2. **RAW現像の子プロセス隔離**: `_develop_raw_to_npy_worker` を `spawn` の子プロセスで
   走らせ、linear RGB を `.npy` 経由で受け取る。適用条件は
   **controller が本物の `edsdk` パッケージ由来** かつ **拡張子が `.cr2` / `.cr3` / `.raw`** のときだけ。
   失敗時は traceback をファイル経由で受け取り `CameraCaptureError` にする。
   理由: フル解像度RAWを連続現像するとネイティブRAWデコーダがプロセスごと落ちる。

#### `packages/core/yoshidacomp/compensation/inverse_lut.py`（08-04 14:38）
1. `run_calibration_capture` の現像ループで `del frame; gc.collect()` を追加。現像フレームは
   フル解像度で約390MB あり、2枚同時に生存するとネイティブRAWデコーダが落ちる。
2. `capture_chunked`（`calibration_chunk_size = 4`）を追加。**しかしこの関数はどこからも
   呼ばれていない。** 呼び出し側3箇所（1172 / 1215 / 1246行）はすべて `capture_pass` のまま。
   **未完了の変更である。** 意図どおり分割撮影にするなら配線が要る。

#### `examples/python/calibrate_graycode_geometry.py`（08-05 13:18 — 最新の変更）
新しいオプション3つ。

| オプション | 内容 |
| --- | --- |
| `--full-frame` | `active_region` を無視して全画面に GrayCode を投影する |
| `--derive-roi` | 復号できた GrayCode 有効領域から正方形の投影ROIを自動導出する（全画面投影を含む） |
| `--roi-inset` | 導出時の内側マージン（プロジェクタ画素、既定 8） |

`_derive_square_roi` の手順:
1. `c2p` のプロジェクタ座標を丸めて有効マスクを作る
2. 7×7 の `MORPH_CLOSE`
3. `connectedComponentsWithStats` で最大連結成分を取る（さらに `MORPH_CLOSE`）
4. `distanceTransform` の最大値の位置＝中心、その値＝内接円半径
5. `半径 − inset` を半辺とする正方形。半辺 < 32 なら失敗、フレームに収まらなければ失敗

出力: `decoded_projector_valid_mask.png` / `derived_roi_preview.png` / `derived_active_region.json`。
`metadata.json` に `configured_active_region` / `projected_full_frame` /
`roi_derived_from_decoded_graycode` / `roi_inset` / `decoded_roi_stats` が追加される。

**この5つのフィールドが metadata.json に無い実行は、この変更より前のものである**（版の見分けに使える）。

### 6.3 実行の時系列と実測結果

| 時刻 | 何をしたか | 出力 |
| --- | --- | --- |
| 08-04 14:16〜14:38 | 上記のコード修正（幾何オプションを除く） | — |
| 08-04 14:49〜14:54 | 投影ウィンドウ位置の探索、ISO100 の露出確認 | `data/iso100_nonhdr_20260804/` |
| 08-04 14:55〜16:32 | ISO160 非HDR の5条件フル評価（**旧 p2c マップ**） | `data/iso160_nonhdr_20260804/` |
| 08-05 11:41〜11:54 | GrayCode 幾何を再校正（ROI は手動指定のまま） | `data/graycode_recalibration_20260805_window4/` |
| 08-05 〜13:05 | 新しい幾何で5条件フル評価をやり直し | `data/iso160_nonhdr_20260805_geom/` |
| 08-05 13:18 | `--full-frame` / `--derive-roi` を実装 | — |
| 08-05 〜13:32 | 全画面投影してROIを自動導出 | `data/graycode_recalibration_20260805_window4_roi_redetermined/` |

#### 投影ウィンドウの位置（`data/iso100_nonhdr_20260804/window_probe.json`）

5箇所を投影して赤−黒の応答を測った結果、**応答があったのは `(3440, 0)` だけ**である。

| 位置 | 赤−黒の平均（ROI内） | 黒の平均 |
| --- | ---: | ---: |
| `(0,0)` / `(-1920,0)` / `(-5120,0)` / `(-3200,0)` | 0.004〜0.006 | 0.034〜0.057 |
| **`(3440, 0)`** | **0.0889 / 0.0023 / 0.0030** | **0.0002** |

黒レベルが2桁違う（0.055 → 0.0002）ことからも、`(3440,0)` 以外はデスクトップ画面を
撮っていたと分かる。`data/iso100_nonhdr_20260804/exposure_check_window5_actual/exposure_check.json`
では黒 0.221 と白 0.219 がほぼ同値で、投影応答がまったく無い。
**「ディスプレイ5」という呼び名は Win32 の `\\.\DISPLAY5` を指さない。** 第7章参照。

#### 5条件評価の比較 — 幾何の鮮度で結果が変わる

同じ ISO160 / av=14 / tv=1/15 / 非HDR、同じターゲット5枚（`data/comp_test_targets_square`）、
同じ ROI `(613,201,722,722)` で、**P2Cマップだけが違う**2回の実行の平均値。

| 条件 | 08-04（旧 p2c: `graycode_recalibration_20260730`） | 08-05（新 p2c: `..._20260805_window4`） |
| --- | --- | --- |
| | PSNR / SSIM / ΔE76 | PSNR / SSIM / ΔE76 |
| `no_comp` | 10.17 / **0.253** / 38.76 | 12.20 / **0.694** / 31.55 |
| `comp_no_gamma` | 10.17 / 0.261 / 38.05 | 13.76 / 0.679 / 21.18 |
| `comp_gamma` | 10.28 / 0.264 / 38.27 | 13.85 / 0.690 / 21.65 |
| `comp_inverse_lut` | 10.26 / 0.263 / 38.28 | 13.82 / 0.689 / 21.21 |
| `comp_inverse_lut_gray` | 10.25 / 0.263 / 38.35 | 13.84 / 0.692 / 21.56 |

**08-04 の実行では補償が実質的に効いていない**（`no_comp` 10.173 → `comp_no_gamma` 10.173、
ΔE 38.76 → 38.05）。**08-05 では効いている**（ΔE 31.55 → 21.18、PSNR +1.6 dB）。

これは JSON から読める事実である。**古い P2C マップが原因だという読みは推論であって、
測って切り分けたわけではない**（08-04 と 08-05 の間には再校正そのものも挟まっている）。
ただし `no_comp` の SSIM が 0.253 → 0.694 と大きく動いていることは、補償の質ではなく
**目標画像と実写の位置合わせ**が変わったことを示しており、幾何を疑う根拠になる。

線形化4方式の差は、08-05 の実行でも PSNR で 0.1 dB 未満、ΔE で 0.5 未満しかない。

#### GrayCode 再校正の2回（`metadata.json`）

| | `..._window4`（11:54） | `..._window4_roi_redetermined`（13:32） |
| --- | ---: | ---: |
| 投影 | `active_region` でマスク | **全画面** |
| `active_region` | `(613, 201, 722, 722)`（手動） | **`(580, 217, 784, 784)`（自動導出）** |
| 対応点数 | 7,247,075 | **14,413,026** |
| `active_extrapolated_fraction` | 0.00673 | **0.0** |
| decode 時間 | 145.5 s | 277.0 s |
| 合計（撮影+現像+decode+補間） | 約 595 s | 約 751 s |

自動導出の内訳（`decoded_roi_stats`）: 連結成分 102個、最大成分の面積 1,007,427 px、
復号有効画素 1,134,358 px、内接円半径 400、中心 (972, 609)。
`inset = 8` なので半辺 392 → 一辺 784。

パターン数38、GrayCode ステップ (4,4)、黒しきい値20、白しきい値5、カメラ 4660×6984、
`camera_coordinate_scale = 1.0` は両実行で共通。

### 6.4 未確定・次の一手

**確定していること**: 新ROI `(580, 217, 784, 784)` と、それに対応する
`data/graycode_recalibration_20260805_window4_roi_redetermined/p2c_dense_active_1920x1080.npy`
が存在する。外挿率0%、対応点は約2倍。

**まだ誰もやっていないこと**:
- 新ROIを使う config が無い。`config.iso160_nonhdr_20260805_geom.toml` は旧ROI
  `(613,201,722,722)` と `..._window4`（`_roi_redetermined` ではない）の p2c を指している。
- 新ROIでの5条件評価が無い。
- `capture_chunked` の配線（6.2参照）。
- 上記6ファイル分の変更に対するテストがゼロ。`YOSHIDA_EDSDK_TRIGGER_SPOOL_DIR`、
  子プロセス現像、`_derive_square_roi` のいずれも `tests/` から到達しない。
- `data/README.md` の索引に 08-04/05 の5ディレクトリが載っていない。
- コミットされていない。

**自然な続きは、新ROIの config を作って評価を回し直すこと**だが、これは
metadata と config から読み取った推測であって、作業者の意図を確認したものではない。
**先に本人に確認すること。**

### 6.5 追記（2026-08-12）: 3ch混色応答の実測と HDR ブラケット修正

正本は [`docs/color_grid_response_20260812.md`](docs/color_grid_response_20260812.md)。要点:

- 321色（6³グリッド+ランプ+対角線）の実測で、**「1D線形化+4×3行列」モデル族は
  ΔE平均13〜19を構造的に残し、TPS 3D写像なら3.3**（10-fold CV）。線形化4方式の
  評価差が小さい理由はここにある。
- **`inverse_lut.capture_hdr_linear` の飽和すり抜けバグを発見・修正**（sample.py の
  08-07 修正と同クラス）。飽和した長露光の現像値が trust_high を下回って採用され、
  鮮色が最大4.7倍過小になっていた。基準露光アンカーのマージ+予測飽和ガードに変更、
  回帰テスト追加、462件緑。HDR有効の過去計測で「鮮色+暗チャンネル」を含むものは
  この影響を疑うこと。
- プロジェクタはコンテンツ切替後 0.6 秒で ±2〜4% ずれる（適応輝度の過渡、2秒で沈静）。
  撮影系の settle 0.3 秒は精度計測には不足。
- 08-07〜08-11 のラン（isoscan / リニア統一）はこの文書にまだ章がない。メモリと
  各 data ディレクトリの metadata が記録である。

同日の続報（同文書 §7）: ノード格子TPS逆写像の補償を実装し、新規216色校正から
fit した行列補償と実機比較した。**クランプ付きTPSは最良の行列条件と同等**
（PSNR 14.08 vs 14.17）。到達可能な色ではTPSが4〜5倍良いが、実ターゲットは
約半分が観測色域の外にあり、**全画像指標は色域の壁に律速される**。TPSの利得を
出すには目標の色域マッピングが前提（未着手、§13 #16）。
コア追加: `compensation/block_tps_map.py`。散布補間は観測域外で外挿が暴れるため
入力クランプが既定（素のTPSは補償画像に斑状アーティファクトが出る）。
追試: 「1D逆LUTの席にTPS焼き込み3D LUT」を置くハイブリッド
（`compensation/color_lut3d.py`、行列+33³三線形）も実装・実機評価。PSNR/SSIMは
2/5枚で最良だがΔEは全枚で劣り（+2.4）、順位は覆らず（同文書 §7.2b）。

（この追記の作業ではスクリプト6本・テスト6ファイル・コア修正+新モジュール2本が
働き木に増えている。テストは一括483件緑（2026-08-13実測）。コミットはユーザー判断。）

### 6.6 追記（2026-08-17）: プロジェクタを EPSON に交換、全再校正

正本は [`docs/epson_recalibration_20260817.md`](docs/epson_recalibration_20260817.md)。要点:

- EPSON は `\.\DISPLAY4` (3440,0) 1920×1080@60Hz。**Tv=1/60・av=14 固定**、
  非HDR、クリーンISO上限は250（320以上は白がROIの5割飽和）。スイープ 100/200/250、
  最良 ISO250（13.5dB/ΔE20.0）。**4線形化方式は全ISOで一致**
- **EPSONはほぼ加法的・γ2.2一致・ドリフト≤1%**。モデルCV: 行列系 ΔE4〜5.6、
  TPS 0.97。3D LUT実機 held-out ΔE1.8。到達殻はほぼ直方体
- 固定ROI **(340,64,890,890)**（ユーザー指示によりROIは初回決定を維持、
  GrayCodeはISO毎に再取得）
- 旧プロジェクタ時代のLUT・行列・p2c・configは EPSON に流用不可

### 7.1 投影ウィンドウ

- プロジェクタは Win32 座標 **`(3440, 0)`**、1920×1080、表示スケーリング 100%。
  Win32 のデバイス名では `\\.\DISPLAY4` にあたる。
- **会話に出てくる「ディスプレイ5」は `\\.\DISPLAY5` を意味しない。**
  6.3 の window probe が実測でこれを確認している。
- `projector.pos_x` / `pos_y` がウィンドウ位置。`refresh_rate = 60.0`。

### 7.2 露光

- 現行の実機設定（08-04/05 の config）: **av = 14 / tv = 1/15 / ISO 160（または100）/
  `image_quality = "LR"` / `half_size = false` / 非HDR**
- `tv = 1/15` は 60 Hz の4周期を積分するフリッカーフリー露光。
  シャッター位相のジッタは 1/60 で 3.3%、1/15 で 1.1%。
- `wait_key_ms = 300`、`trigger_wait_key_ms = 300`（表示更新の完了を安全側で保証）。
- 撮影RAWはフル解像度 **4660×6984**。現像後1枚あたり約390MB。

`config.toml`（追跡されている既定）は **ISO400 / `hdr_enabled = true` / 旧 p2c マップ**
のままで、08-04 以降の実機設定とは別物である。**実機を回すときは日付つきの config を
`--config` で明示すること。**

### 7.3 逆LUTの計測

- `inverse_lut.source` は `"channel_ramp"` と `"gray"` の**どちらか一方のみ**。混在不可。
- 現行 config の計測ROI: `roi_x=4520, roi_y=1600, roi_width=480, roi_height=400`
  （カメラ全解像度座標での右側の均一な白色面）。`num_levels = 17`、`lut_resolution = 256`。
- 信頼域: `trust_low = 10` / `trust_high = 245`（8bit画素値）。
- **プロジェクタは加法的ではない。** グレー駆動と単チャンネル駆動の応答が最大 1.7 倍違う。
  そのため `channel_ramp` の LUT でグレーを線形化しきることはできない。
  グレーを合わせたいなら `source = "gray"` を使う。
  2026-08-12 の面計測では破れは水準依存（暗グレーで R が原色和の 2.7 倍、中間で B が
  0.73 倍）で、G駆動→R発光 32% の実クロストークもある。
  → [`docs/color_grid_response_20260812.md`](docs/color_grid_response_20260812.md)
- 逆LUTの終端は 1.0 ではない（ISO400 の実測で約 0.876 / 0.876 / 0.688）。

原理は [`docs/inverse_lut_説明.md`](docs/inverse_lut_説明.md)。

### 7.4 校正が無効になる条件

`docs/current_color_compensation_pipeline.md` §17 に一覧がある。要点は、
**プロジェクタ/カメラの位置・向き・露光・表示スケーリング・投影ROIのどれかが変われば
幾何と測光の両方をやり直す**ということ。幾何だけが変わった場合は
`rebuild_compensation_with_map.py` で再撮影を避けられる。

---

## 8. テストの規律 — このリポジトリ固有の事情

**「通るが落ちることができないテスト」がこれまでに26件出ている。
緑であることは、テストが何かを守っている証拠にならない。**

### 8.1 繰り返し現れる3つの形

1. **フィクスチャが検証対象の次元について対称。**
   実データの行列は非対称（`max|offdiag| / max|diag| ≈ 0.25`）だが、テストが `2·I` を
   使っていたため、逆行列の転置が14テスト全部を通った。
2. **性質を散文で自信を持って述べたことで検証済みに見える。**
   「ディスク上でわざとアルファベット順を崩している」というコメントに反して NTFS は
   ソート済みで返し、製品側のソートを消しても11件全部通った。
3. **期待値が実環境の既定値とたまたま一致する。**
   実 `config.toml` のガンマが 2.2 なので、2.2 をハードコードしても19件通った。

### 8.2 特に高くついた2件

**シェーダの `categorical` 分岐（B4、Critical）。** `isNaNBits` を消しても、トーンチェーンを
足しても、`u_mode == 2` の分岐を削除しても 324 件が緑だった。実装者は実機GPUから画素を
読み戻して `isNaNBits` が必要であることを**証明した**が、その実行は手作業でCIには乗っていない。
**「証明されている」と「守られている」は別の状態で、その差は緑の結果に現れない。**
消すと実フレームの75%（未照明の周囲）が「範囲内」と報告される。

**`isnan()` と `v != v` は ANGLE/D3D11 で両方とも畳み込まれて消える。**
このマシン（RTX 3060 Ti）で3回再現している。**NaN判定は必ず IEEE ビットパターン
（`isNaNBits`、`highp uint` 必須）で書くこと。** `shaders.js` にある。

### 8.3 運用

- 各テストについて「どの誤実装なら通ってしまうか」を問う。
- 変異試験は形式ではない。実際に壊して、名前の付いたテストが落ちることを確認する。
  落ちなければテストの方を直す。
- `tests/test_web_frontend_js.py` の `_MUTATIONS` は、各変異が**ちょうど1つのチェックだけ**を
  落とすことを検査で強制する（双方向の固定集合）。**Python 側には同等の仕組みが無く、
  変異はすべて手動である（未対処）。**
- **変異試験の復元に `git checkout --` を使わないこと。** 未コミットの作業があると巻き添えで
  消える。B4で実際に起きた。ファイルバックアップを使う。**いまは第6章の未コミット作業が
  あるので特に危険。**
- レビューには「N+1件目があると仮定して探せ。生き残るのは主題が自明すぎて確認されないもの」
  と指示すると実際に見つかる。

---

## 9. 実データについて測って分かっていること

| 事実 | 値 |
| --- | --- |
| 実フレームの色域外率 | **46.5〜85.0%**（実行列4×実写3の12組） |
| `matrices_linear_gamma.npy` の非有限値 | **0個**（領域外74.86%も全て有限） |
| クリップ前にちょうど0.0/1.0になる値 | 3317万中 **0個と2個**（clip=Trueなら786万/845万） |
| numpy vs torch-CPU（uint8、622万画素） | **0個の差** |
| numpy vs torch-CUDA | 9個（CUDA自身がCPUと食い違う位置・大きさと完全一致） |
| 逆LUTの終端（ISO400） | 約 0.876 / 0.876 / 0.688（**1.0ではない**） |
| カメラ座標行列のサイズ | `(12, 4660, 6984)` = 約1.56 GB |
| プロジェクタ座標行列のサイズ | 約100 MB |

**マスク色（マゼンタ）は NaN を意味する。** `coef_tiles` / `surface` / `compensation` /
`gamut` すべてが有効投影領域の外を NaN にする規約で、読み取り欄の「有効投影領域の外」
マーカーもこれに依存している。`gamut` の画面で領域外がマゼンタなのはこの規約によるもので、
「クラス0＝範囲内」ではない。

線形化方式と fit の関係（`data/comp_test_20260724` の行列で確認）:
sRGB-fit は線形化なしを好み（ガンマがsRGBを打ち消す）、linear-fit は LUT/ガンマを好む。

---

## 10. してはいけないこと

- **素の `uv sync` を実行する**（EDSDK / rawpy と手置きDLLが消える）。必ず `--extra sample --extra web`。
- **`.venv` を消す。**
- **`git checkout --` / `git stash` / `git clean` を安易に使う。** 第6章の未コミット作業が消える。
- **`data/` に検証用の中間ファイルを書く。** スクラッチパッドを使う。
- **`data/` の実験記録ディレクトリを再実行・上書きする。** 日付つきディレクトリは凍結記録。
- **納品用スナップショットとの互換維持を試みる。** 互換は維持しない方針。タグ `v1.0-hdr-lut`
  を凍結してあり、diff の基準はそれで取る。
- **`data/` 配下の凍結スクリプト約30本を新パッケージへ追従させる。**
  `src.python.*` を import できないのは意図した状態。
- **GLSL で `isnan()` や `v != v` を書く。** ANGLE/D3D11 で消える。
- **緑のテストを根拠に「守られている」と述べる。** 第8章。
- **実機 config を `config.toml` の既定のまま回す。** 既定は ISO400 / HDR / 旧幾何で、
  現行の実機設定ではない。

---

## 11. コミット規約

`.github/instructions/commit-message.instructions.md` が正本。要点:

- 1行目: `[種別] 要約`（日本語、50〜70字目安、句点なし）
- 2行目: 空行
- 3行目以降: ファイル単位で「変更点 / 理由・意図 / 影響・注意」
- **ステージされた差分から確認できる事実のみを書く。** 存在しないファイル名・挙動・
  テスト結果を推測で書かない。テスト/動作確認は確証が無ければ「未実行」「不明」と書く。
- 複数テーマに跨る場合は最後に「補足: 分割検討」を1行入れる。

種別: `fix` / `hotfix` / `add` / `update` / `change` / `clean` / `disable` / `remove` /
`upgrade` / `revert`

---

## 12. ドキュメント地図

### 現況・仕様・計画

| 文書 | 内容 |
| --- | --- |
| [`docs/matrix_viewer_status_20260803.md`](docs/matrix_viewer_status_20260803.md) | **ビューアの正本。** 現況・アーキテクチャ・次の作業 |
| [`docs/current_color_compensation_pipeline.md`](docs/current_color_compensation_pipeline.md) | **補償パイプラインの正本。** 全挙動を17節で記述 |
| [`docs/color_grid_response_20260812.md`](docs/color_grid_response_20260812.md) | 3ch混色応答の実測・モデル比較・HDRブラケット修正（6.5） |
| [`docs/projector_io_visualization_survey_20260815.md`](docs/projector_io_visualization_survey_20260815.md) | 入出力特性の可視化手法の文献調査（図の体系・我々の図との対応・推奨） |
| [`docs/epson_recalibration_20260817.md`](docs/epson_recalibration_20260817.md) | **EPSON切替後の正本**（6.6）。全再校正・特性計測・新旧比較 |
| [`docs/inverse_lut_説明.md`](docs/inverse_lut_説明.md) | 逆LUTの原理 |
| [`docs/web_viewer_performance_20260731.md`](docs/web_viewer_performance_20260731.md) | ビューアの性能実測（B1/B2） |
| [`data/README.md`](data/README.md) | `data/` の索引（08-04/05 分は未反映） |
| `docs/superpowers/specs/2026-07-31-matrix-viewer-gui-design.md` | ビューアの仕様書 |
| `docs/superpowers/specs/2026-07-03-inverse-lut-design.md` | 逆LUTの設計 |
| `docs/superpowers/plans/*.md` | 計画書。**末尾の「完了記録と申し送り」を読むこと** |

### 実験報告（日付つき・凍結）

`docs/hardware_*`, `docs/full_recalibration_*`, `docs/hdr_*`, `docs/graycode_*`,
`docs/target_*`, `docs/iso200_iso400_*`, `docs/teacher_summary_*`。
`data/README.md` の第4章がディレクトリと報告書の対応表になっている。

`docs/graycode_geometry_and_compensation_20260730.md` は今回（08-05）の再校正の
前身にあたる。手順を再現するならまずこれを読む。

---

## 13. 未解決・未検証の一覧

| # | 項目 | 章 |
| ---: | --- | --- |
| 1 | ビューア計画B5（カメラ座標行列のP2Cワープ、`/api/matrix/{id}/warp` と `/api/job/{job_id}`） | 5.4 |
| 2 | `prepare_target_image_for_compensation` の初回7秒をどう扱うか（未決） | 5.4 |
| 3 | GUI↔CLI のワープ区間のバイト一致（未証明） | 5.4 |
| 4 | 導出した新ROI `(580,217,784,784)` を使う config と評価（未作成） | 6.4 |
| 5 | `inverse_lut.capture_chunked` が定義だけされていて未配線（デッドコード） | 6.2 |
| 6 | 08-04/05 の6ファイル分の変更にテストが無い | 6.4 |
| 7 | `data/README.md` に 08-04/05 の5ディレクトリが未記載 | 6.4 |
| 8 | ストリームBが未コミット | 6.1 |
| 9 | Python側の変異試験に自動化の仕組みが無い | 8.3 |
| 10 | 生成した投影画像の保存先の既定値（仕様 §13） | 5.4 |
| 11 | 複数行列を同時に開いて比較する機能（仕様 §13） | 5.4 |
| 12 | 3D写像（TPS/3D LUT）による補償の実装検討（行列モデルの構造限界 ΔE13〜19 の突破） | 6.5 |
| 13 | プロジェクタの動的処理（グレー207跳び・適応輝度）のメニュー無効化の可否確認 | 6.5 |
| 14 | HDR有効の過去計測（鮮色+暗ch）への飽和すり抜けバグの影響範囲の確認 | 6.5 |
| 15 | 撮影系スクリプトの settle 0.3 s 見直し（切替直後 ±2〜4% の過渡） | 6.5 |
| 16 | 目標の色域マッピング（TIP2015型の前処理）→ TPS補償の利得を実画像で引き出す | 6.5 |
| 17 | 08-07 の color_matrix 撮影だけが持つ約1.6倍スケールの原因（今日の両経路は一致） | 6.5 |

---

## 14. この文書の更新について

- ストリームAの詳細を更新するときは `docs/matrix_viewer_status_20260803.md` を直し、
  ここは要約とリンクだけ直す。
- **ストリームBはここが唯一の記録である。** 実機作業を進めたら第6章を必ず更新するか、
  `docs/` に報告書を作ってここからリンクする。
- 第13章の表は、項目が片付いたら消すのではなく「解決」と結果を書いてから消す。
