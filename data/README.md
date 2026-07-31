# data/ ディレクトリ索引

このディレクトリは入力画像・パイプラインの生成物・実験記録が混在している。
`.gitignore` により `*.npy` / `*.png` などの大容量バイナリは追跡していない
（実測で約 99 GB がディスク上にあり、git 管理下は約 9 MB）。
追跡しているのはスクリプト・設定・計測結果 JSON・README である。

## 1. パイプライン作業ディレクトリ（実行のたびに上書きされる）

`examples/python/sample.py` などが書き出す先。中身は成果物ではなく作業用。

| ディレクトリ | 内容 |
| --- | --- |
| `linear_proj_patterns/` | 64色の線形パターン |
| `inv_gamma_proj_patterns/` | 逆ガンマを適用した投影パターン |
| `inv_lut_proj_patterns/` | 実測逆LUTを適用した投影パターン |
| `captured_images/` | 撮影結果の保存PNG（目視用。行列fitには再読込しない） |
| `compensation_images/` | 線形補償画像 |
| `inv_gamma_comp_images/` | 逆ガンマ適用後の投影用PNG |
| `inv_lut_comp_images/` | 逆LUT適用後の投影用PNG |
| `graycode_pattern/` | GrayCodeパターン |
| `captured/`, `debug/`, `manual_capture/` | 撮影・デバッグ用の一時出力 |
| `inv_gamma_proj_pattern/` | 旧名の空ディレクトリ（`inv_gamma_proj_patterns/` が現行） |

`sample_warp_debug_DEBUG_ONLY/` も同種の生成物だが、`sample.py` が実行のたびに
ディレクトリごとリセットして再生成するため、追跡すると常時 dirty になる。
`.gitignore` で除外している。

## 2. 入力（ターゲット画像）

| ディレクトリ | 内容 |
| --- | --- |
| `target_images/` | 既定のターゲット入力 |
| `comp_test_targets/` | 比較実験用ターゲット（JPEG） |
| `comp_test_targets_square/` | 正方形512x512のターゲット5枚（car, car-new, nature_1..3） |
| `comp_test_targets_nature_srgb/` | 通常sRGB画像のターゲット |
| `comp_test_targets_car_direct/` | 直接入力（sRGBデコードなし）比較用ターゲット |

## 3. カメラ校正

| ディレクトリ | 内容 |
| --- | --- |
| `chess_captured/` | チェッカーボード撮影と `camera_calib.npz` |
| `pose_output/` | PnPで求めた姿勢（`pose_unity.json`） |

## 4. 実験記録（日付つき・凍結）

過去の実験結果。原則として再実行せず、記録として残す。
対応する報告書が `docs/` にある場合は併記した。

### 2026-07-24

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `comp_test_20260724/` | 線形化4方式の比較。行列は 720x1280 座標で8種 | ディレクトリ内 `README.md` |
| `inverse_lut_test_20260724/` | 逆LUTの測定実験。解析スクリプト20本と5パス分の測定JSON | `docs/inverse_lut_説明.md` |
| `linear_pipeline_1080_prev_target/` | 全工程リニア化（1920x1080）。ターゲットは白平面キャンバス | ディレクトリ内 `README.md` |
| `linear_pipeline_1080/` | 同上をグラデーションシートへ変更して再校正。プロジェクタ座標行列4種と `valid_mask.npy` | ディレクトリ内 `README.md` |

### 2026-07-25

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `hardware_trigger_smoke_20260725_121719/` | EDSDKトリガ撮影のスモークテスト | `docs/hardware_trigger_smoke_20260725.md` |
| `hardware_color_smoke_20260725_121856/` | 色パターン撮影のスモークテスト | 同上 |
| `hardware_sample8_smoke_20260725_122114/` | 8枚撮影のスモークテスト | 同上 |
| `hardware_switch_wait_20260725/` | 投影切替の待機時間ベンチマーク | `docs/hardware_switch_wait_20260725.md` |
| `hardware_frame_persistence_20260725/` | 前フレーム残留の検証（赤/シアン交互投影など） | 同上 |
| `hardware_full_comp_20260725/` | 全工程の実機通し | `docs/hardware_full_compensation_20260725.md` |

### 2026-07-29

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `hardware_recalibration_20260729/` | 実機再校正。ホモグラフィP2Cマップを含む | `docs/hardware_recalibration_20260729.md` |
| `hardware_comparison_20260729/` | ガンマ/LUT方式の比較 | `docs/hardware_compensation_gamma_lut_comparison_20260729.md` |
| `hardware_comparison_iso160_20260729/` | 同比較のISO160版と露出確認 | `docs/hardware_compensation_gamma_lut_comparison_iso160_20260729.md` |

### 2026-07-30

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `hardware_comparison_iso200_20260730/` | ISO200での方式比較と幾何補正版の再構築 | `docs/hardware_compensation_gamma_lut_comparison_iso200_20260730.md` |
| `graycode_recalibration_20260730/` | GrayCode再校正。**現行の `p2c_dense_active_1920x1080.npy` の出所** | `docs/graycode_geometry_and_compensation_20260730.md` |
| `full_recalibration_20260730_100pct/` | 表示スケーリング100%変更後の全再校正（方式別4種） | `docs/full_recalibration_20260730_100pct.md` |
| `full_recalibration_20260730_iso400/` | 同上のISO400版 | `docs/iso200_iso400_five_condition_comparison_20260730.md` |
| `hdr_recalibration_20260730_iso400/` | **最新**。HDR撮影による方式別再校正と5条件評価 | `docs/hdr_full_recalibration_iso400_20260730.md` |

## 5. 追跡していないもの

`.gitignore` により以下は git 管理下にない。実験を再現するには再計算が必要。

- 色変換行列 `*.npy`（カメラ座標は1個あたり約1.56 GB、プロジェクタ座標は約100 MB）
- 対応マップ `p2c_*.npy` / `c2p_*.npy`
- 逆LUT `*.npz`
- 撮影PNG・パターンPNG・評価画像

パイプライン全体の挙動は `docs/current_color_compensation_pipeline.md` に記述している。
