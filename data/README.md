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

`src/python/` を削除しパッケージを `packages/core/yoshidacomp/` へ移行したため（package-split 計画）、
この節にある `data/` 配下の実験スクリプト約30本は `src.python.*` を import できなくなっている。
これは意図した状態である。各スクリプトはその実験のデータ・configと対になった凍結記録であり、
新パッケージへの追従（移行）は行わない。動作していた組み合わせはタグ `v1.0-hdr-lut` が保存している。

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

### 2026-08-06

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `graycode_recalibration_20260806_iso100_fullframe/` | ISO100・全画面GrayCodeからROI `(578,214,784,784)` とP2C/C2Pを新規導出 | `docs/full_recalibration_iso100_iso160_20260806.md` |
| `iso100_nonhdr_full_20260806/` | ISO100・非HDR。露出/幾何検証、2種LUT、方式別4行列、5条件評価、画素別マップ | 同上 |
| `graycode_recalibration_20260806_iso160_fullframe/` | ISO160・全画面GrayCodeからROI `(583,214,786,786)` とP2C/C2Pを新規導出 | 同上 |
| `iso160_nonhdr_full_20260806/` | ISO160・非HDR。露出/幾何検証、2種LUT、方式別4行列、5条件評価、画素別マップ | 同上 |
| `graycode_recalibration_20260806_iso100_fullframe_step1_rerun/` | ISO100・フル解像度Step 1 GrayCode 46枚からROI `(566,217,784,784)` とP2C/C2Pを再導出 | `docs/full_recalibration_iso100_iso160_step1_rerun_20260806.md` |
| `iso100_nonhdr_full_20260806_step1_rerun/` | ISO100・保護高速撮影。露出/幾何検証、2種LUT、方式別4行列、5条件評価、画素別マップ | 同上 |
| `graycode_recalibration_20260806_iso160_fullframe_step1_rerun/` | ISO160・フル解像度Step 1 GrayCode 46枚からROI `(591,213,788,788)` とP2C/C2Pを再導出 | 同上 |
| `iso160_nonhdr_full_20260806_step1_rerun/` | ISO160・保護高速撮影。露出/幾何検証、2種LUT、方式別4行列、5条件評価、画素別マップ | 同上 |

（2026-08-07 の isoscan 8ラン、2026-08-09〜11 のリニア統一ランのディレクトリは未記載。
一覧は `AGENTS.md` 第6章の追記とメモリを参照。）

### 2026-08-12

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `color_grid_response_20260812/` | 3ch混色応答グリッド計測 r1。**HDRブラケット飽和すり抜けバグの凍結証拠**（鮮色・混合色は汚染、グレー・暗部は有効）。逐次ランププローブ（`sequential_ramp_probe.py` / `.json`）を含む | `docs/color_grid_response_20260812.md` |
| `color_grid_response_20260812_r2/` | 同計測のバグ修正後再実行（**有効データ**）。321色生駆動 + LUT通し35色 + 番兵13点、`analysis/` にプロットと summary.json | 同上 |
| `tps_calibration_20260812/` | TPS補償用 6³=216色校正。プロジェクタ座標1/4縮小リニアフレーム225枚 + manifest | 同上 §7 |
| `tps_compensation_20260812/` | 同一校正から fit した行列/ガンマ+行列/TPS/行列+3D LUT の補償画像・tps_map.npz・inverse_lut3d.npz・実機評価3ラン（`evaluation`/`evaluation_r2`/`evaluation_hybrid`） | 同上 §7 |

### 2026-08-13

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `lut3d_validation_20260813/` | 3D LUT（TPS焼き）単体の実機精度検証（要求色→LUT→投影→実測、73計測、`analysis/` に summary・誤差可視化・立体プロット群） | `docs/color_grid_response_20260812.md` §7.2c |
| `lut3d_validation_20260813_linear/` | 区分線形補間で焼いた3D LUTの同一プロトコル検証（同一要求色68点、TPS版とのペア比較つき） | 同上 §7.2c |


### 2026-08-17（EPSON プロジェクタ切替）

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `graycode_epson_20260817_iso{100,200,250,320,400}/` | EPSON切替後のGrayCode幾何（ISO毎、固定ROI (340,64,890,890)） | `docs/epson_recalibration_20260817.md` |
| `epson_iso{100,200,250,320,400}_nonhdr_20260817/` | Tv1/60・f14・非HDRのフル校正と5条件評価。総合最良は**ISO400 comp_gamma 17.5dB/ΔE11.8**（320/400は白ROI約5割の飽和許容ラン） | 同上 |
| `epson_tps_calibration_20260817/` | 216色校正フレーム（特性計測・TPS/LUT fit 用） | 同上 |
| `epson_color_grid_response_20260817/` | 321色応答調査（加法比0.91-1.07、モデルCV: 行列4.2 / TPS 0.97） | 同上 |
| `epson_lut3d_validation_20260817/` | 3D LUT実機検証（gray/held-out ΔE1.8）と特性図一式 | 同上 |
| `epson_tps_compensation_20260817/` | inverse_lut3d.npz と行列/ハイブリッド補償画像 | 同上 |

### 2026-08-18（Tv=1/30 スイープと最良条件の特性計測）

| ディレクトリ | 内容 | 対応文書 |
| --- | --- | --- |
| `graycode_epson_20260818_tv30_iso{250,320,400,500}/` | Tv1/30版GrayCode幾何（固定ROI継続） | `docs/epson_recalibration_20260817.md` §2b |
| `epson_iso{250,320,400,500}_tv30_20260818/` | Tv1/30フル校正と5条件評価。**全実験最良 = ISO250 comp_gamma 18.6dB/ΔE9.86**。ISO500で補償利得消失 | 同上 §2b |
| `epson_tps_calibration_20260818_tv30/` ほか `*_20260818_tv30/` 4式 | 最良条件（Tv1/30 ISO250）での特性計測一式（216色/321色/3D LUT検証、白54%飽和条件の実効特性） | 同上 §3b |

## 5. 追跡していないもの

`.gitignore` により以下は git 管理下にない。実験を再現するには再計算が必要。

- 色変換行列 `*.npy`（カメラ座標は1個あたり約1.56 GB、プロジェクタ座標は約100 MB）
- 対応マップ `p2c_*.npy` / `c2p_*.npy`
- 逆LUT `*.npz`
- 撮影PNG・パターンPNG・評価画像

パイプライン全体の挙動は `docs/current_color_compensation_pipeline.md` に記述している。
