# 色補償 4条件比較テスト (2026-07-24)

プロジェクタ線形化 4 条件（`none` / `gamma`=逆ガンマ2.2 / `lut_ch`=channel_ramp 測定LUT / `lut_gray`=gray 測定LUT）で
色補償パイプラインの精度を比較したときの成果物一式。

## セットアップ（当日実測）

- プロジェクタ: (3440, 0) 1280x720（ユーザー呼称「ディスプレイ5」）
- カメラ: Canon EDSDK, av=13, iso=100, tv=1/15（フリッカーフリー・白非飽和）
- 幾何: GrayCode を当日再実行（`result_*_delaunay.npy`、リポジトリルート）
- 評価: 撮影→p2c ワープ→プロジェクタ空間で target と比較（グレイコードが直接デコードできた
  画素のみ = `valid_mask.npy`、全体の34%）。指標は masked PSNR / SSIM(輝度) / ΔE76。

## 結果サマリ（3枚平均 PSNR [dB]）

| フィット空間 | none | gamma | lut_ch | lut_gray |
|---|---|---|---|---|
| sRGB空間（sample.py 現行方式） | **17.78** | 17.41 | 17.42 | 17.37 |
| リニア空間 | 17.13 | 17.86 | **17.86** | 17.84 |

- **sRGB 空間フィット（現行 sample.py 方式）では「補正なし」が最良**。プロジェクタの
  ネイティブガンマ (≈2.2) と撮影の sRGB エンコード (≈1/2.4) が打ち消し合い、フィット空間で
  チェーンがほぼアフィンになるため。線形化を入れると sRGB エンコードの非線形が残って
  アフィン行列モデルの適合が悪化する。
- **リニア空間フィットでは線形化条件が勝つ**（同メカニズムの裏返し）。
- 総合ベストは「リニアフィット + lut_ch/gamma」(17.86dB) ≒「sRGBフィット + none」(17.78dB)。
  上限は線形化の選択ではなく、色域クリップ（空のマゼンタ化）・プロジェクタの輝度揺らぎ
  (±1%)・ワープぼけ等で決まっている。
- lut_ch と gamma は実質同等（このプロジェクタのグレー応答がほぼ純ガンマ2.2 のため）。

## ファイル

- `matrices_<cond>.npy` — sRGB 空間フィットの 4x3 行列 (720x1280x4x3)。ターゲットは sRGB uint8 のまま入力
- `matrices_lin_<cond>.npy` — リニア空間フィット。ターゲットは sRGB→リニア変換して入力
- `valid_mask.npy` — 評価マスク（グレイコード直接デコード領域、720x1280 bool）
- `fit_captures_<cond>/warped_XX.png` — フィット用 27 パターンのワープ済み撮影（sRGB、再フィット用）
- `results/<target>/`, `results_linear/<target>/` — 補償入力・撮影結果・モンタージュ
- `metrics.json`, `metrics_linear.json` — 全指標
- `apply_compensation.py` — **保存済み行列を新しいターゲット画像に適用するスクリプト**
  （`uv run python data/comp_test_20260724/apply_compensation.py <img> [cond] [srgb|linear]`）
- `comp_test.py` / `comp_test_linear.py` / `graycode_run.py` / `warp_sanity.py` — 実行スクリプト一式

## 正方形ターゲットワークフロー（2026-07-24 追加）

512x512 の正方形ターゲットを、指定した投影領域（プロジェクタ座標の正方形、既定
**(x=416, y=142, 一辺=468)** = デコード済みキャンバスの最大内接正方形 − 8pxマージン）へ
補償投影し、撮影→p2cワープ→領域crop→512x512 リサイズしてから画像処理・評価する方式。

- `square_comp_test.py [x y side]` — 4条件×2フィット空間の評価（行列は全画面版から領域切り出し、再フィット不要）
- `apply_compensation_square.py <img> [cond] [fit_space] [x y side]` — 新しい正方形ターゲット→投影用フレーム生成
- `square_region_preview.png` — 採用領域の可視化
- `square_results/` — 512x512 結果画像・モンタージュ・`metrics_square.json`

結果（3枚平均、512x512 全面評価）: linear-fit + lut_ch が最良
（PSNR 18.13dB / ΔE76 16.24）、次点 linear/gamma (18.13dB / 16.91)、
srgb/none (17.87dB / 17.14)。順位構造は全画面評価と同じで、リニアフィット時は
LUT(channel) がガンマより ΔE でわずかに優位。

## 注意

- 行列・マップは 2026-07-24 の物理配置に紐づく。カメラ/プロジェクタ/キャンバスを動かしたら
  `graycode_run.py` → `comp_test.py`（フィット段のみ）を再実行すること。
- inverse LUT の単体検証・校正の成果物は `data/inverse_lut_test_20260724/` を参照。
