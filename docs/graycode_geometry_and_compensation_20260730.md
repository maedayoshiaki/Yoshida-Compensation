# GrayCode幾何再校正・色補償再評価（2026-07-30）

## 結論

色補償が全体的に右上へずれて見えた主因は、旧P2Cが4頂点から作った単一ホモグラフィであり、投影面内部の非線形な対応を表現できていなかったことだった。

最新GrayCodeのdense P2Cへ置き換え、実写チェッカーで検証した結果、残差はP2Cで`(-0.24, +0.08) px`、C2Pで`(-0.29, +0.02) px`となった。ISO200の4条件比較も新マップで再生成・再撮影し、解析ガンマ補正が平均で最良だった。

以後、半解像度RAWは使用しない。設定値を`false`にしただけでなく、Canonバックエンドは`half_size=true`を実行前に例外として拒否する。

## 使用した版と設定

- 作業ブランチ: `feature/triggered-batch-capture`
- GrayCode最新develop基点: `a687b8f9c7a29c0baf3f4961760873cc0d1f30e2`
- C2P inpaint修正: `340b7fe`（`fix/c2p-inpaint-propagation`）
- edsdk-python: v0.2.0
- 投影出力: Win32 `DISPLAY4`、1920 x 1080、デスクトップ位置`(3440, 0)`
- 有効投影領域: `(x=613, y=201, width=722, height=722)`
- カメラ: ISO200、Av14、1/15秒、RAW、`half_size=false`
- RAW現像サイズ: 4660 x 6984
- GrayCode step: 4 x 4

## GrayCode再校正

38パターンを先にすべてトリガ撮影し、その後でフル解像度RAWをまとめて現像した。

| 処理 | 時間 |
| --- | ---: |
| 投影・トリガ撮影 | 90.99秒 |
| フル解像度RAW現像 | 164.98秒 |
| GrayCode decode | 185.73秒 |
| dense P2C補間 | 15.46秒 |

- 対応点数: 8,448,330
- active領域内の外挿率: 0.951%
- production用マップ: `data/graycode_recalibration_20260730/p2c_dense_active_1920x1080.npy`
- C2P検証用マップ: `data/graycode_recalibration_20260730/c2p_sparse_full_camera.npy`
- 全メタデータ: `data/graycode_recalibration_20260730/metadata.json`

旧マップは実写チェッカーのinner相関が約0.797だった。新P2Cで同じ保存撮影を再ワープすると約0.951、現在のセットアップを新規撮影すると0.9533まで改善した。

## P2C/C2Pの実写検証

同じチェッカー投影を1枚のフル解像度RAWで撮影し、P2CとC2Pの両経路を評価した。

| 経路 | 位相差 x/y [px] | inner相関 | RMSE | ワープ時間 |
| --- | ---: | ---: | ---: | ---: |
| dense P2C | -0.241 / +0.084 | 0.9533 | 0.2388 | 0.129秒 |
| sparse C2P | -0.294 / +0.019 | 0.9327 | 0.2470 | 31.71秒 |

C2Pは4 x 4 stepの疎な出力に格子穴が残っていた。GrayCodeのconvolution inpaintで、近傍和の誤った二重正規化と、近傍を持たない穴まで有効化する判定を修正した。修正後は格子穴が消え、位置差も1 px未満になった。

productionでは精度と速度の両方が良いdense P2Cを使用する。

検証画像と数値:

- `data/graycode_recalibration_20260730/geometry_verify_current_fixed/checker_warped_p2c.png`
- `data/graycode_recalibration_20260730/geometry_verify_current_fixed/checker_warped_c2p.png`
- `data/graycode_recalibration_20260730/geometry_verify_current_fixed/geometry_metrics.json`

## ISO200色補償の再構築

4段階 x RGB = 64色で求めた既存のカメラ座標色行列は再利用し、プロジェクタ座標へのワープと補償画像だけを新dense P2Cで再計算した。1.56 GBの行列3個と9枚の補償画像は合計約9.47秒で再構築できた。

再構築先:

- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/raw_4level/`
- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/gamma_4level/`
- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/lut_4level/`

## 4条件のフル実写比較

次の12枚（3画像 x 4条件）を1回のトリガバッチで撮影した。

1. 色補償なし
2. 色補償あり・ガンマ補正なし
3. 色補償あり・解析逆ガンマ2.2
4. 色補償あり・実測逆LUT

撮影はすべてフル解像度RAWで、投影・トリガ29.22秒、後段RAW現像20.96秒、API全体51.15秒だった。現像を投影ループ外へ分離したため、投影を占有する時間は撮影と現像の合計50.18秒に対して約41.8%短い。

| 条件 | PSNR [dB] | SSIM | Delta E76 |
| --- | ---: | ---: | ---: |
| 色補償なし | 15.56 | 0.7494 | 26.21 |
| 色補償あり・ガンマなし | 18.40 | 0.7189 | 16.37 |
| **色補償あり・解析逆ガンマ2.2** | **19.75** | **0.7584** | **13.59** |
| 色補償あり・実測逆LUT | 19.32 | 0.7485 | 15.75 |

旧ホモグラフィ時の解析ガンマ平均（PSNR 17.48、SSIM 0.4057、Delta E76 16.86）に対し、新dense P2CではPSNRが約2.27 dB向上、SSIMが約0.353向上、Delta E76が約3.27低下した。

比較画像と全数値:

- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/evaluation_requested4/comparison_nature_1_sq_requested4.png`
- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/evaluation_requested4/comparison_nature_2_sq_requested4.png`
- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/evaluation_requested4/comparison_nature_3_sq_requested4.png`
- `data/hardware_comparison_iso200_20260730/geometry_corrected_20260730/evaluation_requested4/metrics_requested4.json`

## テスト

- Yoshida-Compensation: 64 passed
- GrayCode: 67 passed
- GrayCode C2P inpaint回帰テストを追加
- Canonバックエンドに半解像度RAW拒否テストを追加

`pytest`の対象は親リポジトリの`tests/`に限定し、`data/`内の保存済み実験スクリプトは収集しない。GrayCode自身のテストは`external/GrayCode/tests`を別に実行する。
