# セットアップ変更後の色補償再校正（2026-07-29）

## 結論

変更後の物理セットアップに対して、投影位置の検出、幾何対応の再作成、
27色のフル解像度撮影、色混合行列計算、3枚の補償画像生成、実投影・再撮影評価まで
最初から完走した。

直接投影と比べ、3枚すべてでPSNR、SSIM、Delta E76が改善した。

| 指標（3枚平均） | 直接投影 | 補償投影 | 改善 |
| --- | ---: | ---: | ---: |
| PSNR | 12.17 dB | 15.45 dB | +3.28 dB |
| SSIM | 0.366 | 0.403 | +0.037 |
| Delta E76 | 31.01 | 20.25 | -10.76 |

## 新しい幾何対応

旧セットアップのROIと2026-07-24作成のP2Cマップは使用せず、黒投影と白投影の
実写差分から現在の投影四隅を検出した。

- 投影先: Win32 `\\.\DISPLAY4`
- プロジェクタ: 1920x1080
- active region: `(613, 201, 722, 722)`
- カメラ画像: 6984x4660
- カメラ上の投影四隅（TL, TR, BR, BL）:
  `(2354,602)`, `(5074,278)`, `(5048,3634)`, `(2182,3288)`

投影面は平面なので、四隅からプロジェクタ座標→カメラ座標のホモグラフィを計算し、
1920x1080全画素のP2Cマップとして保存した。チェッカー再撮影では、中央行の格子境界が
期待位置に約0～7 pxで一致した。

## フル色校正

撮影条件:

- Canon EDSDK、RAW
- Av=13、Tv=1/15、ISO=100
- 27色（`num_divisions=3`）
- 初回表示待機300 ms、2枚目以降1 ms
- フル解像度RAW現像
- CUDAによる全32,545,440画素の色混合行列計算

所要時間:

- 27色の投影・撮影: 64.30秒
- 後段RAW現像: 47.69秒
- 補償画像3枚の生成まで含む全体: 239.6秒

## 実投影評価

各目標画像について、無補償の直接投影と補償画像を同一トリガバッチで撮影し、
新P2Cマップでプロジェクタ座標へ戻して評価した。

| 画像 | 条件 | PSNR [dB] | SSIM | Delta E76 |
| --- | --- | ---: | ---: | ---: |
| nature_1 | 直接 | 11.63 | 0.459 | 28.16 |
| nature_1 | 補償 | 13.99 | 0.502 | 21.46 |
| nature_2 | 直接 | 12.56 | 0.240 | 30.66 |
| nature_2 | 補償 | 16.45 | 0.272 | 17.44 |
| nature_3 | 直接 | 12.31 | 0.399 | 34.22 |
| nature_3 | 補償 | 15.89 | 0.436 | 21.85 |

投影面上の黒い物体など完全に遮光される箇所や、プロジェクタの色域を超える補償は
物理的に復元できない。それらを含めた全面評価でも全画像・全指標が改善した。

## 保存先

- 実行設定: `data/hardware_recalibration_20260729/config_20260729.toml`
- セットアップ確認: `data/hardware_recalibration_20260729/setup_check/`
- P2Cマップ: `data/hardware_recalibration_20260729/maps/`
- チェッカー検証: `data/hardware_recalibration_20260729/geometry_verify/`
- 27色パターンと実写: `data/hardware_recalibration_20260729/color_calibration/`
- 色混合行列: `data/hardware_recalibration_20260729/results/color_mixing_matrices.npy`
- 補償画像: `data/hardware_recalibration_20260729/results/compensation_projector/`
- ワープ中間確認: `data/hardware_recalibration_20260729/warp_debug/`
- 実投影評価: `data/hardware_recalibration_20260729/evaluation/`
- 評価値: `data/hardware_recalibration_20260729/evaluation/metrics.json`
- 実行ログ: `data/hardware_recalibration_20260729/full_pipeline.log`

旧色混合行列は
`data/hardware_recalibration_20260729/backup/color_mixing_matrices_before_20260729.npy`
へ退避した。
