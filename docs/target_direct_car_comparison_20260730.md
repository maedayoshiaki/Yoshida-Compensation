# car / car-new直接入力色補償

実験日: 2026-07-30  
入力条件: `target_decode=none`

## 実施内容

`car` と `car-new` は、投影領域に合わせるための中央正方形切り出しと512 × 512化だけを行った。補償計算へ渡す値は

```text
target = image_uint8 / 255
```

であり、標準sRGB EOTF、単純な `^2.2`、その他のガンマ変換は一切適用していない。

ISO 200/400で次の5条件を、初回・切替とも300 ms待機して実撮影した。

1. 色補償なし
2. 色補償あり・投影ガンマ補正なし
3. 色補償あり・解析逆ガンマ
4. 色補償あり・チャンネル逆LUT
5. 色補償あり・グレー逆LUT

Av14、Tv 1/15秒を維持し、RAWはすべてフル解像度で現像した。色変換行列は各ISO・各方式のRGB各4段階、合計64色の既存行列を使用した。

## 結果

PSNRとSSIMは大きいほど良く、ΔE76は小さいほど良い。

### ISO 200

| 画像 | 条件 | PSNR (dB) | SSIM | ΔE76 |
| --- | --- | ---: | ---: | ---: |
| car | 色補償なし | 14.553 | **0.7679** | 33.283 |
| car | ガンマ補正なし | **17.086** | 0.6602 | **22.048** |
| car | 解析逆ガンマ | 16.263 | 0.6354 | 22.260 |
| car | チャンネル逆LUT | 15.624 | 0.6064 | 22.954 |
| car | グレー逆LUT | 15.736 | 0.6068 | 24.482 |
| car-new | 色補償なし | **17.988** | **0.7798** | **19.871** |
| car-new | ガンマ補正なし | 15.520 | 0.6326 | 26.499 |
| car-new | 解析逆ガンマ | 13.813 | 0.5950 | 31.012 |
| car-new | チャンネル逆LUT | 13.139 | 0.5339 | 32.693 |
| car-new | グレー逆LUT | 13.611 | 0.5459 | 31.729 |

撮影＋現像は44.14秒（撮影ループ27.39秒、後段現像15.76秒）だった。

### ISO 400

| 画像 | 条件 | PSNR (dB) | SSIM | ΔE76 |
| --- | --- | ---: | ---: | ---: |
| car | 色補償なし | **18.640** | **0.8007** | 28.455 |
| car | ガンマ補正なし | 16.416 | 0.7486 | **19.650** |
| car | 解析逆ガンマ | 13.091 | 0.6922 | 25.727 |
| car | チャンネル逆LUT | 14.774 | 0.6705 | 21.073 |
| car | グレー逆LUT | 13.995 | 0.6495 | 22.835 |
| car-new | 色補償なし | **20.988** | **0.7924** | **19.107** |
| car-new | ガンマ補正なし | 15.816 | 0.6625 | 21.669 |
| car-new | 解析逆ガンマ | 12.484 | 0.6008 | 32.929 |
| car-new | チャンネル逆LUT | 13.025 | 0.5532 | 28.410 |
| car-new | グレー逆LUT | 11.895 | 0.5262 | 33.108 |

撮影＋現像は46.19秒（撮影ループ27.47秒、後段現像17.72秒）だった。

## 解釈

`car-new` は画素値としてほぼ `car^2.2` であり、すでに暗い線形値に近い。これを変換せず線形目標として補償すると、カメラの線形応答はその値を目標にする。保存時には確認用PNGへsRGB OETFを掛けるため、表示画像は明るい `car` に近づく。

したがって、元の暗い `car-new` JPEGをそのまま評価基準にした指標では、無補償が最良になる。これは入力を勝手に再変換した結果ではなく、直接入力と表示用sRGB保存の定義から生じる。

`car` では両ISOとも、ΔE76は「色補償あり・投影ガンマ補正なし」が最小だった。一方で構造指標SSIMは無補償が高く、直接入力では明るい領域の飽和と固定投影面模様が目立つ。

## 成果物

### ISO 200

- [car比較画像](../data/full_recalibration_20260730_100pct/evaluation_target_direct_car_wait300/comparison_car_sq_requested5.png)
- [car-new比較画像](../data/full_recalibration_20260730_100pct/evaluation_target_direct_car_wait300/comparison_car-new_sq_requested5.png)
- [全指標JSON](../data/full_recalibration_20260730_100pct/evaluation_target_direct_car_wait300/metrics_requested5.json)

### ISO 400

- [car比較画像](../data/full_recalibration_20260730_iso400/evaluation_target_direct_car_wait300/comparison_car_sq_requested5.png)
- [car-new比較画像](../data/full_recalibration_20260730_iso400/evaluation_target_direct_car_wait300/comparison_car-new_sq_requested5.png)
- [全指標JSON](../data/full_recalibration_20260730_iso400/evaluation_target_direct_car_wait300/metrics_requested5.json)

