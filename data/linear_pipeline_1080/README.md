# 完全リニア色補償パイプライン — 新ターゲット（グラデーションシート）(2026-07-24 午後)

投影対象を**湾曲したグラデーションカラーのシート**（白投影で上=オレンジ/中央=マゼンタ/下=シアン、
G の反射が弱い）に変更してのフル再校正 + 4条件比較。前ターゲット（白平面キャンバス）の成果物は
`data/linear_pipeline_1080_prev_target/` に退避済み。

## 構成（完全リニア）

- カメラ: リニアRAWのまま処理（sRGBは保存PNGのみ）。av=13, iso=100, tv=1/15
- 逆LUT: channel_ramp / gray とも 33階調×3パス中央値、trust_low=5、
  LUT計測ROI (3147, 818, 1770, 2475)
- 幾何: GrayCode @1920x1080（c2p補間はスキップ、p2cのみ）
- 行列: リニア空間フィット（リッジ正則化付き正規方程式）
- 投影領域: **(x=667, y=207, 一辺=692)**、512x512ターゲットを拡大投影→撮影→ワープ→512に戻して評価

## LUT検証（グレー駆動17階調, RMSE）

raw 0.203/0.201/0.203 → 逆ガンマ2.2 0.0051/0.0067/0.0072 → gray LUT **0.0031/0.0090/0.0055**
（R/Bはガンマ超え、GはSNR低下でわずかに劣後）

## 最終結果（3枚平均, 512x512全面）— tv=1/15, ISO 125（最終設定）

| 条件 | PSNR(sRGB) [dB] | SSIM | ΔE76 |
|---|---|---|---|
| 補償なし（ターゲット直投影） | 13.92 | 0.597 | 32.67 |
| 補償 + 線形化なし (none) | 17.10 | 0.667 | 17.04 |
| **補償 + gamma 2.2** | **18.14** | **0.694** | **15.69** |
| 補償 + lut_ch | 18.02 | 0.686 | 16.09 |
| **補償 + lut_gray** | **18.18** | 0.685 | 15.78 |

補償の総合効果: +4.2dB / ΔE −17.0（無補償比）。うち行列補償が +3.2dB、線形化が +1.0dB。

ISO 100→125（1.25倍）で全条件 +0.8〜1.4dB 改善。ISO 160/200 は本機の RAW では実質2倍ゲインで
白 R/B が3〜4割飽和するため不採用（ISO 125 が tv=1/15 での無飽和上限）。ISO 100 時の結果と
行列は `tv1-15_backup/`（旧 ISO100 行列は matrices を参照）に退避。

### 参考: ISO 100 時（旧）

| 条件 | PSNR(sRGB) | ΔE76 |
|---|---|---|
| none | 16.30 | 17.69 |
| gamma | 16.81 | 16.89 |
| lut_ch | 16.67 | 17.08 |
| lut_gray | 16.79 | 16.90 |

- 線形化条件（gamma/lut）が none を一貫して上回る（+0.4〜0.5dB, ΔE −0.6〜−0.8）—
  リニアフィットでは線形化が正解、という白キャンバスでの結論を新ターゲットでも再現。
- gamma / lut_gray / lut_ch は誤差範囲で同等（このプロジェクタの応答がほぼ純ガンマ2.2のため）。
- 絶対値が前ターゲット（19.6dB）より低いのは、シートの色勾配・曲面を打ち消すために
  補償が色域限界に当たりやすいため。nature_1（白い霧空）は全条件 14dB 弱で色域律速。

## ファイル

- `matrices_linear_{none,gamma,lut_ch,lut_gray}.npy` — 4条件のリニア空間行列 (1080x1920x4x3)
- `region.json` / `valid_mask.npy` / `square_region_preview.png` — 投影領域 (667,207,692)
- `final_results/<target>/square_<cond>.png` + モンタージュ + `metrics_final.json`
- 実行スクリプト一式（probe_new_surface / graycode_run / region_derive / final_experiment /
  linear_pipeline）— カバーウィンドウ機能は全スクリプトから除去済み
- LUT本体: `data/inverse_lut.npz`（channel）/ `data/inverse_lut_gray.npz`（gray）

## 新しいターゲット画像の処理

`data/comp_test_20260724/apply_compensation_square.py` と同じ流れで、この行列
（`matrices_linear_lut_ch.npy` 等）と領域 (667,207,692) を使う。物理配置・解像度変更時は
probe → LUT再構築（必要なら）→ graycode_run → region_derive → final_experiment の順で再実行。
