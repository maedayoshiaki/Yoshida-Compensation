# ISO100 / ISO160 全再校正・色補償評価（2026-08-06）

## 1. 実行範囲

既存の撮影・投影・Webサーバープロセスを停止し、ISO100とISO160を独立した条件として、次を最初から実行した。

1. 黒/全画面白の露出確認
2. 全画面GrayCode 38枚の撮影、復号、ROI自動導出、dense P2C/C2P生成
3. チェッカー実写によるP2C/C2P幾何検証
4. 0/85/170/255の4段階グレー評価
5. チャンネルランプ49撮影とグレーランプ17撮影による2種の逆LUT構築
6. 線形化なし、解析逆ガンマ、チャンネルLUT、グレーLUTの4方式について、各64色を新規撮影して画素別4×3行列をfit
7. 5ターゲット分の補償画像生成
8. 5ターゲット×5条件の実写評価（PSNR / SSIM / Delta E76）
9. 全25組の画素別Delta E76、局所PSNR、局所SSIMマップと3×2比較グリッド生成

1 ISOあたり392撮影、合計784撮影。すべてAv14、Tv 1/15秒、`image_quality="LR"`、フル解像度4660×6984、非HDR、投影位置 `(3440,0)`、60 Hzで実行した。RAW現像の安定化のため、64色校正と最終評価は4枚単位のトリガバッチを用いた。撮影順・露出・教師値は変更していない。

使用config:

- `config.iso100_nonhdr_full_20260806.toml`
- `config.iso160_nonhdr_full_20260806.toml`

## 2. 幾何とROI

| 項目 | ISO100 | ISO160 |
| --- | ---: | ---: |
| 自動導出ROI | `(578,214,784,784)` | `(583,214,786,786)` |
| GrayCode対応点数 | 12,409,241 | 14,282,812 |
| 復号有効プロジェクタ画素 | 914,452 | 1,123,211 |
| 内接円半径 | 400 | 401 |
| active外挿率 | 0.00537% | 0% |
| P2C位相ずれ `(x,y)` [px] | `(+0.019,-0.178)` | `(-0.010,+0.097)` |
| P2C内部相関 | 0.9315 | 0.9314 |
| C2P位相ずれ `(x,y)` [px] | `(-0.008,-0.014)` | `(+0.001,-0.003)` |
| C2P内部相関 | 0.9417 | 0.9412 |

ISOごとに全画面投影からROIを独立導出した。full-frame runで得たdense P2Cは、その実行で導出されたROIへすでにマスク済みである。

## 3. 露出と4段階グレー

| 項目 | ISO100 | ISO160 |
| --- | ---: | ---: |
| 全画面黒・確認ROI平均 | 0.000233 | 0.000388 |
| 全画面白・確認ROI平均 | 0.205225 | 0.409159 |
| 全画面白・`any >= 0.98` | 0% | 0.584% |
| 4段階グレー厳密単調 | yes | yes |
| コード255・確認ROI平均 | 0.205186 | 0.420836 |
| コード255・`any >= 0.98` | 0% | 0.881% |

ISO160では白のごく一部に単一露光飽和がある。ユーザー指定どおりISO160・非HDRの条件は変更せず、そのまま測光行列と評価へ使用した。

逆LUTは両ISO・両sourceとも `(3,256)` float32、全値有限、単調増加。終端値は次のとおり。

| ISO | source | R | G | B |
| --- | --- | ---: | ---: | ---: |
| 100 | channel_ramp | 1.0000 | 1.0000 | 1.0000 |
| 100 | gray | 1.0000 | 1.0000 | 1.0000 |
| 160 | channel_ramp | 1.0000 | 1.0000 | 0.9384 |
| 160 | gray | 1.0000 | 1.0000 | 1.0000 |

## 4. 5条件評価

5枚のターゲットに対する平均。PSNR/SSIMは大きいほど、Delta E76は小さいほど良い。

### ISO100

| 条件 | PSNR [dB] | SSIM | Delta E76 |
| --- | ---: | ---: | ---: |
| no_comp | 12.775 | 0.6969 | 30.474 |
| comp_no_gamma | 15.645 | 0.7089 | 18.052 |
| comp_gamma | 15.814 | 0.7224 | 18.676 |
| comp_inverse_lut | **15.834** | **0.7270** | **17.011** |
| comp_inverse_lut_gray | 15.717 | 0.7183 | 18.565 |

チャンネルランプLUTが3指標すべてで最良。未補償比でPSNR +3.060 dB、SSIM +0.0301、Delta E76 -13.463。

### ISO160

| 条件 | PSNR [dB] | SSIM | Delta E76 |
| --- | ---: | ---: | ---: |
| no_comp | 14.928 | **0.7710** | 27.395 |
| comp_no_gamma | 17.279 | 0.7087 | 16.685 |
| comp_gamma | 17.380 | 0.7443 | 17.045 |
| comp_inverse_lut | 17.659 | 0.7515 | **15.134** |
| comp_inverse_lut_gray | **18.010** | 0.7582 | 15.381 |

PSNRはグレーLUT、Delta E76はチャンネルLUTが最良。SSIMは未補償が最高で、補償方式内ではグレーLUTが最高。グレーLUTの未補償比はPSNR +3.081 dB、Delta E76 -12.015。Delta E76最小のチャンネルLUTでは -12.261。

## 5. 成果物完全性

各ISOについて次を確認した。

- 4方式すべてでprojected pattern 64枚、capture 64枚、校正metadata、行列が存在
- 8行列すべて `(4660,6984,4,3)` float32、1,562,181,248 bytes、全値有限
- 最終評価capture 25枚、warped画像25枚、比較画像5枚
- 画素別raw map 75配列、可視化row 25枚、3×2比較グリッド5枚
- P2C active有限行数がROI面積と一致（ISO100: 614,656、ISO160: 617,796）
- 撮影終了後のEDSDKスプールに残留ファイルなし

主要成果物のSHA-256:

| ISO | 種別 | SHA-256 |
| --- | --- | --- |
| 100 | P2C active | `2EA5C59B704AC8963A216E125A4371370AB9DDEC6E682483C79EEAB8A6E3AAF0` |
| 100 | LUT channel | `1D1271D8B4D7A27FB793B912AB766A1A593EEE00CCE43E756E5A9F6365E1B16F` |
| 100 | LUT gray | `30B177983C5B0FB6CCD6F0530D22847A24EED2083C9435653CEAD74DC1C2035B` |
| 100 | matrix none | `FD227F2111E0A4FC446097F3014376A1B7EC82089A31A7947A40696B9512EC17` |
| 100 | matrix gamma | `2618E9A200E93D2F3DCBC1F2BC49CEA7960F778244CB2090B9B2E5FA876C90B8` |
| 100 | matrix channel LUT | `1DCB3BCFB7867C1258B6540EB1FE5C99BB25EE4AB93C87019A95DFB76315C4E0` |
| 100 | matrix gray LUT | `51215D9F5C0E671F308306B473DE8488E47AB0675884274418A45706A04140CF` |
| 160 | P2C active | `3E441998131350FAFB4311E219B5FB42C5795684714126B47670BE5622D0216B` |
| 160 | LUT channel | `1B378DD4874476739FB50638227542511366BF28B40341EE49EE1D8920A8F9E8` |
| 160 | LUT gray | `5ECA697CC040FB241B33965035C0C133DD0AA5AECE2D851B11EDFA7FF57FB3C2` |
| 160 | matrix none | `C0E0ABB46F860E99F46C4858F3A237D4BA043349B176C64A8ADC16143247B908` |
| 160 | matrix gamma | `1329096A325CF2398BC0F22E3DDC866A9BE10A137B3AB78B25F0B4A0339A65EE` |
| 160 | matrix channel LUT | `D7BDF4EBB803A6DB289C45CDA931E812C4D8D782FE7AA546A8F046E5C068B7C6` |
| 160 | matrix gray LUT | `BC007F29A765A3116195CDC09046A9E6308B2C866131C897A6FED9EA4988EB3A` |

## 6. 注意点

- `evaluate_compensation_conditions.py` の出力JSONは `setup_date` を `2026-07-30` にハードコードしている。実行日と成果物ディレクトリの日付は2026-08-06であり、本書を日付の正本とする。
- 色評価はCanon RAW現像後のlinear sRGBとsRGBターゲットを用いた画像ベース評価で、測色器による絶対色評価ではない。
- 巨大行列のmmapワープ時にPyTorchから「NumPy array is not writable」警告が出た。入力へ書き込む処理はなく、4方式×2 ISOの生成はすべて成功した。

主要な結果JSON:

- `data/iso100_nonhdr_full_20260806/evaluation_requested5/metrics_requested5.json`
- `data/iso160_nonhdr_full_20260806/evaluation_requested5/metrics_requested5.json`

## 7. ソフトウェア検証

```text
432 passed, 1 warning in 254.57s
```

実行コマンドは `uv run --no-sync python -m pytest -q`。警告は既知の `.pytest_cache/v/cache/nodeids` への書き込み権限エラーだけで、テスト失敗は0件。
