# ISO100 / ISO160 フル解像度Step 1再校正・色補償評価（2026-08-06）

## 1. 実行範囲

色補償プログラム改良後の再実行として、既存の撮影・投影・Webサーバープロセスを停止し、ISO100とISO160を独立した条件として次を最初から実行した。

1. 黒／全画面白の露出確認
2. **フル解像度Step 1**の全画面GrayCode 46枚を撮影し、ROIを自動導出してdense P2C/C2Pを生成
3. チェッカー実写によるP2C/C2P幾何検証
4. 0/85/170/255の4段階グレー評価
5. チャンネルランプ49撮影とグレーランプ17撮影による2種類の逆LUT構築
6. 線形化なし、解析逆ガンマ、チャンネルLUT、グレーLUTの4方式について、各64色を新規撮影して画素別4×3行列をfit
7. 5ターゲット分の補償画像を方式別に生成
8. 5ターゲット×5条件の実写評価（PSNR / SSIM / Delta E76）
9. 全25組の画素別Delta E76、局所PSNR、局所SSIMマップと3×2比較グリッド生成

1 ISOあたり400撮影、合計800撮影。すべてAv14、Tv 1/15秒、`image_quality="LR"`、フル解像度4660×6984、非HDR、投影位置 `(3440,0)`、60 Hzで実行した。GrayCodeは `height_step=1`、`width_step=1`、`camera_coordinate_scale=1.0` であり、初回から間引いていない。

使用config:

- `config.iso100_nonhdr_step1_rerun_20260806.toml`
- `config.iso160_nonhdr_step1_rerun_20260806.toml`

## 2. EDSDKPythonと保護高速撮影

- EDSDKPythonのcheckoutは `v0.2.0`、HEAD `a8b630f98cf38f77ed1652268c428b9d98499a3a`。確認時点で `origin/main` とタグ `v0.2.0` も同一コミットだった。
- 実際にimportされるモジュールは `.venv/Lib/site-packages/edsdk/`。改行差を正規化したソース比較で上記checkoutと一致した。
- Yoshida側の全 `CameraController` 生成を `protected=True, register_property_events=False` に統一した。既存の `arm_triggered_capture(transfer="memory")` 高速経路も維持した。
- カメラ再接続後、ISO100の黒／白2枚で保護高速撮影を実機確認した。トリガ5.16秒、撮影＋現像19.20秒で成功した。
- その後の本再校正800撮影も同じ保護高速経路で完了し、終了時のEDSDKスプール残留ファイルと関連プロセスはいずれも0件だった。

## 3. 幾何とROI

| 項目 | ISO100 | ISO160 |
| --- | ---: | ---: |
| GrayCode step | `(1,1)` | `(1,1)` |
| パターン数 | 46 | 46 |
| 自動導出ROI | `(566,217,784,784)` | `(591,213,788,788)` |
| GrayCode対応点数 | 9,038,630 | 10,462,066 |
| 復号有効プロジェクタ画素 | 902,166 | 1,014,838 |
| 最大連結成分面積 | 806,583 | 939,907 |
| 内接円中心 | `(958,609)` | `(985,607)` |
| 内接円半径 | 400 | 402 |
| active外挿率 | 0.03661% | 0% |
| P2C位相ずれ `(x,y)` [px] | `(-0.073,+0.334)` | `(-0.134,-0.194)` |
| P2C内部相関 | 0.9271 | 0.9270 |
| C2P位相ずれ `(x,y)` [px] | `(-0.095,+0.312)` | `(-0.157,-0.200)` |
| C2P内部相関 | 0.9257 | 0.9255 |

P2C activeの有限行数はROI面積と一致した（ISO100: 614,656、ISO160: 620,944）。幾何検証の位相ずれは両方向ともサブピクセルである。

## 4. 露出、4段階グレー、逆LUT

| 項目 | ISO100 | ISO160 |
| --- | ---: | ---: |
| 全画面黒・確認ROI平均 | 0.000237 | 0.000390 |
| 全画面白・確認ROI平均 | 0.211311 | 0.413623 |
| 全画面白・`any >= 0.98` | 0% | 0.773% |
| 4段階グレー厳密単調 | yes | yes |
| コード255・確認ROI平均 | 0.206273 | 0.432157 |
| コード255・`any >= 0.98` | 0% | 2.155% |

ISO160では白側に軽い単一露光飽和がある。指定されたISO160・Av14・Tv 1/15秒・非HDR条件は変更せず、その特性を含めて校正・評価した。

両ISO・両sourceの逆LUTはすべて `(3,256)` float32、全値有限、単調非減少、入力／出力端点0→1だった。チャンネルランプとグレー投影のLUTは混在させず、対応する64色行列と補償生成だけに使用した。

## 5. 5条件評価

5枚のターゲットに対する平均。PSNR/SSIMは大きいほど、Delta E76は小さいほど良い。

### ISO100

| 条件 | PSNR [dB] | SSIM | Delta E76 |
| --- | ---: | ---: | ---: |
| no_comp | 12.941 | 0.7027 | 30.707 |
| comp_no_gamma | 15.962 | 0.7142 | 17.510 |
| comp_gamma | 16.149 | 0.7270 | 18.001 |
| comp_inverse_lut | **16.161** | **0.7331** | **16.439** |
| comp_inverse_lut_gray | 16.068 | 0.7271 | 17.915 |

チャンネルランプLUTが3指標すべてで最良。未補償比でPSNR +3.220 dB、SSIM +0.0304、Delta E76 -14.269。

### ISO160

| 条件 | PSNR [dB] | SSIM | Delta E76 |
| --- | ---: | ---: | ---: |
| no_comp | 14.781 | **0.7606** | 27.717 |
| comp_no_gamma | 17.324 | 0.7163 | 16.594 |
| comp_gamma | 17.537 | 0.7391 | 16.766 |
| comp_inverse_lut | **17.607** | 0.7549 | **16.355** |
| comp_inverse_lut_gray | 17.378 | 0.7367 | 17.079 |

PSNRとDelta E76はチャンネルランプLUTが最良。SSIMは未補償が最高で、補償方式内ではチャンネルランプLUTが最高。未補償比でPSNR +2.827 dB、Delta E76 -11.362、SSIM -0.0058。

## 6. 成果物完全性

各ISOについて次を確認した。

- 4方式すべてでlinear pattern 64枚、projected pattern 64枚、capture 64枚、校正metadata、行列が存在
- 8行列すべて `(4660,6984,4,3)` float32、1,562,181,248 bytes、全値有限
- 最終評価capture 25枚、warped画像25枚、比較画像5枚
- 画素別raw map 75配列、個別可視化75枚、条件比較row 25枚、3×2比較グリッド5枚
- P2C active有限行数がROI面積と一致
- 撮影終了後のEDSDKスプール残留ファイル0件、関連プロセス0件

主要成果物のSHA-256:

| ISO | 種別 | SHA-256 |
| --- | --- | --- |
| 100 | P2C active | `43A33F0C7A9FC1CF0EB8572EB3F618E47A20AD99D82683F86D7CF74BDE7FF136` |
| 100 | LUT channel | `D030F198B3E0B31E4EC881A1F00B40E8B8CCEAE15F195BFC5B7C8E6EEE0FB0B2` |
| 100 | LUT gray | `498E814BBE227604E8F4B2239CB0DC3D60C12CF86DD2FBA3E616C9E75E1435C4` |
| 100 | matrix none | `161AC9BDD1536816BBBB8A6091BC880B981E1B503F67E3A693ED437B47A00092` |
| 100 | matrix gamma | `FCDD0D833989A32C467357EDD875B8CDD9D4491EA160E341834F728F980BA5AD` |
| 100 | matrix channel LUT | `266302D7E37DD19346061E02F3A8FDCBC6A3CC3F3C319219A596D2209169C766` |
| 100 | matrix gray LUT | `10E5D96785429FDD2273C199EC377A531B876603E9B5F539523C6A4275F19E6C` |
| 160 | P2C active | `85B52C4566FFF9C376C7625636B181C2E1E143362484A11CA46A0BE9B7858AAD` |
| 160 | LUT channel | `E0CB2FB1FDFEAF323B42033A6B2CAF0ACE21073B90AFB62A300B1A4A7139D27D` |
| 160 | LUT gray | `DD098816F957A14857448056A2F00E7C27FC0F3DD90EC3D9E135195BD12D0BEF` |
| 160 | matrix none | `9F98B76B652C5EF9295152AD7F42129F5143E6E5DB376CFE74E00C9BBAE3D329` |
| 160 | matrix gamma | `977122E027317C7D0E2D47C6ADC8E6122FF9F0BD9E05778B13AC96A9B4C47AF8` |
| 160 | matrix channel LUT | `57423D6451A9B86CFB814FE00D40D79CC1165E59BFEDD155E1E7CF94162A3FE1` |
| 160 | matrix gray LUT | `050C9792644F85D43A1835B628C0D92E991C3F3660E2488E87BD6F963B648D25` |

## 7. 成果物と注意点

主要な結果JSON:

- `data/iso100_nonhdr_full_20260806_step1_rerun/evaluation_requested5/metrics_requested5.json`
- `data/iso160_nonhdr_full_20260806_step1_rerun/evaluation_requested5/metrics_requested5.json`

比較グリッド:

- `data/iso100_nonhdr_full_20260806_step1_rerun/comparison_grids_3x2/`
- `data/iso160_nonhdr_full_20260806_step1_rerun/comparison_grids_3x2/`

注意点:

- `evaluate_compensation_conditions.py` の出力JSONは `setup_date` を `2026-07-30` にハードコードしている。実行日は2026-08-06であり、本書と成果物ディレクトリの日付を正本とする。
- 色評価はCanon RAW現像後のlinear sRGBとsRGBターゲットを用いた画像ベース評価で、測色器による絶対色評価ではない。
- 巨大行列のmmapワープ時にPyTorchから「NumPy array is not writable」警告が出た。入力への書き込みはなく、4方式×2 ISOの生成はすべて成功した。

## 8. ソフトウェア検証

```text
440 passed, 1 warning in 277.18s
```

実行コマンドは `uv run --no-sync python -m pytest -q`。警告は既知の `.pytest_cache/v/cache/nodeids` への書き込み権限エラーだけで、テスト失敗は0件。

追加確認:

- YoshidaのCanon EDSDK関連テスト: 49 passed
- EDSDKPythonの非実機worker/trigger/lifecycleテスト: 34 passed
- 実機の保護高速撮影: ISO100の2枚スモークテストおよび本再校正800撮影が完了
