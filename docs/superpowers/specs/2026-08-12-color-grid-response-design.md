# 3チャンネル混色応答の実測による線形化検証 — 設計書

日付: 2026-08-12 / 状態: ユーザー承認済み（計測規模・駆動系・撮影条件・実行方針）

## 1. 目的

現行パイプラインの測光モデル「チャンネル毎 1D 線形化（逆LUT/ガンマ）→ 4×3 カラーミキシング
行列（バイアス行含む）」が、実プロジェクタ→カメラ応答をどこまで説明できるかを、
**入力RGB空間を面的にカバーする実測**で定量化する。

背景となる既知の事実:

- プロジェクタは加法的でない。グレー駆動と単チャンネル駆動の応答が最大 1.7 倍違う
  （AGENTS.md §7.3）。単チャンネル LUT では原理的にグレーを線形化しきれない。
- 線形化 4 方式の評価指標差は小さい（PSNR 0.4dB / ΔE 数単位）が、これは「行列 fit が
  線形化誤差をある程度吸収した後」の残差であり、線形化そのものの良否を直接見ていない。

本計測は**単一の空間点（LUT ROI の平均）における色応答の診断**である。画素ごとの
空間変動は対象外（現行の逆LUT計測と同じ縮約）。

## 2. 先行研究の根拠（2026-08-12 文献調査）

| 文献 | 本設計への根拠 |
| --- | --- |
| Nayar et al., PROCAMS 2003 | `C = V·p(I)` 型（1D非線形+3×3）の原典。V 推定は単チャンネル差分のみ → **gray の非加法性はこのモデルでは原理的に捉えられない**（DLPの白セグメントによる混色複雑化を脚注で自認） |
| Grossberg et al., CVPR 2004 | `C = V·p(I) + F` — **黒オフセット項 F を必ず fit に含める**形式の原典 |
| Grundhöfer & Iwai, IEEE TIP 2015 | **n³（n=4〜6）のフルスクリーン単色投影**から、チャンネル交差を含む色写像を正則化 thin-plate-spline 散布補間で獲得。本設計の 6³=216 グリッドはこの標準設計。scipy `RBFInterpolator(kernel="thin_plate_spline")` で等価実装可能 |
| Sajadi et al. (ADICT), ECCV 2010 | 非加法性 `T(I₁)+T(I₂)≠T(I₁+I₂)` を 9³=729 計測で実証。**1D LUT+3×3 型はDLPでΔE約2倍劣化**。理想モデル残差のΔE表化という評価設計を踏襲 |
| Seime & Hardeberg, J. SID 2003 | 加法性の直接実測（原色和 vs 白: LCD 0.33% / DLP 56%）。本計測の加法性検査と直接対応 |

原文PDFはスクラッチパッド/tool-results に取得済み（セッション限り）。

## 3. 計測設計

### 3.1 パッチセット（生コード値駆動、計 321 ユニーク色）

| セット | 定義（8bit） | 枚数 | 新規 |
| --- | --- | ---: | ---: |
| GRID6 | {0,51,102,153,204,255}³ の全組合せ | 216 | 216 |
| RAMP17×4 | R/G/B 単独と gray の 17 段（round(i·255/16), i=0..16） | 68 | 60 |
| DIAG17×3 | (v,v,0)/(v,0,v)/(0,v,v) の 17 段 | 51 | 45 |

重複（0 と 255 の端点はグリッドと一致）は入力8bit値で排除する。

### 3.2 LUT通し確認セット（gray-fit 逆LUT 駆動、計 35 色）

`data/iso200_hdr_isoscan_d65_20260807/inverse_lut_gray/inverse_lut.npz`（同一条件で計測済み）
を通した投影で「線形化後に本当に線形か」を直接検証する。

- gray 17 段、R/G/B 単独 4 段（{51,102,153,204}）、混合 6 色
  （(128,128,0), (128,0,128), (0,128,128), (192,64,64), (64,192,64), (64,64,192)）。
- 記録は「要求8bit値」と「LUT適用後に実投影した8bit値」の両方を持つ。

### 3.3 ドリフト監視

30 パッチごとに番兵色 (128,128,128) を再計測し（約12回）、タイムスタンプ付きで記録する。
約4時間の連続運転中のランプ/カメラの漂移を解析時に検出・補正できるようにする。
番兵は重複排除の対象外。

### 3.4 投影順序

固定シード（seed=0）でシャッフルする。入力レベルと時間ドリフトの相関を切るため。

### 3.5 逐次保存と再開

- 1 パッチ計測するごとに JSONL へ 1 行追記（`measurements.jsonl`）。
  行の内容: index / drive（raw・lut_gray・sentinel）/ 要求8bit RGB / 実投影8bit RGB /
  ROI平均リニアRGB / ROI標準偏差 / 使用露光（tv と実効比のリスト）/ ISO時刻。
- 起動時に既存 JSONL を読み、計測済み（drive, 要求RGB）をスキップして再開できる。
  番兵はスキップしない。
- 完了時に `metadata.json`（config 内容・パッチセット定義・LUTパス・WB確認結果・
  総所要時間・パッチ単価）を書く。

### 3.6 撮影機構（既存コードの再利用）

- カメラ: `create_camera_backend` + 単一セッション（`with backend:`）。
- 露光: `snap_to_flicker_free_tv`（1/15, 60Hz）。HDR は `capture_hdr_linear`
  （trust 10/245、±3段、LUT ROI をゲートに使用）。
- 計測: `reduce_frame` で LUT ROI (4520,1600,480,400) の平均。標準偏差も同 ROI で取る。
- 投影: `active_region` (607,209,794,794) にマスクした単色を cv2 フルスクリーン表示、
  `wait_key_ms=300` + settle 0.3 s（`_PROJECTION_SETTLE_SECONDS`）。
- 本計測前に黒/白のスモークテストを行い、白の ROI 平均が既知レンジ
  （isoscan 実測とオーダー一致）にあることを確認してから続行する。

### 3.7 撮影条件と config

`config.color_grid_iso200_hdr_d65_20260812.toml`（新規、リポジトリルート）。
`config.iso200_hdr_isoscan_d65_20260807.toml` を複製し、出力パスのみ変更。
ISO200 / av14 / tv1/15 / HDR / D65。active_region は導出済み (607,209,794,794)
（`derived_active_region.json` と一致することを確認済み）。

WB はカメラ本体保持の ColorTemp 6500K を前提とする（08-07 D65 キャンペーンで設定）。
スクリプト冒頭で EDSDK から読み戻しを試み、結果（または「読み戻し不可」）を
metadata に記録する。読み戻し失敗は中断理由にしない。

### 3.8 出力先

`data/color_grid_response_20260812/`（実測記録の慣例に従う日付つき凍結ディレクトリ）。
デバッグ crop は容量を考慮し既定 OFF（`--debug-dir` 指定時のみ）。

## 4. 解析設計（オフライン・再実行可能）

スクリプト: `examples/python/analyze_color_grid_response.py`（matplotlib / scipy 1.16 使用）。
入力は `measurements.jsonl` のみ。出力は PNG 群 + `summary.json`。

1. **ドリフト検査**: 番兵色の時系列。変動が ±1% を超える場合は区分線形で正規化
   （適用有無を summary に記録）。
2. **ランプ線形性**: 生駆動（素のガンマ）と LUT 通し駆動の R/G/B/gray ランプを
   理想対角線と比較（`validate_inverse_lut.py` と同じ正規化・指標: max_dev / RMSE）。
   → 「ガンマ補正（線形化）がうまくいっているか」への直接回答。
3. **加法性**: グリッド点で `f(r,g,b) − f(0)` vs `Σ_c (f(単独c) − f(0))` の比と差。
   Seime & Hardeberg 型の「原色和 vs 白」を 6 レベルで表化。B レベル別スライスの
   ヒートマップ。
4. **クロストーク曲線**: 固定 (r,b) に対する G 入力掃引時の計測 R/B の変化 等、
   軸外応答の可視化。
5. **モデル比較**（すべて同じ 321 点から fit、混合色での残差を報告）:
   - M1: 解析ガンマ 2.2 + 4×3 行列（バイアス行含む、以下同じ）
   - M2: channel_ramp 由来 1D LUT + 4×3 行列
   - M3: gray 由来 1D LUT + 4×3 行列
   - M4: 正則化 TPS（`RBFInterpolator`, thin_plate_spline, smoothing はLOOCVで選択）
   - M5: 2次 root-polynomial（交差項 √(rg) 等を含む線形回帰）— 中間の自由度の参考
   指標: リニアRGB RMSE / 最大誤差 / ΔE76（白点は実測 (255,255,255)）。
   fit は (a) 全点 fit → LOOCV、(b) 軸+gray のみ fit → 混合色で検証、の2通り。
   M1〜M3 と M4 の残差差が「1D+行列モデルが原理的に取りこぼす量」の推定値になる。
6. **LUT通し確認との突合**: M3 で予測した LUT 通し応答と実測（§3.2）の一致を検査
   （数値合成の妥当性確認）。

## 5. 実装

| ファイル | 内容 |
| --- | --- |
| `examples/python/measure_color_grid_response.py` | パッチ生成・投影・HDR撮影・JSONL逐次保存・再開 |
| `examples/python/analyze_color_grid_response.py` | §4 の解析・プロット・summary |
| `config.color_grid_iso200_hdr_d65_20260812.toml` | 撮影条件（§3.7） |
| `tests/test_color_grid_response.py` | 純粋部分のテスト |

テスト（実機に依存しない純粋関数のみ。落ちることができるテストにする）:

- パッチセット: 総数がちょうど 321、端点重複が排除されること、シャッフルが
  シード固定で再現すること。
- 再開ロジック: JSONL 途中再開で計測済みが投影されず、番兵は再計測されること。
- 加法性計算: 加法的な合成データで比≈1、非加法な合成データで比≠1 になること。
- モデル fit: 「V·p(I)+f で生成した合成データは M2 残差≈0」かつ
  「非加法合成データでは M2 残差 ≫ M4 残差」の両方向を検査する
  （片方だけでは fit の実装ミスが通り抜ける）。

## 6. リスクと制約

- **所要時間 約4時間**（HDR 実測 41 s/枚 × 356 枚 + 番兵12枚）。ドリフトは番兵で監視。
- **WB 前提**: 本体 WB が 6500K のままである前提。読み戻し不可の場合は前提のまま
  進み、metadata に記録する。
- **単一空間点**: LUT ROI 平均のみ。画素別の混色差（V の空間変動）は対象外。
- **ISO200 HDR は補償評価の最良条件（ISO400 非HDR）と異なる**: 本計測は応答の
  「形」の診断が目的であり、暗部SNRを優先して HDR を選択（ユーザー指定）。
  条件間で応答形状が大きく変わらないことは isoscan の LUT 比較で概ね確認済みだが、
  ISO400 非HDR への外挿は解析で明示的に注意書きする。
- 撮影失敗時: EDSDK の再試行は既存機構に任せ、パッチ単位の失敗は JSONL に
  error 行を残して続行（最後に失敗一覧を報告、再実行で埋められる）。

## 7. 実行手順

```bash
# スモークテスト込みの本計測（約4時間）
uv run --no-sync python examples/python/measure_color_grid_response.py \
  --config config.color_grid_iso200_hdr_d65_20260812.toml \
  --lut data/iso200_hdr_isoscan_d65_20260807/inverse_lut_gray/inverse_lut.npz \
  --output-dir data/color_grid_response_20260812

# 解析（撮影後、何度でも）
uv run --no-sync python examples/python/analyze_color_grid_response.py \
  --input data/color_grid_response_20260812/measurements.jsonl \
  --output-dir data/color_grid_response_20260812/analysis
```
