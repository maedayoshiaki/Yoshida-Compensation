# `inverse_lut.py` 仕様解説書

| 項目 | 内容 |
| --- | --- |
| 対象モジュール | [src/python/inverse_lut.py](../src/python/inverse_lut.py) |
| 関連ファイル | [src/python/config.py](../src/python/config.py)（`InverseLutConfig`）、[examples/python/build_inverse_lut.py](../examples/python/build_inverse_lut.py)（CLI）、[tests/test_inverse_lut.py](../tests/test_inverse_lut.py)（テスト） |
| 設計書 | [docs/superpowers/specs/2026-07-03-inverse-lut-design.md](superpowers/specs/2026-07-03-inverse-lut-design.md) |

---

## 1. 概要

本モジュールは、プロジェクタの入力信号と出力（発光）との間に存在する非線形性（ガンマ特性）を、
カメラをリニアな測光器として測定し、その非線形性を打ち消すチャンネル独立の 1 次元逆ルックアップ
テーブル（以下、逆LUT）を構築するものである。

- **目的**: プロジェクタ固有の応答特性 `g`（入力 `p` から発光への写像）を測定し、その逆写像
  `g⁻¹` を求める。カメラ側の見え方を合わせることを目的とするものではない。
- **処理の方向**: チャンネル `c` ごとに測定応答 `n_c(p)`（入力から正規化出力への写像）を求め、
  これを反転して `L_c(t)`（目標とするリニア出力 `t` からプロジェクタ入力 `p` への写像）を得る。
- **位置づけ**: 解析的な逆ガンマ補正 `t^(1/γ)`（既存の `apply_inverse_gamma_correction`）に対する
  測定ベースの代替実装である。同一のシグネチャで置き換え可能なように設計している。
- **適用粒度**: グローバル（チャンネルごとに 1 本、計 3 本の 1 次元 LUT）であり、画素単位ではない。

なお、本モジュールは既存の画素単位補償処理（4×3 アフィン変換による `calc_color_mixing_matrices`）
とは別系統であり、既存の補償パイプラインおよび `sample.py` には組み込まれていない独立モジュール
である。

---

## 2. 処理フロー

```
generate_calibration_patterns()   グレーランプ または チャンネル単独ランプ（source により一方のみ）
        │  生パターン（逆ガンマ非適用）を投影し、リニア撮影（ROI 平均、簡易 HDR は任意）
        ▼
run_calibration_capture()  ─────►  CalibrationData
        │                          （入力RGB, 計測値, 記録チャンネル）の集合
        │  入力レベルで集約 → 黒引き・白正規化 → 単調化 → 反転
        ▼
build_inverse_lut()  ───────────►  InverseLut   （(3, lut_resolution) の密LUT、ノット、メタ情報）
        │
        ▼
apply_inverse_lut(image, lut)      逆ガンマ補正の代替として画像へ適用
save_inverse_lut / load_inverse_lut   （.npz 形式で保存・読込）
```

本モジュールの中核となる計測処理は、「リニアフレームを返す callable」を引数として受け取る構造と
している。これにより、実カメラを用いずにモック撮影を注入でき、ハードウェア非依存でのテストが
可能である。投影ウィンドウおよびカメラバックエンドとの接続は `run_calibration_capture` に集約し、
中核処理から分離している。

---

## 3. データ構造

いずれも dataclass として定義している。

| 型 | 主なフィールド | 役割 |
| --- | --- | --- |
| `CalibrationSample` | `input_rgb`（[0,1]）、`image`（H,W,3 RGB）、`kind`（"gray"/"channel"）、`recorded_channels` | 1 枚の投影パターンと、その撮影で記録すべきチャンネル |
| `CalibrationMeasurement` | `input_rgb`、`measured_rgb`（リニア[0,1]、未記録は NaN）、`recorded_channels`、`kind` | 1 サンプルの記録済み計測値 |
| `CalibrationData` | `measurements: list`、`meta: dict` | 計測の集合およびメタ情報（roi, source, hdr など） |
| `InverseLut` | `lut`（3, R）、`t_grid`（R,）、`knots`、`meta` | 構築済みの逆LUT。`lut[c, i]` は「正規化出力 `t_grid[i]` を得るためのチャンネル `c` の入力値」を表す |

---

## 4. 関数仕様（機能別）

### 4.1 パターン生成

- **`normalize_source(source)`**
  サンプリング系統名を検証・正規化する。許容値は `VALID_SOURCES = ("channel_ramp", "gray")` のみ
  であり、廃止された `"both"` を含む不正値は `ValueError` を送出する。1 回の実行では必ず一方の系統
  のみを使用し、両系統を混在させない。

- **`_make_solid_image(rgb01, x, y, dtype)`**（内部関数）
  指定した正規化色の単色 RGB 画像を生成する。`dtype` に応じて uint8 / uint16 / float32 へ量子化する。

- **`generate_calibration_patterns(x, y, num_levels, source, dtype)`**
  `source` に応じて、いずれか一方のランプを [0,1] 等間隔の `num_levels` 点で生成する。
  - `"gray"`: `(v,v,v)` を生成し、カメラの 3 チャンネルすべてを記録対象とする
    （`recorded_channels=(0,1,2)`、`kind="gray"`）。
  - `"channel_ramp"`（既定）: `(v,0,0)` / `(0,v,0)` / `(0,0,v)` を生成し、投影中のチャンネルのみを
    記録対象とする（`recorded_channels=(c,)`、`kind="channel"`）。単独駆動のため、チャンネル間の
    クロストークが混入しない。
  - `num_levels` および `source` が `None` の場合は Config から補完する。`num_levels` は最小 2 に
    切り上げる。

  生成枚数の目安（`num_levels=17` の場合）: `gray` は 17 枚、`channel_ramp` は `1 + (17-1)*3 = 49` 枚
  （全黒パターンは各ランプで共有）。

### 4.2 計測・集約

- **`reduce_frame(frame, roi)`**
  撮影フレームの矩形 ROI `(x, y, width, height)` を平均し、1 組の RGB 値へ集約する（グローバル LUT の
  ため）。`roi=None`、または `width`/`height` が非正の場合は全フレームを平均する。ROI はフレーム境界に
  クリップし、完全に範囲外の場合は全フレームにフォールバックする。

- **`measure_calibration_samples(samples, capture_fn, roi)`**
  各サンプルを `capture_fn` により撮影し、ROI 平均を求める。入力 RGB により重複排除を行い（全黒
  `(0,0,0)` は 1 回のみ撮影）、各サンプルの `recorded_channels` に対応する値のみを格納し、未記録
  チャンネルは NaN として `CalibrationData` を返す。

### 4.3 撮影オーケストレーション（簡易 HDR）

- **`scale_shutter_tv(tv, scale)`**
  Canon のシャッタースピード文字列（`"1/15"` や `"0.5"` の形式）を露光係数でスケールする。
  `0.5` は露光時間を半分（高速シャッター）、`2.0` は倍とする。整形はベストエフォートであり、
  実機で受理される値かは別途検証が必要である。

- **`capture_hdr_linear(capture_at_fn, enabled, trust_low, trust_high, max_stops)`**
  露光ブラケットおよびマージによる簡易 HDR 撮影を行う。
  - `enabled=False` の場合は、基準露光 `capture_at_fn(1.0)` の結果をそのまま返す。
  - `enabled=True` の場合は、飽和した画素（`trust_high` 以上）については露光を短くし（scale 0.5,
    0.25, …）、`1/scale` 倍して復元する。黒潰れした画素（`trust_low` 以下）については露光を長くし
    （scale 2, 4, …）、`1/scale` 倍して復元する。信頼できる画素で非信頼画素を置換し、基準露光の
    リニア単位でダイナミックレンジを拡張する（値は 1.0 を超える場合がある）。
  - `trust_low` / `trust_high` は 8bit 画素値（0–255）で指定し、内部で `/255` によりリニア [0,1] へ
    換算して比較する。

- **`run_calibration_capture(samples, config, camera_backend, show_window, window_name)`**
  実機との接続を担うラッパである。生パターン（逆ガンマ非適用）を全画面投影し、
  `camera.capture_linear_rgb()` によりリニア撮影を行い、`measure_calibration_samples` に委譲する。
  - HDR は `hdr_enabled=true` かつ Canon バックエンド使用時のみ有効となる。非対応バックエンドの
    場合は警告を出力し、単一露光にフォールバックする。
  - ROI は Config の `roi_x` / `roi_y` / `roi_width` / `roi_height` から構築する。
  - 投影ウィンドウは `sample.py` と同様の方式（`namedWindow` / `WINDOW_FULLSCREEN` /
    `moveWindow(pos_x, pos_y)`）に準拠する。`show_window=False` によりウィンドウ生成を無効化できる
    （テスト用途）。
  - `meta` に `source` / `hdr_enabled` / `roi` を記録する。

### 4.4 LUT 構築

- **`_strictly_increasing(x, y)`**（内部関数）: `x` が狭義単調増加とならない点を除去する（最初の点を
  保持）。
- **`_channel_observations(measurements, channel, source)`**（内部関数）: 指定したチャンネルおよび
  系統に一致する `(入力[c], 計測[c])` の対のみを収集する（NaN は除外し、`kind` と `source` を照合して
  混在を排除する）。
- **`_aggregate_by_level(inputs, values)`**（内部関数）: 一意な入力レベルでグルーピングし、値を平均
  してノット化する。
- **`_build_channel_lut(inputs, values, t_grid, trust_low, trust_high)`**（内部関数）: 1 チャンネル分の
  LUT を構築する。処理手順は以下のとおりである。
  1. **黒引き**: `m0 = m(黒)`（黒点が存在しない場合は最小値）を基準とする。
  2. **信頼レンジ判定**: `trust_low ≤ m ≤ trust_high` を満たす点のみを採用する（黒アンカーは常に
     保持）。有効な点が 2 点未満の場合は恒等 LUT にフォールバックする。
  3. **白正規化**: `n = (m − m0) / (m_white − m0)` を [0,1] にクリップする。`m_white − m0 ≤ 0`（縮退）
     の場合は恒等 LUT にフォールバックする。
  4. **端点強制**: `p=0 → n=0`、`p=1 → n=1` を強制する。
  5. **単調化**: 非減少化した後、狭義単調に整形する。
  6. **反転**: 前向き応答を PCHIP 補間で平滑化し、密サンプル（1024 点以上）から `t_grid` へ逆補間
     して `L_c(t)` を得る。

- **`build_inverse_lut(data, source, lut_resolution, trust_low, trust_high)`**（公開 API）
  未指定の引数は Config から補完する。`trust_low` / `trust_high` は 8bit 画素値であるため、`/255` に
  より正規化してから比較する。3 チャンネルそれぞれについて上記処理を実行し、`InverseLut`（`lut`、
  `t_grid`、`knots`、`meta`）を返す。観測が存在しないチャンネルは恒等 LUT（`t_grid`）にフォール
  バックする。

### 4.5 適用・保存・読込

- **`apply_inverse_lut(image, lut, dtype)`**
  RGB 画像（uint8 / uint16 は正規化）をチャンネルごとに `L_c` で線形補間し、プロジェクタ入力画像を
  返す。`apply_inverse_gamma_correction` と置き換え可能なシグネチャである。出力は既定で float[0,1]
  であり、`dtype` により 8bit / 16bit への量子化に対応する。
- **`save_inverse_lut(path, lut)` / `load_inverse_lut(path)`**
  `.npz` 形式（`lut`、`t_grid`、および JSON 文字列化した `meta`）で保存・読込を行う。読込時、`knots`
  は保存対象外のため空となる。

---

## 5. 設定項目（`config.toml` の `[inverse_lut]` セクション）

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `num_levels` | `17` | 各ランプの階調数 |
| `source` | `"channel_ramp"` | サンプリング系統（`channel_ramp` / `gray` のいずれか一方） |
| `lut_resolution` | `256` | 逆 LUT の密グリッド解像度 |
| `roi_x` / `roi_y` | `0` / `0` | 計測 ROI の左上座標（ピクセル） |
| `roi_width` / `roi_height` | `0` / `0` | ROI の幅・高さ（0 の場合は全フレーム平均） |
| `trust_low` | `10.0` | 信頼下限（8bit 画素値、以下は非信頼／黒潰れ扱い） |
| `trust_high` | `245.0` | 信頼上限（8bit 画素値、以上は非信頼／飽和扱い） |
| `hdr_enabled` | `false` | 簡易 HDR ブラケット撮影の有効化 |
| `hdr_max_stops` | `3` | 各方向の最大露光段数 |
| `inverse_lut_path` | `"data/inverse_lut.npz"` | LUT の保存先パス |

---

## 6. テスト方針および実行方法

### 6.1 モックの基本方針

合成応答 `m(p) = black + (white − black) · p^γ`（チャンネル別 γ に対応）により「プロジェクタおよび
カメラ」を模擬し、`capture_fn` として注入する（[tests/test_inverse_lut.py](../tests/test_inverse_lut.py)
の `_synth_frame` を参照）。これにより、実機を用いずに全処理を検証できる。検証の中核は次の往復関係
である。

> `apply_inverse_lut` により生成した入力を応答 `g` に通すと、リニアに復元される
> （`L_c(t) ≈ t^(1/γ)`、許容誤差内）。

### 6.2 既存テストの検証観点

- Config の既定値および TOML 読み込み
- パターン生成（枚数、記録チャンネル、全黒の重複除去、画像形状・dtype）、および不正 `source` の拒否
- `reduce_frame` の ROI / 全フレーム / クリップ挙動
- 計測の重複排除および記録チャンネル（単独ランプは投影チャンネルのみ、他は NaN）
- HDR の単一露光、飽和・黒潰れの回復、`scale_shutter_tv`
- LUT 構築（チャンネル別 γ の復元、黒引き、単調性、飽和除外、系統排他）、および不正 `source` の拒否
- `apply_inverse_lut` の線形化、uint8 往復、save/load 往復、CLI（`build_inverse_lut.py`）

### 6.3 実行コマンド

```powershell
# 本テストファイルのみを実行
uv run pytest tests/test_inverse_lut.py -v
```

> リポジトリ全体で `uv run pytest` を実行した場合、他ファイルの `src` インポートに起因する収集
> エラーにより中断されることがある。テストはファイルを明示して実行することを推奨する。

---

## 7. 対象外事項

以下は本モジュールの範囲外である。

- `sample.py` および補償パイプラインへの組み込み
- 画素単位 LUT（本モジュールはグローバルのみを対象とする）
- 高度な HDR 処理（位置ずれ・ゴースト補正、トーンマッピング等。本モジュールは簡易な露光ブラケット
  およびマージのみを実装する）
- 混合色（n³ グリッド）サンプリング
- カメラ・プロジェクタ間の幾何ワープ
