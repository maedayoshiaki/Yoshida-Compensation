# 逆LUT（測定ベース プロジェクタ応答線形化）設計書

- 日付: 2026-07-03
- 対象: `src/python/inverse_lut.py`（新規）ほか
- ステータス: 設計確定（実装計画へ）

## 1. 目的とセマンティクス

プロジェクタの入力→出力（発光）間の非線形性（ガンマ特性など）を、カメラを
**リニアな測光器**として測定し、それを打ち消す **チャンネル独立の 1D 逆LUT** を作る。

- **正体**: 「カメラで見た見え方を合わせる」ものではない。あくまで **プロジェクタ固有の
  応答 `g`（入力 → 発光）** を測り、その逆 `g^{-1}` を作る。
- **向き**: チャンネル `c` ごとに測定応答 `n_c(p)`（入力 `p` → 正規化出力）を作り、
  **反転**して `L_c(t)`（欲しいリニア出力 `t` → プロジェクタ入力 `p`）を得る。
- **位置づけ**: 解析的な `t^(1/γ)`（既存 `color_mixing_matrix.apply_inverse_gamma_correction`）
  の **測定版**。概念的には逆ガンマ補正の置換だが、**今は `examples/python/sample.py` へは
  組み込まず、独立モジュールとして実装する**。
- **粒度**: グローバル（チャンネルごと 1 本、計 3 本の 1D LUT）。ピクセルごとではない。

### 既存パイプラインとの関係

- 既存の `calc_color_mixing_matrices`（ピクセルごと 4×3 アフィン）は **別系統**であり本設計では変更しない。
- 既存 `generate_projection_patterns` は **流用**する（後述）。
- `sample.py` の色補償撮影は **逆ガンマ済みパターン**を投影し `capture_image()`（sRGB uint8）で
  撮影する。本 LUT は目的が異なるため、この撮影データは再利用しない（§4 参照）。

## 2. モジュール構成 / データフロー

新規 `src/python/inverse_lut.py`:

```
generate_calibration_patterns()   # 粗い混合グリッド ∪ 細かいグレーランプ（生成器は既存流用）
        ↓  生パターン投影 ＆ リニア撮影（中央領域平均, 簡易HDR=仮実装）
run_calibration_capture()  →  CalibrationData（(入力RGB, 計測RGB) の対の集合）
        ↓  入力レベルで周辺平均 → 黒引き・白正規化 → 単調化 → 反転
build_inverse_lut()  →  InverseLut（(3, lut_resolution) の密LUT + ノット + メタ）
        ↓
apply_inverse_lut(image, lut)      # 逆ガンマ補正のドロップイン代替
save_inverse_lut / load_inverse_lut  (npz)
```

- **HW 非依存の分離**: コア計測は「リニアフレームを返す callable」を受け取る形にし、
  テストでモック可能にする。投影ウィンドウ＋カメラバックエンド配線は薄いラッパに閉じる。
- CLI: `examples/python/build_inverse_lut.py`（`--config` 対応、生成→撮影(or ロード)→構築→保存、
  `--apply <img>` で適用デモ）。**`sample.py` は変更しない。**

### 主要データ構造

- `CalibrationSample`: `input_rgb: tuple[float,float,float]`（[0,1]）, `image: np.ndarray`, `kind: str`（"grid"|"gray"|"channel"）。
- `CalibrationMeasurement`: `input_rgb: (3,)`, `measured_rgb: (3,)`（リニア [0,1]）。
- `CalibrationData`: `measurements: list[CalibrationMeasurement]` + メタ（reference_region, 露光条件など）。
- `InverseLut`: `lut: np.ndarray (3, R)`（[0,1]）, `t_grid: np.ndarray (R,)`, `knots: list[(p_i, n_i) per ch]`, `meta: dict`。

## 3. キャリブレーションのサンプリング

`num_divisions^3` グリッドは **グレー（対角 (v,v,v)）・単独チャンネル（軸 (v,0,0) 等）・
混合（内部 (0.25,0,0.5) 等）を全て内包**する。よって専用生成器は作らず、既存
`color_mixing_matrix.generate_projection_patterns(width, height, num_divisions)` を流用する。

サンプリング対象 = **粗い混合グリッド ∪ 細かいグレーランプ**:

- **粗い混合グリッド**: `generate_projection_patterns(w, h, grid_divisions)`。`grid_divisions`
  既定 **3**（= 27 枚）。軸（単独チャンネル）＋混合を粗く網羅し、周辺平均のロバスト性を供給。
- **細かいグレーランプ**: `(v,v,v)`, `v ∈ {i/(gray_levels-1)}`。`gray_levels` 既定 **17**。
  1 回のグレー掃引で **R/G/B 各チャンネルの応答曲線を細かく取得**できる（クロストーク
  非考慮の前提下では camera_c は projector_c 入力のみに依存）。これが主・高解像度データ。
- **単独ランプ（任意）**: `use_channel_ramps` 既定 `false`。粗い単独点はグリッド軸で足りるため。
- 入力 RGB が完全一致するパターンは重複除去 → **合計 約 41 枚**（HDR 仮実装 = 1 露光なら 41 撮影）。

グリッド解像度を上げると `grid_divisions^3` で撮影枚数が増える点に注意（config で調整可能）。

## 4. 撮影・計測（正しさの要）

- **生パターン（逆ガンマ非適用）を投影**する。逆ガンマを先に掛けると測っているのは「既に
  線形化された出力」になり、`g` を打ち消す情報が消える（＝逆ガンマの代替を作れない）。
- 計測は `camera.capture_linear_rgb()` の **リニア値**（[0,1] float32）を使用する。
  `sample.py` の `capture_image()`（`linear_to_srgb` を掛けた sRGB uint8）は使わない。
- 各撮影は **中央領域（既定 中央 20%, `reference_region`）の平均**で 1 組の計測 RGB に集約
  （グローバル LUT のため）。
- 投影ウィンドウは `sample.py` 流儀（`namedWindow`/`WINDOW_FULLSCREEN`/`moveWindow(pos_x,pos_y)`/
  `waitKey(wait_key_ms)`）を踏襲。
- **信頼レンジ**: 計測値（[0,1] 換算）の `≤ trust_low`（既定 10/255 ≈ 0.039）/
  `≥ trust_high`（既定 245/255 ≈ 0.961）はサチュ/非信頼として **構築時に除外**（config 可）。

### 簡易 HDR（仮実装）

- インターフェースだけ用意し、当面は **単一露光**（`hdr_enabled=false`）。
- 将来のブラケット仕様は TODO コメントで明示: サチる→ tv 半分・画素値 2 倍、黒潰れ→
  tv 倍・画素値 1/2、信頼レンジ外をマスクして露光間でマージ。
- Canon の `tv` をショットごとに切り替える方式（config を一時上書き）を想定するが、
  今回は骨組みのみ。

## 5. LUT 構築（核心）

チャンネル種別に依存しない周辺集約で構築する。チャンネル `c` ごと:

1. **周辺集約**: 全撮影から `(入力[c], 計測[c])` を収集し、**入力レベルでグルーピングして
   平均** → ノット `(p_i, m_i)`。グレー（細）とグリッド（粗）の観測が自然に統合される。
2. **黒引き・白正規化**: `m0 = m(0)`（黒/環境光）, `m_white = m(最大入力)`。
   `n_i = (m_i − m0) / (m_white − m0)` → `n(0)=0, n(最大)=1`。
   これで「0,0,0 でカメラ 10 なら LUT 値は 0」要件を黒引きで満たす。
   `m_white − m0 ≤ 0`（縮退）の場合はエラー/警告し恒等 LUT にフォールバック。
3. **単調化**: 非減少に補正（`scipy` 利用: PCHIP または等調回帰）。信頼レンジ外の点は除外済み。
   有効ノットが 2 点未満なら恒等にフォールバックし警告。
4. **反転**: 一様 `t` グリッド（`lut_resolution` 既定 256）に対し
   `L_c(t) = interp(t; xp=n_i, fp=p_i)`。**[0,1] で保持**、プロジェクタ入力直前で 8bit 量子化。

### 構築ソースの選択（config）

`source` 既定 `"marginal"`（上記 §5.1、全撮影を周辺集約）。代替として `"channel_ramp"`
（軸のみ）/`"gray"`（グレーのみ）を選べる。既定は網羅サンプルを最大活用する `"marginal"`。

## 6. LUT 適用

- `apply_inverse_lut(image, lut) -> np.ndarray`:
  入力画像（[0,1] float or uint を正規化）をチャンネルごとに `L_c` で線形補間 →
  プロジェクタ入力画像を返す。既存 `apply_inverse_gamma_correction` と差し替え可能な
  シグネチャ（画像とオプションを取り、同種を返す）にする。
- 出力は既定 float [0,1]。`dtype` 指定で 8bit/16bit 量子化に対応（投影直前想定）。

## 7. 設定 & 保存

`config.py` に `[inverse_lut]` セクションと `InverseLutConfig`（frozen dataclass）を追加。
既存 `_build_section` / `_serialize_config_toml` / `generate_config_file` の規約に準拠。

| キー | 既定 | 説明 |
| --- | --- | --- |
| `grid_divisions` | `3` | 粗い混合グリッドの分割数（`generate_projection_patterns` に渡す） |
| `gray_levels` | `17` | 細かいグレーランプの階調数 |
| `use_channel_ramps` | `false` | 単独チャンネルランプを追加するか |
| `source` | `"marginal"` | 曲線構築ソース（`marginal`/`channel_ramp`/`gray`） |
| `lut_resolution` | `256` | 逆 LUT の密グリッド解像度 |
| `reference_region` | `0.2` | 計測に使う中央領域の割合 |
| `trust_low` | `0.039` | 信頼下限（≈10/255 未満は除外） |
| `trust_high` | `0.961` | 信頼上限（≈245/255 超は除外） |
| `hdr_enabled` | `false` | 簡易 HDR（仮実装） |
| `inverse_lut_path` | `"data/inverse_lut.npz"` | LUT 保存先 |

- LUT は npz（`lut`, `t_grid`, ノット, メタ）で保存/読込。

## 8. テスト（`tests/test_inverse_lut.py`, ハードウェア非依存）

合成応答 `m(p) = black + (1 − black) · p^γ`（チャンネル別 γ も）でモックし検証:

- `apply_inverse_lut` 後に応答 `g` を通すと **リニアに復元**される（`L(t) ≈ t^(1/γ)`, 許容誤差内）。
- **黒引き**: `m(0) > 0`（例 10/255）でも `L(0)=0` になる。
- **単調性**: 生成 LUT が非減少。
- **サチュ除外**: `≤trust_low` / `≥trust_high` の点が構築から除外される。
- **統合**: グレー（細）＋グリッド（粗）の観測が周辺集約に反映される。
- **チャンネル独立性**: R/G/B で別々の γ を与えて各々復元できる。
- **save/load 往復**、**パターン枚数/重複除去**、**中央領域平均**。
- 撮影は「リニアフレームを返す callable」をモックして HW 非依存に。既存
  `tests/test_sample_mocked.py` の `monkeypatch`/`tmp_path` 流儀に合わせる。

## 9. 非目標（YAGNI）

- `sample.py` / 補償パイプラインへの組み込み（今回はしない）。
- ピクセルごと LUT（グローバルのみ）。
- 本格 HDR マージ（仮実装のみ）。
- カメラ↔プロジェクタ間の幾何ワープ（無関係）。

## 10. 撮影枚数の見積り

既定（`grid_divisions=3` の 27 枚 + `gray_levels=17`、重複除去）で **約 41 パターン**。
`grid_divisions` は 3 乗で効くため、コスト調整は主にこの値で行う。
