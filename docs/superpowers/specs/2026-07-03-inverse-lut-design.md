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
- パターン生成は既存 `generate_projection_patterns`（n³ グリッド）は **流用しない**。
  本 LUT は 1D ランプ掃引なので専用の軽量生成器を新設する（§3）。
- `sample.py` の色補償撮影は **逆ガンマ済みパターン**を投影し `capture_image()`（sRGB uint8）で
  撮影する。本 LUT は目的が異なるため、この撮影データは再利用しない（§4 参照）。

## 2. モジュール構成 / データフロー

新規 `src/python/inverse_lut.py`:

```
generate_calibration_patterns()   # グレーランプ **または** 各チャンネル単独ランプ（source で一方のみ）
        ↓  生パターン投影 ＆ リニア撮影（中央領域平均, 簡易HDR=仮実装）
run_calibration_capture()  →  CalibrationData（(入力RGB, 計測値, 記録チャンネル) の集合）
        ↓  入力レベルで集約 → 黒引き・白正規化 → 単調化 → 反転
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

- `CalibrationSample`: `input_rgb: tuple[float,float,float]`（[0,1]）, `image: np.ndarray`,
  `kind: str`（"gray"|"channel"）, `recorded_channels: tuple[int,...]`。
- `CalibrationMeasurement`: `input_rgb: (3,)`, `measured_rgb: (3,)`（リニア [0,1]、未記録は NaN）,
  `recorded_channels: tuple[int,...]`, `kind: str`。
- `CalibrationData`: `measurements: list[CalibrationMeasurement]` + メタ（roi, 露光条件など）。
- `InverseLut`: `lut: np.ndarray (3, R)`（[0,1]）, `t_grid: np.ndarray (R,)`,
  `knots: list[(p_i, n_i) per ch]`, `meta: dict`。

## 3. キャリブレーションのサンプリング（生成画像）

以下の **2 系統**の 1D ランプのうち、**`source` で指定した一方のみ**を生成する
（`num_levels` 既定 **17** 点、[0,1] を等間隔、投影直前に 8bit 量子化）。両系統は
**混在させない**（混ぜると系統間で黒レベル/クロストークが食い違い、逆曲線フィットが不整合になる）。

1. **グレーランプ**（`source="gray"`）: `(v,v,v)`, 例 `(0,0,0),(15,15,15),(31,31,31),…,(255,255,255)`。
   1 回のグレー掃引で 3 チャンネル同時に駆動する。
2. **各チャンネル単独ランプ**（`source="channel_ramp"`, 既定）: `(v,0,0)` / `(0,v,0)` / `(0,0,v)`、
   例 `(0,0,0),(15,0,0),…,(255,0,0)`（G, B も同様）。各チャンネルを単独で掃引する。

- 全黒 `(0,0,0)` は複数ランプに現れるため、**同一投影画像は 1 回だけ撮影**する（計測は系統ごとに
  導出）。distinct な投影数は `source="gray"` で `num_levels`（**17**）、
  `source="channel_ramp"` で `1 + (num_levels-1) * 3` = **約 49**（`num_levels=17` のとき）。

## 4. 撮影・計測 / 保管方法（正しさの要）

- **生パターン（逆ガンマ非適用）を投影**する。逆ガンマを先に掛けると測っているのは「既に
  線形化された出力」になり、`g` を打ち消す情報が消える（＝逆ガンマの代替を作れない）。
- 計測は `camera.capture_linear_rgb()` の **リニア値**（[0,1] float32）を使用する。
  `sample.py` の `capture_image()`（`linear_to_srgb` を掛けた sRGB uint8）は使わない。
- 各撮影は **ROI 矩形 `(x, y, width, height)`（`roi_x`/`roi_y`/`roi_width`/`roi_height`, ピクセル）の平均**で
  1 組の計測 RGB に集約（グローバル LUT のため）。`width`/`height` が 0（既定）なら全フレームを平均。
  ROI はフレーム境界にクリップする。
- 投影ウィンドウは `sample.py` 流儀（`namedWindow`/`WINDOW_FULLSCREEN`/`moveWindow(pos_x,pos_y)`/
  `waitKey(wait_key_ms)`）を踏襲。

### 保管方法（記録するチャンネル）

- **グレーランプ**: カメラの **r, g, b の全チャンネル**を記録（`recorded_channels=(0,1,2)`）。
- **各チャンネル単独ランプ**: **投影しているチャンネルのみ**を記録
  （R ランプ → camera R のみ、`recorded_channels=(0,)`。G, B も同様）。未記録チャンネルは NaN。

### 信頼レンジ（8bit画素値で Config 可）

- `≤ trust_low`（既定 **10**）/ `≥ trust_high`（既定 **245**）を **8bit画素値(0-255)** で指定し、
  黒潰れ/サチュ/非信頼として扱う。構築時はこの範囲外の点を除外し、HDR撮影では露光段の選択に用いる
  （`[inverse_lut]` の `trust_low`/`trust_high`。内部でリニア[0,1]比較用に /255 変換）。

### 簡易 HDR（実装済み）

- `capture_hdr_linear` は露光ブラケット＋マージを実装: 基準露光でサチる (≥ trust_high) 画素は露光を
  短く（scale 0.5, 0.25…）、黒潰れ (≤ trust_low) 画素は露光を長く（scale 2, 4…）して撮影し、
  **実効露光比**（公称 scale ではなく実際に取れた `tv` の秒比）で割って基準露光リニア単位に戻す。
  信頼できる画素で非信頼画素を置換し、ダイナミックレンジを拡張する（値は 1.0 超も可、白正規化で吸収）。
  段数は `hdr_max_stops`（既定 3）。
- 露光制御は Canon の `tv` を段ごとにスケール（`scale_shutter_tv`）して実現。**スケール後の目標秒は、
  実機の受理 Tv メニュー（`camera.supported_shutter_speeds()` → `list_supported()["Tv"]`）— 取得不可時は
  標準 Canon Tv — の中で最も近い値へスナップする**。これにより EDSDK が受理する値を保証し、
  キャリブレーション途中の `ValueError` 中断を防ぐ。離散化で公称と実効露光がずれるが、マージは実効比で
  割るため回復値は偏らない。`hdr_enabled=true` かつ Canon バックエンド時のみ有効化し、非対応
  バックエンドは警告して単一露光にフォールバック。
- スナップ／秒変換／実効比マージは単体テスト済み。実機での受理値・露光ずれは、投影もシャッターも
  行わない `examples/python/verify_hdr_tv.py`（`--dry-run` はカメラ不要）で確認できる。

## 5. LUT 構築（核心）

チャンネル `c` ごとに、**選択した系統**の計測から `(入力[c], 計測[c])` を集める。
系統は撮影・構築を通じて一方のみを用いる（`normalize_source` で検証。廃止した `"both"` は
`ValueError`）。

- `source="channel_ramp"`（既定）: 単独ランプ `c` のみ。各チャンネルを単独で駆動して測るため
  プロジェクタチャンネル間クロストークが混入しない。
- `source="gray"`: グレーランプのみ（1 掃引で 3ch 同時記録。撮影枚数は最小だが R=G=B 同時点灯）。

構築手順:

1. **集約**: 収集した `(入力[c], 計測[c])` を **入力レベルでグルーピングして平均** →
   ノット `(p_i, m_i)`。全黒など重複入力は平均で 1 点に。
2. **黒引き・白正規化**: `m0 = m(0)`（黒/環境光）, `m_white = m(最大入力)`。
   `n_i = (m_i − m0) / (m_white − m0)` → `n(0)=0, n(最大)=1`。
   これで「0,0,0 でカメラ 10 なら LUT 値は 0」要件を黒引きで満たす。
   `m_white − m0 ≤ 0`（縮退）の場合は警告し恒等 LUT にフォールバック。
3. **単調化**: 非減少に補正（`scipy` 利用: PCHIP または等調回帰）。信頼レンジ外の点は除外済み。
   有効ノットが 2 点未満なら恒等にフォールバックし警告。
4. **反転**: 一様 `t` グリッド（`lut_resolution` 既定 256）に対し
   `L_c(t) = interp(t; xp=n_i, fp=p_i)`。**[0,1] で保持**、プロジェクタ入力直前で 8bit 量子化。

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
| `num_levels` | `17` | 各ランプ（グレー / 単独）の階調数 |
| `source` | `"channel_ramp"` | サンプリング系統（`channel_ramp`/`gray` の一方のみ・混在不可） |
| `lut_resolution` | `256` | 逆 LUT の密グリッド解像度 |
| `roi_x` | `0` | 計測 ROI 左端 x（ピクセル） |
| `roi_y` | `0` | 計測 ROI 上端 y（ピクセル） |
| `roi_width` | `0` | 計測 ROI 幅（0 なら全フレーム平均） |
| `roi_height` | `0` | 計測 ROI 高さ（0 なら全フレーム平均） |
| `trust_low` | `10` | 信頼下限（8bit画素値, ≤は非信頼/黒潰れ） |
| `trust_high` | `245` | 信頼上限（8bit画素値, ≥は非信頼/サチュ） |
| `hdr_enabled` | `false` | 簡易HDRブラケット撮影 |
| `hdr_max_stops` | `3` | 各方向の最大露光段数 |
| `inverse_lut_path` | `"data/inverse_lut.npz"` | LUT 保存先 |

- LUT は npz（`lut`, `t_grid`, ノット, メタ）で保存/読込。

## 8. テスト（`tests/test_inverse_lut.py`, ハードウェア非依存）

合成応答 `m(p) = black + (1 − black) · p^γ`（チャンネル別 γ も）でモックし検証:

- `apply_inverse_lut` 後に応答 `g` を通すと **リニアに復元**される（`L(t) ≈ t^(1/γ)`, 許容誤差内）。
- **黒引き**: `m(0) > 0`（例 10/255）でも `L(0)=0` になる。
- **単調性**: 生成 LUT が非減少。
- **サチュ除外**: `≤trust_low` / `≥trust_high` の点が構築から除外される。
- **保管方法**: グレーは r,g,b を記録、単独ランプは投影チャンネルのみ記録（他は NaN）。
- **系統排他**: `source` は `channel_ramp`/`gray` の一方のみ。混在データを与えても選択系統の
  観測だけが集約に使われ、廃止した `"both"` は `ValueError`。
- **チャンネル独立性**: R/G/B で別々の γ を与えて各々復元できる。
- **save/load 往復**、**パターン枚数/全黒の重複除去**、**中央領域平均**。
- 撮影は「リニアフレームを返す callable」をモックして HW 非依存に。既存
  `tests/test_sample_mocked.py` の `monkeypatch`/`tmp_path` 流儀に合わせる。

## 9. 非目標（YAGNI）

- `sample.py` / 補償パイプラインへの組み込み（今回はしない）。
- ピクセルごと LUT（グローバルのみ）。
- 高度な HDR（位置ずれ/ゴースト補正、トーンマッピング等）。簡易な露光ブラケット＋マージのみ。
- 混合色（n³ グリッド）サンプリング（グレー＋単独ランプのみ）。
- カメラ↔プロジェクタ間の幾何ワープ（無関係）。

## 10. 撮影枚数の見積り

`source` で選んだ一方の系統のみを撮影する（`num_levels=17` のとき）:

- `source="channel_ramp"`（既定）: **約 49 パターン**（単独 R/G/B 各 17、全黒重複を除去）。
- `source="gray"`: **17 パターン**（グレー 17）。

`num_levels` を変えると線形に増減する。両系統を同時に撮影する `"both"` は廃止した。
