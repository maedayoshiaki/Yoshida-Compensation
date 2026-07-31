# 色変換行列インタラクティブビューア 設計仕様

作成日: 2026-07-31
対象ブランチ: `feature/matrix-viewer-gui`
基準タグ: `v1.0-hdr-lut` (`1322538`)

## 1. 目的

画素ごとの色変換行列 `(H, W, 4, 3)` を対話的に可視化し、行列が何を学習したのかを
人が読めるようにする。あわせて、任意のターゲット画像から投影画像を生成し、
線形化方式（解析ガンマ / 実測逆LUT）の効果をGUI上で比較できるようにする。

現在の評価は PSNR / SSIM / ΔE76 の集約値だけで、「どこがなぜ悪いのか」が分からない。
設計検討中に実データ `data/linear_pipeline_1080/matrices_linear_gamma.npy` を可視化した
ところ、次が判明した。

- 白投影時のカメラ応答の中央値は `RGB = (0.526, 0.245, 0.436)`。緑が赤の半分以下しか
  返らないマゼンタ寄りの投影面である。`data/linear_pipeline_1080/README.md` の
  「白投影で中央=マゼンタ、Gの反射が弱い」という当日の観察と一致する。
- 環境光は `(-0.015, 0.009, 0.005)` でほぼ暗室。
- 線形 0.5 のグレーを出すのに必要な投影量の中央値は `Q = (0.726, 1.969, 1.082)`。
  緑は約2倍必要で、プロジェクタの表現範囲を超えている。
- 実ターゲット `nature_1_sq.png` では有効領域の 43.3% が上側クリップ、9.0% が下側クリップ。

集約指標からはこれらが読み取れなかった。本ツールはこの種の情報を日常的に見えるようにする。

## 2. スコープ

### 対象

- 既存の色変換行列 `.npy`、逆LUT `.npz`、P2Cマップ `.npy` の読み込みと可視化
- 画像入力を受け付けた投影画像の生成（CLIと同一の計算経路を通す）
- 解析逆ガンマ / 実測逆LUT の適用と、LUTカーブの表示
- 将来の色補償手法を追加できるビューレジストリ

### 対象外

- プロジェクタ投影とカメラ撮影の制御。撮影・校正は従来どおりCLIで行う。
- 色行列の再fit、GrayCode再校正。
- `warp_method = "c2p"` 経路。現行 config は `c2p_map = ""` で使用していない。

## 3. アーキテクチャ

`hyperspectral-color` の `packages/web`（ADR-0002 で PyQt6 から移行済み）と同じ構成を採る。
バックエンドが検証済みの float 計算結果をバイナリで返し、フロントエンドは WebGL2 で
表示トーンのみを担う。

```
ブラウザ (WebGL2, ビルド工程なしの ES module)
   │  float32 バイナリ（JSONにしない）
   ▼
FastAPI (yoshidacomp_web)
   │  yoshidacomp のみに依存
   ▼
yoshidacomp（純計算・ハードウェア非依存）
```

### 3.1 パッケージ構成

uv workspace の2分割とする。`hyperspectral-color` が core/viz/gui の3分割なのは
matplotlib 描画層を別消費者と共有するためだが、本ツールは描画がブラウザ側にあるため
viz に相当する層が存在しない。

```
Yoshida-Compensation/
├── pyproject.toml                       # uv workspace root
├── packages/
│   ├── core/pyproject.toml              # 配布名 yoshidacomp
│   │   └── yoshidacomp/
│   │       ├── config.py
│   │       ├── compensation/
│   │       │   ├── color_mixing_matrix.py
│   │       │   ├── photometric_compensation.py
│   │       │   ├── projection_region.py
│   │       │   ├── inverse_lut.py
│   │       │   └── pipeline.py          # 新規
│   │       ├── geometry/warp.py         # 新規（sample.py から抽出）
│   │       ├── calibration/
│   │       └── capture/                 # optional extra: camera
│   └── web/pyproject.toml               # 配布名 yoshidacomp-web
│       └── yoshidacomp_web/
│           ├── app.py store.py deps.py models.py
│           ├── routers/{session,matrix,view,target,lut,generate,meta}.py
│           ├── views/                   # ビューレジストリ
│           └── static/{index.html,style.css,js/}
├── examples/python/                     # 既存CLIは yoshidacomp を呼ぶ薄い層に
└── tests/
```

依存方向は `web → core` の一方向。`yoshidacomp` は `fastapi` / `edsdk` / `rawpy` を
module scope で import しない。ハードウェア依存は optional extra に置く。

### 3.2 既存コードの移行

現在は `from src.python.config import get_config` の形でリポジトリルートを `sys.path` に
入れて解決している。正式なパッケージへ移すため、以下の書き換えが発生する。

| 現在 | 移行後 |
| --- | --- |
| `src/python/config.py` | `yoshidacomp/config.py` |
| `src/python/color_mixing_matrix.py` | `yoshidacomp/compensation/color_mixing_matrix.py` |
| `src/python/photometric_compensation.py` | `yoshidacomp/compensation/photometric_compensation.py` |
| `src/python/projection_region.py` | `yoshidacomp/compensation/projection_region.py` |
| `src/python/inverse_lut.py` | `yoshidacomp/compensation/inverse_lut.py` |
| `src/python/camera/` | `yoshidacomp/capture/` |
| `src/python/{cam_in_calib,cap_chess,solvePnP}.py` | `yoshidacomp/calibration/` |

`examples/python/sample.py`（1,363行）はハードウェア撮影と純計算が混在している。
純計算部分を `yoshidacomp/compensation/pipeline.py` と `yoshidacomp/geometry/warp.py` へ
移し、`sample.py` は撮影と CLI 引数処理だけを持つ薄い層にする。抽出対象は次のとおり。

- `srgb_to_linear`, `linear_to_srgb`, `capture_to_srgb_uint8`
- `_load_target_images`
- `resolve_warp_settings`, `resolve_target_image_space`, `compensation_space_from_warp_method`
- `center_rect`
- `warp_camera_array_to_projector`, `warp_color_mixing_matrices_to_projector`, `warp_image`, `invwarp_image`
- `prepare_target_image_for_compensation`

これにより `examples/python/rebuild_compensation_with_map.py` は `sample.py` を
まったく import せずに済むようになる。従来は `from examples.python import sample` で
CLIスクリプトをライブラリとして参照していた。

なお `examples/python/sample_inverse_lut.py` は `importlib.util.spec_from_file_location`
で `sample.py` を直接ロードしており、未移動の私的デバッグ補助を含む約30個の
`sample.*` 属性に依存する。その解消は本計画の対象外とし、別タスクとする。

GrayCode は `external/GrayCode` の submodule のままとし、`geometry/warp.py` の
import 1箇所で吸収する。submodule は `340b7fe`（`fix/c2p-inpaint-propagation`、push済み）
を指しており、`import graycode` が torch を引かない状態（`a687b8f`, PEP 562）になっている。

### 3.3 既存テストへの影響

現在70テストが通っている。`tests/test_sample_mocked.py` は `sys.modules` に `colour` と
`psutil` のスタブを差し込んでから `importlib` で `sample.py` を直接ロードするという
回避策を取っている。正式パッケージ化により通常の import とフィクスチャで書けるように
なるため、このモック手法は書き直しになる。移行中は全テストが緑であることを維持する。

## 4. バックエンド API

転送規約は `hyperspectral-color` の `packages/web` に合わせる。画素データは
`application/octet-stream` の float32（リトルエンディアン）で送り、形状は
`X-Height` / `X-Width` / `X-Channels` などのヘッダで伝える。JSON には載せない。
`/api/*` を先に登録し、`StaticFiles(html=True)` を最後に `/` へマウントする。

| メソッド | パス | 内容 |
| --- | --- | --- |
| POST | `/api/session/matrix` | `.npy` を読み込み、行列IDを返す |
| GET | `/api/matrix/{id}/meta` | 形状・座標空間・有効領域・利用可能な座標空間 |
| POST | `/api/matrix/{id}/warp` | カメラ座標→プロジェクタ座標のP2Cワープを開始（ジョブID） |
| GET | `/api/job/{job_id}` | 進捗と完了状態 |
| GET | `/api/views` | ビュー一覧とパラメータ定義 |
| GET | `/api/matrix/{id}/view/{key}` | ビューの float32 バイナリ |
| GET | `/api/matrix/{id}/probe` | 指定画素の 4×3 と派生量（JSON） |
| POST | `/api/target` | ターゲット画像のアップロードまたはパス指定 |
| POST | `/api/lut` | 逆LUT `.npz` の読み込み |
| GET | `/api/lut/{id}/curve` | LUTカーブ `(3, 256)` と `t_grid` |
| POST | `/api/generate` | 投影画像の生成と保存 |
| GET | `/api/meta/config` | 現在の config（プロジェクタ解像度・有効領域・ガンマ等） |

`store.py` は FastAPI 非依存のスレッドセーフな辞書として、読み込み済みの行列・ターゲット・
LUT・ワープ結果を保持する。

## 5. ビューレジストリ

拡張性の中心。ビューは Python 側に1つ登録すればGUIへ現れ、標準的な種類であれば
JavaScript の変更を要さない。

```python
@dataclass(frozen=True)
class ViewSpec:
    key: str
    label: str
    kind: Literal["scalar", "rgb", "categorical"]
    params: list[ParamSpec]
    render: Callable[[MatrixContext, dict], ViewResult]
```

- `kind` がフロントの描画方法を決める。`scalar` は発散カラーマップ、`rgb` は
  線形RGBとして露出・γ・sRGB OETF を適用、`categorical` は凡例付きの離散色。
- `params` から GUI のコントロール（数値入力・選択・画像選択）を自動生成する。
- `ViewResult` は float32 の `(H, W, C)` と、値域・凡例・単位のメタデータを持つ。
- `MatrixContext` は行列本体・座標空間・有効領域マスク・関連するconfigを保持する。

### 5.1 初期登録ビュー

| key | kind | 内容 | パラメータ |
| --- | --- | --- | --- |
| `coef_tiles` | scalar × 12 | 4行(in_R/in_G/in_B/bias) × 3列(out_R/out_G/out_B) の係数マップ | 正規化方法 |
| `surface` | rgb | 投影面の色。順方向モデル `C = (P − b)·A⁻¹` | 投影コマンド `P`（既定 `(1,1,1)`） |
| `compensation` | rgb | ターゲットに行列を適用した結果と投影コード値 | ターゲット画像、線形化方式 |
| `gamut` | categorical | 色域外（負クリップ / 1超クリップ / 両方 / 範囲内） | ターゲット画像 |

### 5.2 順方向モデルの導出

行列 `M` は `[C_r, C_g, C_b, 1] @ M = P` を満たす。`M = [A; b]`（`A` は 3×3、`b` は 1×3）と
書けば `C·A + b = P` なので、任意の投影コマンド `P` に対するカメラ応答は

```
C = (P − b) · A⁻¹
```

となる。`P = (1,1,1)` が投影面の色、`P = (0,0,0)` が環境光のみの応答である。
`det(A)` が極小の画素は逆行列を持たないため、有効領域マスクから除外する。

## 6. 数値の読み取り

参照元GUIと同じく、ホバーとクリックで役割を分ける。

- **ホバー** — サーバ往復なし。取得済みバイナリのCPU側コピーから表示中の値を即座に読む。
  マウス移動レートで動作するため、重い処理を挟まない。
- **クリック** — `GET /api/matrix/{id}/probe?x=&y=` を呼び、その画素の 4×3 の全係数と
  派生量（面の色、必要投影量、クリップ判定）を詳細パネルへ固定する。マーカーを描画し、
  複数箇所を比較できる。

座標変換は `glview.js` の `screenToImage(cssX, cssY) -> {col, row}` と
`imageToScreen(col, row) -> {x, y}` の対で行う。バッキングストアは
`clientWidth * devicePixelRatio` の実デバイス画素で確保し、`ResizeObserver` で追従する。

## 7. 表示と再計算の分離

`hyperspectral-color` の `main.js` が明記している原則をそのまま採る。

| サーバ往復なし（GLのuniform更新のみ） | サーバ往復（再フェッチ） |
| --- | --- |
| 露出EV、γ、カラーマップ、スケール、pan/zoom、ディザ | ビュー切替、パラメータ変更、行列切替、座標空間切替、ターゲット変更、線形化方式変更 |

トーンマッピング・sRGB OETF・ディザ・NaNマスクはすべてフラグメントシェーダで行い、
8bitへの量子化はディザ後にGPUで一度だけ実施する。これによりバンディングを避ける。
連続するフェッチには最新優先のリクエストID（`state.reqId`）で順序保証を与える。

## 8. 線形化とLUTカーブ

線形化方式もレジストリとして扱い、`none` / `gamma` / `inverse_lut` を登録する。
将来の方式追加はここへの登録で済む。

- `gamma` は `yoshidacomp.compensation.color_mixing_matrix.apply_inverse_gamma_correction`
  を使う。**この関数は渡された指数をそのまま `pow` に使うため、明るくするには
  `gamma=1/projector.gamma`（≈0.4545）を渡す必要がある。** 引数を省略すると 2.2 が
  適用され逆方向に暗くなる。GUIからは必ず明示的に渡す。
- `inverse_lut` は `yoshidacomp.compensation.inverse_lut.apply_inverse_lut` を使う。
  純NumPy実装で `np.interp` をチャンネルごとに適用する。
- LUTは `.npz` に `lut` `(3, 256) float32`、`t_grid` `(256,) float32`、`meta`（JSON文字列）
  として保存されている。`lut[c, i]` は出力 `t_grid[i]` を得るためのプロジェクタ入力値。

`GET /api/lut/{id}/curve` はこの配列を返し、フロントは DPR 対応の 2D canvas
（`plot.js`）で描画する。解析ガンマも同じ256点形式にサンプルできるため、
**実測LUTと解析ガンマを同一グラフに重ねて比較できる**。実測LUTの終端は必ずしも 1.0 に
ならない（ISO400実測でチャンネルランプが約 0.876/0.876/0.688）ため、縦軸は 1.0 で
固定せず実データの範囲を表示する。

## 9. 行列の読み込みと座標空間

形状から座標空間を自動判定する。

| 形状 | 座標空間 | 例 |
| --- | --- | --- |
| `(4660, 6984, 4, 3)` | カメラ | `data/color_mixing_matrices.npy`（1.56 GB） |
| `(1080, 1920, 4, 3)` | プロジェクタ | `data/linear_pipeline_1080/matrices_linear_*.npy`（約100 MB） |
| `(720, 1280, 4, 3)` | プロジェクタ（旧解像度） | `data/comp_test_20260724/matrices_*.npy` |

判定は「プロジェクタ解像度と一致するか」を基準とし、一致しなければカメラ座標として扱う。

カメラ座標の行列は **そのまま表示** と **P2Cワープ後にプロジェクタ座標で表示** の
両方を選べる。ワープは `(12, 4660, 6984)` のテンソルを扱うため重く、UIを止めないよう
バックグラウンドジョブとして実行し進捗を返す。結果は `store.py` にキャッシュし、
同じ行列とマップの組み合わせで再計算しない。読み込みは `np.load(mmap_mode="r")` を使う。

P2Cマップは `(N, 4) float32` で各行が `[proj_x, proj_y, cam_x, cam_y]`。現行の
`data/graycode_recalibration_20260730/p2c_dense_active_1920x1080.npy` は 2,073,600行
（1920×1080の全プロジェクタ画素）で、うち 74.9% は有効投影領域外のため
`cam_x, cam_y` が NaN である。ワーパはこの行を除去する。

## 10. 投影画像の生成

`POST /api/generate` は `yoshidacomp/compensation/pipeline.py` を呼ぶ。CLI の
`rebuild_compensation_with_map.py` と同じ関数を通るため、GUIとCLIで結果が一致する。

処理順は現行パイプラインを踏襲する。

1. ターゲット画像を読み、sRGB / 指定ガンマ / 素通しのいずれかで線形化
2. `embed_in_active_region` で有効投影領域へ配置（アスペクト比は保持されず引き伸ばされる）
3. ワープ済み行列で `calc_compensation_image`（clip後に8bit量子化）
4. `mask_to_active_region` で有効領域外を黒に
5. 線形化方式を適用して投影用コード値を得る
6. 線形補償画像と投影画像を保存し、メタデータJSONを書く

## 11. テスト方針

- **ゴールデンテスト** — バックエンドの float 出力に、シェーダ相当の処理（露出・γ・
  sRGB OETF）を numpy で適用した結果が、既存CLIの出力と画素単位で一致することを確認する。
  FastAPI を必要としないため通常のテストゲートに乗る。
- **境界テスト** — `yoshidacomp` が `yoshidacomp_web` を import しないことの静的スキャンと、
  `fastapi` / `edsdk` / `rawpy` を `sys.meta_path` で遮断した subprocess で
  `yoshidacomp` が import できることの確認。`hyperspectral-color` の
  `tests/test_package_boundaries.py` の手法を流用する。
- **ビューテスト** — 合成した小さな行列で全登録ビューが例外なく描画でき、
  `ViewResult` の形状とメタデータが `ViewSpec` の宣言と整合することを確認する。
- **移行の回帰テスト** — 既存70テストを緑のまま維持する。
- **手動スモーク** — `uv run python -m yoshidacomp_web` を起動し、実際の1.56GB行列の
  読み込み・ワープ・4ビューの表示・ホバー/クリックの数値読み取り・投影画像生成を確認する。
  Windowsの表示スケーリング150%で文字と画像が鮮明であることも見る。

## 12. 実装順序

1. パッケージ分割と既存コードの移行（70テスト緑を維持）
2. `pipeline.py` / `geometry/warp.py` の抽出と、CLIの薄層化
3. FastAPI の骨組み、`store.py`、行列読み込みとメタ取得
4. `glview.js` を中心としたフロントの縦スライス（`coef_tiles` のみ表示できる状態）
5. ビューレジストリと残り3ビュー
6. ホバー/クリックの数値読み取り
7. ターゲット入力、線形化、LUTカーブ
8. 投影画像生成
9. カメラ座標行列のワープ（バックグラウンドジョブ）

3〜4 の段階で一度実機の1.56GB行列を通し、性能上の問題がないことを確認してから
先へ進む。

## 13. 未決事項

- 生成した投影画像の保存先の既定値。当面は出力ディレクトリをGUIで指定する。
- 複数行列を同時に開いて比較する機能。参照元GUIはタブによる複数ドキュメントを
  持つが、初版では単一行列とし、必要になってから追加する。
