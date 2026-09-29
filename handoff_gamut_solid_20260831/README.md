# 6³格子の投影・撮影と「Input lattice → measured output solid」図の作成一式

プロジェクタ・カメラ系で 6³=216 色の校正格子を投影・撮影し、実測出力色空間
（白正規化 c）にできる「変形立方体」を 3D 図（PNG / 回転GIF）として描くための
プログラム一式。Yoshida-Compensation リポジトリ（2026-08-31 時点の作業ツリー）
から**コード改変なし**で抽出した。ディレクトリ配置も元リポジトリと同一なので、
スクリプト内の相対パス解決（`external/GrayCode/src` の探索など）はそのまま動く。

## 構成

| パス | 役割 |
| --- | --- |
| `examples/python/capture_tps_calibration.py` | **投影・撮影本体**。6³格子+ドリフト番兵を全面単色で投影し、HDR撮影→プロジェクタ座標へワープ→ダウンサンプルして色ごとの `.npz` と `manifest.jsonl` に保存（中断・再開可能） |
| `examples/python/plot_gamut_solid_3d.py` | **図の作成本体**。撮影結果から `gamut_solid_view_{iso,rg,top}.png`、`--gif` で `gamut_solid_rotation.gif` を生成 |
| `examples/python/calibrate_graycode_geometry.py` | 幾何校正（グレイコード）。撮影に必要な p2c 対応マップを作る |
| `examples/python/measure_color_grid_response.py` | 撮影ジョブ計画のヘルパ（capture が import する）。単体では 321 色の応答調査撮影スクリプト |
| `examples/python/sample.py` | ワープ関数の再輸出と撮影パイプライン（capture / 幾何校正が import する） |
| `examples/python/generate_tps_and_matrix_compensation.py` | `_load_calibration`（撮影結果の読込）を plot が import する |
| `examples/python/analyze_lut3d_validation.py` ほか2本 | `--validation` オーバレイ用のオプション依存 |
| `packages/core/yoshidacomp/` | コアパッケージ（カメラバックエンド、HDR合成、config、ワープ、正規化） |
| `external/GrayCode/src/graycode/` | グレイコード幾何校正・ワープ実装（github.com/maedayoshiaki/GrayCode の src を同梱） |
| `config.color_grid_iso200_hdr_d65_20260812.toml` | 実際に使った config の例（値は元機材固有。下記「機材ごとに変える値」参照） |

## セットアップ

- **Python 3.13 必須**（EDSDK バインディングの wheel が cp313 / win_amd64）。
- uv の場合、バンドルのルートで:

  ```
  uv sync                  # 解析・作図のみ
  uv sync --extra capture  # 実機撮影もする場合（EDSDK バインディング + rawpy）
  ```

  pip の場合: `pip install -e packages/core matplotlib`（撮影するなら
  `pip install -e "packages/core[camera]"`）。
- **Canon EDSDK の DLL は同梱していない**（Canon のライセンス上、各自で
  Canon Developer Programme から入手する）。`EDSDK.dll` / `EdsImage.dll` を
  任意のディレクトリに置き、config の `[camera] edsdk_dll_dir` にそのパスを
  書く。バインディング本体は
  github.com/youme-town/edsdk-python の release wheel（`capture` extra が自動で引く）。
- torch は CPU 版で動く（ワープに使うだけ）。GPU を使いたい場合は
  `pyproject.toml` 末尾のコメント参照。

## ハードウェア前提

- プロジェクタを OS の拡張ディスプレイとして接続。config の
  `[projector] pos_x / pos_y` に全画面ウィンドウを移動するので、その仮想
  デスクトップ座標を実機に合わせる。
- カメラは Canon EDSDK 対応機（RAW 現像で線形 RGB を得る）。`backend = "opencv"`
  も選べるが、線形性が保証されないので本計測には非推奨。
- 暗室推奨。パターン切替後の settle は既定 2 秒（プロジェクタの
  コンテンツ適応輝度の過渡を待つため。短くすると ±4% 程度暴れた実績あり）。

## 実行手順（3ステップ）

```
# 0. 幾何校正: p2c マップを作り、config の [paths] p2c_map に書く
uv run python examples/python/calibrate_graycode_geometry.py --config config.<自分の>.toml

# 1. 投影・撮影（216色+番兵、HDR。中断しても同じコマンドで続きから再開）
uv run python examples/python/capture_tps_calibration.py \
    --config config.<自分の>.toml --output-dir data/tps_calibration_<日付>

# 2. 図の作成
uv run python examples/python/plot_gamut_solid_3d.py \
    --calibration-dir data/tps_calibration_<日付> \
    --output-dir data/tps_calibration_<日付>/analysis --gif
```

所要時間の目安: 元環境（HDR、settle 2s）で1色あたり十数秒 × 225 ジョブ。

## config で機材ごとに変える値

| キー | 意味 |
| --- | --- |
| `[projector] width/height/pos_x/pos_y/refresh_rate` | プロジェクタ解像度と全画面ウィンドウの移動先座標、リフレッシュレート（フリッカ回避のシャッター選定に使う） |
| `[projector] active_x/y/width/height` | 投影する有効領域（スクリーン内に収まる矩形） |
| `[camera] av/tv/iso` | 露出。白 (255,255,255) が飽和しない（≲0.96）ように選ぶ |
| `[camera] edsdk_dll_dir` | EDSDK DLL の置き場所 |
| `[paths] p2c_map` | ステップ0で作った対応マップ |
| `[inverse_lut] roi_x/y/width/height` | カメラ画像内の計測 ROI（投影面の一様な部分） |
| `[inverse_lut] trust_low/trust_high/hdr_*` | HDR 合成の信頼範囲（8bit換算）と最大段数 |

`[paths]` の calibration 系ディレクトリと `inverse_lut_path` はこの計測では
使わない（config を補償パイプラインと共用しているだけ）。

## 図の読み方（正規化の規約）

- 出力空間 c はカメラ線形 RGB をチャンネル別に
  `c = (実測 − 黒(0,0,0)の実測) / (白(255,255,255)の実測 − 黒の実測)`
  で正規化したもの。**定義上 (1,1,1) になるのは白の角だけ**で、他の格子点が
  1 に一致しないのは正規化の不備ではなく、白（3ch同時駆動の実測）が分母で
  あることの帰結。クロストークと非加法性のぶんだけ理想からずれる。
- 参考実測（2026-08-12、旧プロジェクタ、D65 WB）: R単独フル駆動
  c=(0.695, 0.00, 0.01)、G単独 c=(0.24, **1.37**, 0.00)、B単独
  c=(0.00, 0.00, 0.96)。G単独は白の camera-G を 37% 上回る（c>1）ため
  立体は単位立方体の外へ膨らみ、軸レンジを 1.4 まで取ってある。
- 破線の単位立方体は「線形・加法的・クロストーク無しの理想デバイス」で、
  実測立体との差分そのもの（グレー対角の切れ目、稜線の曲がり、全体の
  湾曲・膨らみ）が診断情報。
- 実装は `packages/core/yoshidacomp/compensation/color_lut3d.py` の
  `normalize_reference_response`。

## 出典・注意

- 抽出元: Yoshida-Compensation リポジトリ（ブランチ feature/path-file-picker、
  2026-08-31 の作業ツリー）。計測設計と結果の詳細は元リポジトリの
  `docs/color_grid_response_20260812.md` / `docs/projector_io_visualization_survey_20260815.md`。
- 元リポジトリのユニットテストは同梱していない。
- `external/GrayCode` は別リポジトリ（git submodule）の同梱コピーなので、
  改修する場合は本家 github.com/maedayoshiaki/GrayCode 側へ反映すること。
