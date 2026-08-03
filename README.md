# Yoshida-Compensation

プロジェクタ-カメラシステムにおける色補償（Photometric Compensation）パイプラインの実装です。
プロジェクタから投影したパターンをカメラでキャプチャし、ピクセルごとのカラーミキシング行列を算出することで、目標画像に近い投影結果を得るための補償画像を生成します。

## 必要な環境

- **Python** 3.13 以上
- **uv**（パッケージマネージャ）
- **CUDA 対応 GPU**（推奨・なくても CPU で動作可能）
- **カメラデバイス**（`canon_edsdk` または `opencv` バックエンド）

カメラバックエンドごとの要件:

- `canon_edsdk`: Canon カメラ + EDSDK 環境、`rawpy`（詳細は「Canon EDSDK バックエンドのセットアップ」）
- `opencv`: OpenCV でアクセス可能な UVC/USB カメラ

## セットアップ

### 1. リポジトリのクローン

このリポジトリはサブモジュール（[GrayCode](https://github.com/youme-town/GrayCode)）を含みます。`--recursive` オプションを付けてクローンしてください。

```bash
git clone --recursive https://github.com/youme-town/Yoshida-Compensation.git
cd Yoshida-Compensation
```

既にクローン済みでサブモジュールが取得されていない場合は、以下を実行してください。

```bash
git submodule update --init --recursive
```

### 2. 依存パッケージのインストール

[uv](https://docs.astral.sh/uv/) を使って仮想環境の作成と依存パッケージのインストールを行います。

```bash
uv sync
```

サンプルスクリプト（`examples/`）を実行する場合は、追加の依存パッケージもインストールしてください。

```bash
uv sync --extra sample
```

`canon_edsdk` を使う場合は、この `--extra sample` に含まれる `rawpy` が必要です。
加えて、以下の「Canon EDSDK バックエンドのセットアップ」を参照してください。

### Canon EDSDK バックエンドのセットアップ

`camera.backend = "canon_edsdk"` で実機撮影を行うには Canon EDSDK 本体が必要です。
Python ラッパー `edsdk-python` v0.2.0 と `pywin32` は通常の `uv sync` で
GitHub Releases のビルド済み wheel から自動導入されるため、パッケージの手動コピーや
ローカルビルドは不要です。

Canon から取得した `EDSDK.dll` と `EdsImage.dll` のディレクトリを、起動前に指定してください。

```powershell
$env:EDSDK_PYTHON_DLL_DIR = "C:\path\to\EDSDK_64\Dll"
uv run python examples/python/build_inverse_lut.py
```

`sample.py` と `build_inverse_lut.py` の Canon 撮影では v0.2.0 のトリガモードを
自動利用します。1 パス分の投影・撮影とRAW転送を先に終え、その後に一時ファイルから
RAW現像などの処理を順次行います。逆LUTのHDRも、基準露光を一括処理した後、
飽和または黒潰れしたパターンだけを次の露光パスで再撮影するため、従来と同じ
適応ブラケットを保ったまま現像待ちを投影ループから分離できます。一時ファイルは
各パスの終了時に削除されます。

> 撮影が上記のいずれかで失敗する場合、まずカメラの電源を入れ直し、EOS Utility など他の Canon アプリを閉じ、モードダイヤルを Manual/Av/Tv にしてから再試行してください。

## パッケージ構成

- `packages/core`（配布名 `yoshidacomp`）— ハードウェア非依存の計算コア
- `examples/python/` — 撮影を含むCLI

```python
from yoshidacomp.config import get_config
from yoshidacomp.compensation.pipeline import generate_projection_image
```

撮影機能を使う場合は `uv sync --extra sample` を実行する。

このリポジトリは [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) で、
ルートの `pyproject.toml` が `packages/core`（`yoshidacomp`）に依存する形になっている。
`src/python/` は廃止済みで存在しない。

## プロジェクト構成

```text
Yoshida-Compensation/
├── config.toml                  # プロジェクト設定ファイル
├── pyproject.toml               # ワークスペース定義（yoshidacomp に依存）
├── packages/core/
│   ├── pyproject.toml           # yoshidacomp パッケージ定義
│   └── yoshidacomp/
│       ├── config.py                    # 設定ローダー
│       ├── capture/                     # カメラ抽象層と各バックエンド実装
│       ├── calibration/                 # 校正スクリプト（チェッカーボード・PnP等）
│       ├── geometry/warp.py             # 幾何ワーピング（c2p/p2c）
│       └── compensation/
│           ├── color_mixing_matrix.py       # パターン生成・カラーミキシング行列算出
│           ├── photometric_compensation.py  # 補償画像の計算
│           ├── projection_region.py         # 投影領域の絞り込み
│           ├── inverse_lut.py                # 逆LUT計測・適用
│           └── pipeline.py                   # 生成パイプライン一式
├── examples/python/
│   ├── sample.py                # 補償パイプラインのサンプル実装
│   └── capture.py               # キャプチャ＆ワーピングのサンプル
├── external/
│   └── GrayCode/                # 画像ワーピング用サブモジュール
└── data/
    ├── target_images/           # 目標画像の格納先
    ├── linear_proj_patterns/    # 生成されたリニアパターン
    ├── inv_gamma_proj_patterns/ # 逆ガンマ補正済みパターン
    ├── captured_images/         # カメラキャプチャ画像
    ├── compensation_images/     # 補償画像（リニア）
    └── inv_gamma_comp_images/   # 補償画像（逆ガンマ補正済み）
```

## 設定

`config.toml` を編集して、プロジェクタ・カメラ・パスなどの設定を行います。

CLI で別設定を使いたい場合は、`--config <path>`（または `-c <path>`）を指定できます。

`config.toml` が未作成の場合は、Python からデフォルト設定ファイルを生成できます。

```python
from yoshidacomp.config import generate_config_file

generate_config_file()  # プロジェクトルートに config.toml を生成
```

既存ファイルを上書きしたい場合は `generate_config_file(overwrite=True)` を使用してください。

### カメラバックエンド切替

- `camera.backend = "canon_edsdk"`: Canon + EDSDK を使用
- `camera.backend = "opencv"`: 汎用カメラを使用（`camera.device_index` を参照）

```toml
[projector]
gamma = 2.2
width = 1920
height = 1080
pos_x = 5360    # プロジェクタウィンドウの X 位置
pos_y = 0

[camera]
backend = "canon_edsdk" # "canon_edsdk" または "opencv"
av = "8"
tv = "1/15"
iso = "400"
image_quality = "LR"
device_index = 0
wait_key_ms = 200
trigger_wait_key_ms = 1

[paths]
c2p_map = ""                              # C2P 対応マップ (.npy)
p2c_map = ""                              # P2C 対応マップ (.npy)
warp_method = "p2c"                       # "c2p" または "p2c"
target_image_space = "projector"          # "camera" または "projector"
rpcc_matrix = ""                          # RPCC 行列 (.npy)
target_image_dir = "data/target_images"
# ... 他のパスはデフォルト値あり

[compensation]
num_divisions = 3       # パターン分割数 (パターン総数 = num_divisions^3)
use_gpu = true          # GPU を使用するか
```

バックエンドごとの設定例:

```toml
# Canon (EDSDK)
[camera]
backend = "canon_edsdk"
av = "8"
tv = "1/15"
iso = "400"
image_quality = "LR"
wait_key_ms = 200
trigger_wait_key_ms = 1
```

```toml
# Generic camera (OpenCV)
[camera]
backend = "opencv"
device_index = 0
wait_key_ms = 200
```

主な設定項目:

| セクション | キー | 説明 |
| --- | --- | --- |
| `projector` | `gamma` | プロジェクタのガンマ値 |
| `projector` | `width`, `height` | プロジェクタの解像度 |
| `projector` | `pos_x`, `pos_y` | マルチディスプレイ環境でのプロジェクタウィンドウ位置 |
| `projector` | `active_x`, `active_y`, `active_width`, `active_height` | 実際に発光させる投影領域。幅または高さが0以下なら全画面 |
| `camera` | `backend` | 使用するカメラ実装（`canon_edsdk` / `opencv`） |
| `camera` | `device_index` | OpenCVバックエンドのカメラインデックス |
| `camera` | `av`, `tv`, `iso`, `image_quality` | Canon EDSDK バックエンドの撮影パラメータ |
| `camera` | `wait_key_ms`, `trigger_wait_key_ms` | 初回・通常撮影の投影待機時間と、Canonトリガバッチ内の画像切替待機時間（ms） |
| `paths` | `c2p_map` | カメラ-プロジェクタ間のピクセル対応マップ（`.npy`） |
| `paths` | `p2c_map` | プロジェクタ-カメラ間のピクセル対応マップ（`.npy`） |
| `paths` | `warp_method` | ワーピング時に使う対応方向（`c2p` / `p2c`） |
| `paths` | `target_image_space` | 目標画像が属する座標系（`camera` / `projector`） |
| `compensation` | `num_divisions` | 色空間の分割数（パターン総数 = n³） |
| `compensation` | `use_gpu` | GPU アクセラレーションの有効化 |

## 使い方

### 補償パイプラインの実行

1. `data/target_images/` に目標画像（`.png` / `.jpg`）を配置します。
2. `config.toml` で `paths.warp_method` を `c2p` または `p2c` に設定し、対応する `paths.c2p_map` / `paths.p2c_map` を指定します。
3. 同じく `paths.target_image_space` に、`data/target_images/` の画像が `camera` 座標か `projector` 座標かを指定します。`sample.py` はこの値を見て、補償前にターゲット画像をワープする必要があるかを判定します。
4. 必要なら `projector.active_*` で投影領域を絞ります。校正パターン・逆LUT計測・補償画像は領域外が黒になり、projector座標の目標画像はこの領域へリサイズ配置されます。
5. 以下のコマンドで補償パイプラインを実行します。

```bash
uv run python examples/python/sample.py
```

別の設定ファイルを使う場合:

```bash
uv run python examples/python/sample.py --config path/to/config.toml
```

このスクリプトは以下の処理を自動で行います:

1. 投影パターンの生成（リニア＆逆ガンマ補正済み）
2. プロジェクタにパターンを表示し、カメラでキャプチャ
3. キャプチャ画像からカラーミキシング行列を算出
4. 目標画像に対する補償画像を計算
5. 幾何学的ワーピングと逆ガンマ補正を適用
6. 結果を `data/` 以下に保存

### キャプチャ＆ワーピングのみ実行

カメラキャプチャとワーピングだけを行いたい場合:

```bash
uv run python examples/python/capture.py
```

別の設定ファイルを使う場合:

```bash
uv run python examples/python/capture.py --config path/to/config.toml
```

`sample.py` を通常実行すると、ワーピング確認用の画像が
`data/sample_warp_debug_DEBUG_ONLY/` に自動保存されます。
このフォルダはデバッグ専用で、`sample.py` 実行のたびに再生成されます。

ワーピングだけを行った中間画像は
`data/sample_warp_debug_DEBUG_ONLY/warp_only_targets/` に集約保存されます。

保存内容:

1. 元のターゲット画像
2. プロジェクタ上で使う中央配置領域の可視化
3. 実際に通った warp / inverse warp の入力画像
4. 実際に通った warp / inverse warp の出力画像
5. 対象ごとの `metadata.json` と実行全体の `session.json`

## 色変換行列ビューア

ブラウザ上で色変換行列を見る、オフライン専用のビューア。
**現況・アーキテクチャ・次の作業は
[`docs/matrix_viewer_status_20260803.md`](docs/matrix_viewer_status_20260803.md) にまとめてある。**

```bash
uv sync --extra sample --extra web
uv run --no-sync python -m yoshidacomp_web
```

`http://127.0.0.1:8000` を開き、行列の `.npy` パスを入力する。

表示できるビュー:

| key | 内容 |
| --- | --- |
| `coef_tiles` | 4行(in_R/in_G/in_B/bias)×3列(out_R/out_G/out_B) の係数マップ |
| `surface` | 投影面の色（順方向モデル `C = (P − b)·A⁻¹`） |
| `compensation` | ターゲット画像に行列を適用した補償結果と投影コード値 |
| `gamut` | 補償が破綻する画素（負クリップ / 1超クリップ / 両方 / 範囲内） |

ホバーするとどのビューでもその画素の 4×3 係数が読める。ターゲット画像と逆LUTを
読み込めば、線形化方式（なし / 解析ガンマ / 実測逆LUT）を切り替えて比較でき、
LUTカーブも表示される。`POST /api/generate` で CLI と同じ投影画像を保存できる。

**カメラ座標の行列はまだプロジェクタ座標へワープできない**（計画B5）。
現状は明示的に 400 で断る。

ポート 8000 が別のプロセスに使われている環境では、起動が
`[WinError 10013] ソケットにアクセスしようとしました` で失敗する。
`python -m yoshidacomp_web` はポートを 8000 に固定しているため、
その場合は uvicorn を直接呼んでポートを変える。

```bash
uv run --no-sync python -m uvicorn yoshidacomp_web.app:app --host 127.0.0.1 --port 8123
```

使用中のプロセスは `Get-NetTCPConnection -LocalPort 8000` で確認できる。

実測した性能（1080pで0.70秒、カメラ空間1.56 GBで18.2秒、ヒープ2980 MB）は
[`docs/web_viewer_performance_20260731.md`](docs/web_viewer_performance_20260731.md)
にまとめてある。

## ライブラリとしての利用

`yoshidacomp` パッケージ（`packages/core/yoshidacomp/`）を直接インポートして利用することもできます。

```python
from yoshidacomp.compensation.color_mixing_matrix import (
    generate_projection_patterns,
    apply_inverse_gamma_correction,
    calc_color_mixing_matrices,
)
from yoshidacomp.compensation.photometric_compensation import calc_compensation_image
from yoshidacomp.compensation.pipeline import generate_projection_image
from yoshidacomp.config import get_config, generate_config_file
```

## 開発

### テストの実行

```bash
uv run pytest
```
