# Yoshida-Compensation

プロジェクタ-カメラシステムにおける色補償（Photometric Compensation）パイプラインの実装です。
プロジェクタから投影したパターンをカメラでキャプチャし、ピクセルごとのカラーミキシング行列を算出することで、目標画像に近い投影結果を得るための補償画像を生成します。

## 必要な環境

- **Python** 3.13 以上
- **uv**（パッケージマネージャ）
- **CUDA 対応 GPU**（推奨・なくても CPU で動作可能）
- **カメラデバイス**（`canon_edsdk` または `opencv` バックエンド）

カメラバックエンドごとの要件:

- `canon_edsdk`: Canon カメラ + EDSDK 環境、`rawpy`
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

## プロジェクト構成

```text
Yoshida-Compensation/
├── config.toml                  # プロジェクト設定ファイル
├── pyproject.toml               # パッケージ定義・依存関係
├── src/python/
│   ├── config.py                # 設定ローダー
│   ├── camera/                  # カメラ抽象層と各バックエンド実装
│   ├── color_mixing_matrix.py   # パターン生成・カラーミキシング行列算出
│   └── photometric_compensation.py  # 補償画像の計算
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

[paths]
c2p_map = ""                              # C2P 対応マップ (.npy)
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
| `camera` | `backend` | 使用するカメラ実装（`canon_edsdk` / `opencv`） |
| `camera` | `device_index` | OpenCVバックエンドのカメラインデックス |
| `camera` | `av`, `tv`, `iso`, `image_quality` | Canon EDSDK バックエンドの撮影パラメータ |
| `paths` | `c2p_map` | カメラ-プロジェクタ間のピクセル対応マップ（`.npy`） |
| `compensation` | `num_divisions` | 色空間の分割数（パターン総数 = n³） |
| `compensation` | `use_gpu` | GPU アクセラレーションの有効化 |

## 使い方

### 補償パイプラインの実行

1. `data/target_images/` に目標画像（`.png` / `.jpg`）を配置します。
2. `config.toml` の `paths.c2p_map` にカメラ-プロジェクタ対応マップのパスを設定します。
3. 以下のコマンドで補償パイプラインを実行します。

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

## ライブラリとしての利用

`src/python/` 内のモジュールを直接インポートして利用することもできます。

```python
from src.python.color_mixing_matrix import (
    generate_projection_patterns,
    apply_inverse_gamma_correction,
    calc_color_mixing_matrices,
)
from src.python.photometric_compensation import calc_compensation_image
from src.python.config import get_config
```

## 開発

### テストの実行

```bash
uv run pytest
```
