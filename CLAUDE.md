# CLAUDE.md

**このリポジトリの手引きは [`AGENTS.md`](AGENTS.md) にある。作業を始める前に読むこと。**

`AGENTS.md` の内容をここに複製しない。更新は `AGENTS.md` 側だけで行う。

## 最初に読む場所

| 知りたいこと | 行き先 |
| --- | --- |
| このリポジトリは何か / 全体像 | `AGENTS.md` 第1章 |
| いま何が進行中か | `AGENTS.md` 第2章（作業ストリームが2本ある） |
| 環境・起動・テストの実行 | `AGENTS.md` 第3章 |
| ディレクトリとモジュールの構造 | `AGENTS.md` 第4章 |
| 色変換行列ビューア（GUI）の現況 | `AGENTS.md` 第5章 → `docs/matrix_viewer_status_20260803.md` |
| 実機の再校正（未コミット・進行中） | `AGENTS.md` 第6章 |
| 補償パイプラインそのものの挙動 | `docs/current_color_compensation_pipeline.md` |
| 未解決事項の一覧 | `AGENTS.md` 第13章 |

## 読む前に踏むと壊れるもの

以下は `AGENTS.md` を読む前でも守ること。理由と詳細は同ファイルにある。

- **`uv sync` は必ず `--extra sample --extra web` を付ける。**
  素の `uv sync` は EDSDK / rawpy と手置きの DLL を消す。`.venv` を削除しない。
- **`git checkout --` / `git stash` / `git clean` を安易に使わない。**
  2026-08-04〜05 の実機作業が未コミットで残っている（`AGENTS.md` 第6章）。
- **`data/` に一時ファイルを書かない。** 再生成できない実測が約99GBある。
  一時ファイルはスクラッチパッドへ。
- **ビューアの起動でポート 8000 を使わない。** 別プロセスが占有している。
  `uv run --no-sync python -m uvicorn yoshidacomp_web.app:app --host 127.0.0.1 --port 8123`
- **緑のテストを「守られている」根拠にしない。** 落ちることができないテストが26件出ている
  （`AGENTS.md` 第8章）。
- **GLSL で `isnan()` / `v != v` を書かない。** ANGLE/D3D11 で消える。IEEE ビットパターンを使う。

## コミットメッセージ

`.github/instructions/commit-message.instructions.md` の規約に従う（要約は `AGENTS.md` 第11章）。
確認できない事実を書かない。テスト未実行なら「未実行」と書く。
