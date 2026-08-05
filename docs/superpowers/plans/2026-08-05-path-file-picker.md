# パス入力欄のファイル選択 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ビューアの4つのパス欄に、サーバのディレクトリを辿って選ぶモーダルのファイラを足す。

**Architecture:** サーバに `GET /api/browse`（1階層の一覧）と `GET /api/browse/suffixes`（拡張子表）を足し、フロントの新規モジュール `static/js/filepicker.js` がそれを描く。ファイラはテキスト欄を埋めるだけで、読み込み・生成の既存4経路には触れない。マシン全体を閲覧可能にするため、`TrustedHostMiddleware` で DNS リバインディングを塞ぐ。

**Tech Stack:** FastAPI / Starlette、素の ES モジュール（ビルド工程なし）、pytest、`tests/js/frontend_harness.mjs`（node、依存なし）

## Global Constraints

仕様書は `docs/superpowers/specs/2026-08-05-file-picker-design.md`。以下は全タスクに掛かる。

- **`uv sync` は必ず `--extra sample --extra web` を付ける。** 素の `uv sync` は EDSDK / rawpy と手置きの DLL を消す。依存を変えないので、基本は `uv run --no-sync` を使う。
- **`data/` に書かない。** 再生成できない実測が約99GBある。一時ファイルはスクラッチパッドへ。
- **`git checkout --` / `git stash` / `git clean` を使わない。** 2026-08-04〜05 の実機作業が未コミットで残っている。`git add` は必ずパスを明示する（`git add -A` を使わない）。
- テストコマンドは `uv run --no-sync python -m pytest -q`。全体で約3分36秒（394 passed、2026-08-05 実測）。
- **ビューアの起動は必ずポート 8123。** 8000 は別プロセスが占有している。
  `uv run --no-sync python -m uvicorn yoshidacomp_web.app:app --host 127.0.0.1 --port 8123`
- コミットメッセージは `.github/instructions/commit-message.instructions.md` の規約に従う。1行目は `[種別] 要約`（種別は `add`/`fix`/`update`/`change`/`clean`/`remove` など）、3行目以降はファイル単位で「変更点 / 理由・意図 / 影響・注意」。**確認できない事実を書かない。テスト未実行なら「未実行」と書く。**
- ブランチは `feature/path-file-picker`（作成済み、仕様書のコミット `0f9ed12` が乗っている）。
- **緑を「守られている」根拠にしない。** このリポジトリには落ちることのできないテストが26件ある。新しいチェックは必ず `_MUTATIONS` に対応する変異を1つずつ足し、そのチェックだけが落ちることを確認する。

## File Structure

| ファイル | 責務 |
| --- | --- |
| `packages/web/yoshidacomp_web/browse.py`（新規） | ディレクトリ一覧の純粋なロジック。FastAPI に依存しない。`Entry` / `Listing` / `list_directory` / `drive_roots` / `SUFFIX_TABLE` |
| `packages/web/yoshidacomp_web/routers/browse.py`（新規） | `GET /api/browse` と `GET /api/browse/suffixes`。例外を HTTP ステータスに翻訳するだけ |
| `packages/web/yoshidacomp_web/app.py`（変更） | ルータ登録と `TrustedHostMiddleware` |
| `packages/web/yoshidacomp_web/static/js/filepicker.js`（新規） | モーダルの状態と描画。`pickPath()` ひとつを export |
| `packages/web/yoshidacomp_web/static/index.html`（変更） | 4つの `参照` ボタンと、モーダルの器 |
| `packages/web/yoshidacomp_web/static/style.css`（変更） | モーダルの見た目 |
| `packages/web/yoshidacomp_web/static/js/main.js`（変更） | `参照` ボタンの結線のみ。ファイラの中身は持たない |
| `tests/test_web_browse.py`（新規） | 一覧APIとホスト検査 |
| `tests/js/frontend_harness.mjs`（変更） | `picker` シナリオと、DOMスタブへの `dispatchEvent` 追加 |
| `tests/test_web_frontend_js.py`（変更） | `picker` シナリオの起動と `_MUTATIONS` |

---

### Task 1: 一覧APIの計算部分（`browse.py`）

FastAPI を通さない純粋な関数から作る。並び順・フィルタ・打ち切りはすべてここの性質で、
HTTP を経由せずに検証できる。

**Files:**
- Create: `packages/web/yoshidacomp_web/browse.py`
- Test: `tests/test_web_browse.py`

**Interfaces:**
- Consumes: `yoshidacomp.compensation.pipeline.SUPPORTED_TARGET_IMAGE_SUFFIXES`（既存のタプル）
- Produces:
  - `MAX_ENTRIES: int = 2000`
  - `SUFFIX_TABLE: dict[str, tuple[str, ...]]` — キーは `"matrix"` / `"target"` / `"lut"`
  - `Entry(name: str, is_dir: bool, size: int | None, mtime: float | None)` — frozen dataclass
  - `Listing(path: Path, parent: Path | None, roots: list[str], entries: list[Entry], total: int, truncated: bool)` — frozen dataclass
  - `drive_roots() -> list[str]`
  - `list_directory(path: Path, suffixes: tuple[str, ...] = ()) -> Listing`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_browse.py` を新規作成する。

```python
"""ディレクトリ一覧APIのテスト。

並び順・フィルタ・打ち切りは ``browse.list_directory`` の性質なので HTTP を
通さずに検証する。HTTP 層に残るのは例外からステータスへの翻訳だけで、それは
Task 2 で別に検証する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "web"))

from yoshidacomp_web import browse  # noqa: E402


def _tree(root: Path) -> None:
    """ディレクトリ2つ・ファイル4つの木を作る。"""
    (root / "Beta").mkdir()
    (root / "alpha").mkdir()
    (root / "B.npy").write_bytes(b"0123456789")
    (root / "a.npy").write_bytes(b"01")
    (root / "c.npz").write_bytes(b"012")
    (root / "readme.txt").write_text("x", encoding="utf-8")


def test_directories_come_first_and_both_groups_are_casefold_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """並びは「ディレクトリ先・ファイル後」、いずれも casefold 順。

    ``os.listdir`` を差し替えるのは、NTFS が名前順で返してしまうため。実際に
    未整列の入力を渡せないと、このテストは並べ替えを一切していない実装でも
    通る（``target.py:88-91`` に同じ理由の記録がある）。
    """
    _tree(tmp_path)
    real = os.listdir
    monkeypatch.setattr(
        browse.os, "listdir",
        lambda p: sorted(real(p), reverse=True) if Path(p) == tmp_path else real(p),
    )

    listing = browse.list_directory(tmp_path)

    assert [(e.name, e.is_dir) for e in listing.entries] == [
        ("alpha", True), ("Beta", True),
        ("a.npy", False), ("B.npy", False), ("c.npz", False), ("readme.txt", False),
    ]


def test_suffix_filter_drops_files_but_never_directories(tmp_path: Path) -> None:
    """フィルタはファイルにだけ効く。

    ディレクトリを消すと、目的のファイルがある階層へ辿り着けなくなる。
    """
    _tree(tmp_path)

    listing = browse.list_directory(tmp_path, (".npy",))

    assert [e.name for e in listing.entries] == ["alpha", "Beta", "a.npy", "B.npy"]


def test_a_file_size_is_reported_and_a_directory_has_none(tmp_path: Path) -> None:
    """サイズはファイルにのみ付く。ディレクトリの「サイズ」は意味を持たない。"""
    _tree(tmp_path)

    by_name = {e.name: e for e in browse.list_directory(tmp_path).entries}

    assert by_name["B.npy"].size == 10
    assert by_name["alpha"].size is None


def test_a_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        browse.list_directory(tmp_path / "nope")


def test_a_file_raises_not_a_directory(tmp_path: Path) -> None:
    target = tmp_path / "a.npy"
    target.write_bytes(b"0")
    with pytest.raises(NotADirectoryError):
        browse.list_directory(target)


def test_an_unreadable_directory_raises_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``C:/System Volume Information`` の類。500 にしてはいけない。

    Windows で権限を落として実際に作る方法は移植性が無いので注入する。
    """
    def deny(_path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(browse.os, "listdir", deny)

    with pytest.raises(PermissionError):
        browse.list_directory(tmp_path)


def test_a_long_listing_is_cut_but_reports_its_true_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """黙って切り詰めない。``total`` は実数、``truncated`` は真。"""
    monkeypatch.setattr(browse, "MAX_ENTRIES", 3)
    for i in range(5):
        (tmp_path / f"f{i}.npy").write_bytes(b"0")

    listing = browse.list_directory(tmp_path)

    assert len(listing.entries) == 3
    assert listing.total == 5
    assert listing.truncated is True


def test_a_short_listing_is_not_marked_truncated(tmp_path: Path) -> None:
    _tree(tmp_path)
    listing = browse.list_directory(tmp_path)
    assert listing.total == 6
    assert listing.truncated is False


def test_a_root_has_no_parent(tmp_path: Path) -> None:
    """ドライブ直下（POSIX なら ``/``）では ``↑ 上へ`` の行き先が無い。"""
    root = Path(tmp_path.anchor)
    assert browse.list_directory(root).parent is None


def test_a_subdirectory_has_its_parent(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    assert browse.list_directory(tmp_path / "sub").parent == tmp_path.resolve()


def test_every_reported_root_exists(tmp_path: Path) -> None:
    """存在しないドライブを選択肢に出さない。"""
    roots = browse.drive_roots()
    assert roots
    assert all(Path(r).exists() for r in roots)


def test_the_suffix_table_takes_target_images_from_the_pipeline() -> None:
    """ターゲットの拡張子はフロントにもここにも書かない。出所は1つ。"""
    from yoshidacomp.compensation.pipeline import SUPPORTED_TARGET_IMAGE_SUFFIXES

    assert browse.SUFFIX_TABLE["target"] == tuple(SUPPORTED_TARGET_IMAGE_SUFFIXES)
    assert browse.SUFFIX_TABLE["matrix"] == (".npy",)
    assert browse.SUFFIX_TABLE["lut"] == (".npz",)
```

- [ ] **Step 2: 落ちることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_browse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'yoshidacomp_web.browse'` が全件に出る。

- [ ] **Step 3: `browse.py` を書く**

```python
"""パス入力欄のファイラが読むディレクトリ一覧。

``routers/browse.py`` から切り離してあるのは、並び順・フィルタ・打ち切りが
すべてこの層の性質だからである。HTTP を通さずに検証できるものを HTTP 越しに
検証すると、落ちた理由が「一覧が違う」のか「翻訳が違う」のか分からなくなる。

このモジュールはディスクを**読むだけ**である。作成・削除・改名はしない。
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from pathlib import Path

from yoshidacomp.compensation.pipeline import SUPPORTED_TARGET_IMAGE_SUFFIXES

# 1回の応答に載せる最大件数。``data/`` には約99GBの実測があり、1ディレクトリの
# 実件数は分かっていない。この値は実測に基づかない暫定値である（仕様書8節）。
MAX_ENTRIES = 2000

# 4つの入力欄が使うフィルタの唯一の出所。フロントには拡張子を書かない。
# ``target`` を ``pipeline`` から取るのは、対応拡張子を増やしたときに
# ファイラだけが古い集合を持つ状態を作らないため。
SUFFIX_TABLE: dict[str, tuple[str, ...]] = {
    "matrix": (".npy",),
    "target": tuple(SUPPORTED_TARGET_IMAGE_SUFFIXES),
    "lut": (".npz",),
}


@dataclass(frozen=True)
class Entry:
    """一覧の1行。``size`` と ``mtime`` は読めなければ ``None``。"""

    name: str
    is_dir: bool
    size: int | None
    mtime: float | None


@dataclass(frozen=True)
class Listing:
    """1ディレクトリ分の応答。

    ``total`` は打ち切り**前**の件数で、``entries`` の長さとは一致しないことが
    ある。UIはこの2つの差を「128件中2000件を表示」と出す。黙って切らない。
    """

    path: Path
    parent: Path | None
    roots: list[str]
    entries: list[Entry]
    total: int
    truncated: bool


def drive_roots() -> list[str]:
    """選択肢に出せるルート。Windows は存在するドライブ、他は ``/`` のみ。

    存在しないドライブを出さないために ``exists()`` で確かめる。空のカードリーダ
    などは文字が割り当たっていても開けない。
    """
    if os.name != "nt":
        return ["/"]
    return [f"{letter}:/" for letter in string.ascii_uppercase if Path(f"{letter}:/").exists()]


def list_directory(path: Path, suffixes: tuple[str, ...] = ()) -> Listing:
    """*path* の直下を1階層だけ返す。再帰しない。

    Args:
        path: 一覧するディレクトリ。
        suffixes: 小文字の拡張子。空なら全ファイル。**ディレクトリには効かない** --
            フィルタでディレクトリを消すと、目的のファイルがある階層へ降りられなくなる。

    Raises:
        FileNotFoundError: 存在しない。
        NotADirectoryError: 存在するがディレクトリではない。
        PermissionError: 読む権限が無い。``C:/System Volume Information`` の類は
            マシン全体を辿れる以上ありふれた通り道であり、500 にしてはならない。
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"no such directory: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")

    # ``os.listdir`` はモジュール経由で呼ぶ（``from os import listdir`` にしない）。
    # NTFS はファイルシステム自身が名前順で返すため、テストが本当に未整列の入力を
    # 渡せないと並び順のテストが落ちられなくなる。``target.py:88-91`` に同じ理由の
    # 記録がある。
    names = os.listdir(resolved)

    dirs: list[Entry] = []
    files: list[Entry] = []
    # 先に一度だけ並べ、そのまま2つに振り分ける。casefold 順なのは
    # ``target.py::_listing`` がこの順で ``names`` を作り、``LoadTargetRequest.index``
    # がその添字だからである。画面の並びと ``index`` の基準がずれると、選んだ
    # つもりのない画像が読まれる。
    for name in sorted(names, key=str.casefold):
        full = resolved / name
        is_dir = full.is_dir()          # OSError は False として返る
        try:
            stat = full.stat()
            size = None if is_dir else stat.st_size
            mtime = stat.st_mtime
        except OSError:
            # 壊れたリンク、列挙中に消えたファイル、開けないデバイス。
            # **行は落とさない。** 落とすと ``total`` が実数でなくなり、
            # 「黙って切らない」という約束が名前だけになる。
            size, mtime = None, None
        (dirs if is_dir else files).append(Entry(name, is_dir, size, mtime))

    filtered = files if not suffixes else [
        e for e in files if e.name.lower().endswith(suffixes)
    ]
    ordered = dirs + filtered
    total = len(ordered)
    # ``Path("C:/").parent`` は ``Path("C:/")`` 自身。そこが上端である。
    parent = resolved.parent if resolved.parent != resolved else None
    return Listing(
        path=resolved,
        parent=parent,
        roots=drive_roots(),
        entries=ordered[:MAX_ENTRIES],
        total=total,
        truncated=total > MAX_ENTRIES,
    )
```

- [ ] **Step 4: 通ることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_browse.py -q`
Expected: PASS（13件）

- [ ] **Step 5: 打ち切りのテストが本当に落ちることを確認する**

`MAX_ENTRIES` を読まずに常に全件返す実装でも通らないことを、手で一度だけ確かめる。
`browse.py` の `entries=ordered[:MAX_ENTRIES]` を `entries=ordered` に、
`truncated=total > MAX_ENTRIES` を `truncated=False` に一時的に書き換えて実行する。

Run: `uv run --no-sync python -m pytest tests/test_web_browse.py -q`
Expected: FAIL — `test_a_long_listing_is_cut_but_reports_its_true_length` のみ。
確認したら**元に戻す**。

- [ ] **Step 6: コミット**

```bash
git add packages/web/yoshidacomp_web/browse.py tests/test_web_browse.py
git commit -F - <<'EOF'
[add] ディレクトリ一覧の計算部分とそのテストを追加

- packages/web/yoshidacomp_web/browse.py
  - 変更点: 新規追加。list_directory が1階層を返し、ディレクトリ先・casefold順に
    並べ、拡張子フィルタをファイルにのみ適用し、MAX_ENTRIES=2000 で打ち切って
    total と truncated を返す。SUFFIX_TABLE は4欄のフィルタの唯一の出所。
  - 理由/意図: 並び順・フィルタ・打ち切りは HTTP を通さずに検証できる性質なので
    ルータから切り離した。casefold 順は target.py::_listing と揃える必要がある
    （LoadTargetRequest.index がその添字であるため）。
  - 影響/注意: os.listdir はモジュール経由で呼ぶ。NTFS が名前順で返すため、
    テストが未整列の入力を注入できないと並び順のテストが落ちられない。
- tests/test_web_browse.py
  - 変更点: 新規追加、13件。
  - 影響/注意: pytest tests/test_web_browse.py -q で 13 passed を確認済み。
    打ち切りのテストは、切り詰めを外した実装で当該1件だけが落ちることを確認済み。
EOF
```

---

### Task 2: HTTPエンドポイントとホスト検査

**Files:**
- Create: `packages/web/yoshidacomp_web/routers/browse.py`
- Modify: `packages/web/yoshidacomp_web/app.py`
- Test: `tests/test_web_browse.py`（追記）

**Interfaces:**
- Consumes: `browse.list_directory`, `browse.SUFFIX_TABLE`, `browse.Listing`（Task 1）
- Produces:
  - `GET /api/browse?path=&suffixes=` → `{path, parent, roots, entries: [{name, is_dir, size, mtime}], total, truncated}`
  - `GET /api/browse/suffixes` → `{"matrix": [...], "target": [...], "lut": [...]}`
  - `app.py` に `ALLOWED_HOSTS: list[str]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_web_browse.py` の末尾に追記する。

```python
# ------------------------------------------------------------------ HTTP 層

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402


def test_browse_returns_the_listing_as_posix_paths(tmp_path: Path) -> None:
    """パスは POSIX 区切りで返す。Windows の ``\\`` は JSON でも URL でも扱いにくい。"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.npy").write_bytes(b"0")
    client = TestClient(create_app())

    body = client.get("/api/browse", params={"path": str(tmp_path)}).json()

    assert body["path"] == tmp_path.resolve().as_posix()
    assert body["parent"] == tmp_path.resolve().parent.as_posix()
    assert [e["name"] for e in body["entries"]] == ["sub", "a.npy"]
    assert body["total"] == 2
    assert body["truncated"] is False


def test_browse_with_no_path_lists_the_working_directory() -> None:
    """欄が空のまま ``参照`` を押したときの出発点。"""
    client = TestClient(create_app())

    body = client.get("/api/browse").json()

    assert body["path"] == Path.cwd().resolve().as_posix()


def test_browse_applies_the_suffix_filter_from_the_query(tmp_path: Path) -> None:
    (tmp_path / "keep.npy").write_bytes(b"0")
    (tmp_path / "drop.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    client = TestClient(create_app())

    body = client.get(
        "/api/browse", params={"path": str(tmp_path), "suffixes": ".npy"}
    ).json()

    assert [e["name"] for e in body["entries"]] == ["sub", "keep.npy"]


@pytest.mark.parametrize(
    ("make", "status"),
    [
        (lambda p: p / "missing", 404),
        (lambda p: p / "a.npy", 400),
    ],
)
def test_browse_translates_the_two_ordinary_mistakes(
    tmp_path: Path, make, status: int
) -> None:
    """存在しないパスと、ディレクトリでないパス。どちらも手打ちの日常。"""
    (tmp_path / "a.npy").write_bytes(b"0")
    client = TestClient(create_app())

    res = client.get("/api/browse", params={"path": str(make(tmp_path))})

    assert res.status_code == status
    assert str(make(tmp_path).name) in res.json()["detail"]


def test_browse_reports_an_unreadable_directory_as_403_not_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """マシン全体を辿れる以上、権限の無いディレクトリは通り道に必ず現れる。"""
    def deny(_path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(browse.os, "listdir", deny)
    client = TestClient(create_app())

    res = client.get("/api/browse", params={"path": str(tmp_path)})

    assert res.status_code == 403


def test_the_suffix_endpoint_serves_all_three_filters() -> None:
    client = TestClient(create_app())

    body = client.get("/api/browse/suffixes").json()

    assert body["matrix"] == [".npy"]
    assert body["lut"] == [".npz"]
    assert ".png" in body["target"]


# --------------------------------------------------------------- Host 検査
#
# CORS が無いため他オリジンのページは応答を読めないが、DNS リバインディングは
# その保護を迂回する（攻撃者のドメインを 127.0.0.1 に再解決させると、ブラウザ
# から見て同一オリジンになり CORS の判定自体が起きない）。ファイラはマシン全体の
# ファイル名を返すので、この検査とセットでなければ成立しない。


def test_a_foreign_host_header_is_refused() -> None:
    client = TestClient(create_app())

    res = client.get("/api/browse", headers={"Host": "evil.example"})

    assert res.status_code == 400
    assert "host" in res.text.lower()


@pytest.mark.parametrize("host", ["127.0.0.1:8123", "localhost:8123", "127.0.0.1"])
def test_the_loopback_hosts_are_allowed_with_and_without_a_port(host: str) -> None:
    client = TestClient(create_app())

    assert client.get("/api/health", headers={"Host": host}).status_code == 200


def test_the_host_check_also_covers_the_pre_existing_endpoints() -> None:
    """検査はアプリ全体に掛かる。ファイラだけを守っても意味が無い --
    ``POST /api/session/matrix`` は既に任意のパスを読み、``/api/generate`` は
    既に任意のパスへ書く。"""
    client = TestClient(create_app())

    res = client.post(
        "/api/session/matrix", json={"path": "x.npy"}, headers={"Host": "evil.example"}
    )

    assert res.status_code == 400
```

- [ ] **Step 2: 落ちることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_browse.py -q`
Expected: FAIL — 一覧側は 404（ルート未登録）、ホスト検査側は 200 が返る。

- [ ] **Step 3: ルータを書く**

`packages/web/yoshidacomp_web/routers/browse.py` を新規作成する。

```python
"""ディレクトリ一覧エンドポイント。

このルータは翻訳しかしない。並び順・フィルタ・打ち切りは ``browse.py`` の
性質であり、そちらでテストされている。

**このルータはディスクを読むだけである。** 作成・削除・改名の口は無い。
ファイラはテキスト欄を埋めるための道具であって、ファイル管理ツールではない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..browse import SUFFIX_TABLE, list_directory

router = APIRouter()


@router.get("/browse")
def browse_directory(
    path: str = Query("", description="一覧するディレクトリ。空ならサーバの作業ディレクトリ"),
    suffixes: str = Query("", description="ファイルに掛ける拡張子フィルタ（カンマ区切り）"),
) -> dict[str, Any]:
    """1階層分の一覧を返す。

    ``path`` が空のときに ``Path("")``（＝``Path(".")``）へ落とさず ``Path.cwd()``
    を使うのは、応答の ``path`` を絶対パスで返すためである。``.`` を返すと、
    フロントが「上へ」を作るときの基準が無くなる。
    """
    target = Path(path) if path.strip() else Path.cwd()
    wanted = tuple(s.strip().lower() for s in suffixes.split(",") if s.strip())
    try:
        listing = list_directory(target, wanted)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        # 500 にしない。マシン全体を辿れる以上、読めないディレクトリは
        # 例外的な事故ではなく通常の通り道である。
        raise HTTPException(
            status_code=403, detail=f"cannot list {target}: {exc.strerror or exc}"
        ) from exc

    return {
        "path": listing.path.as_posix(),
        "parent": listing.parent.as_posix() if listing.parent is not None else None,
        "roots": listing.roots,
        "entries": [
            {"name": e.name, "is_dir": e.is_dir, "size": e.size, "mtime": e.mtime}
            for e in listing.entries
        ],
        "total": listing.total,
        "truncated": listing.truncated,
    }


@router.get("/browse/suffixes")
def browse_suffixes() -> dict[str, list[str]]:
    """4つの入力欄が使うフィルタ。

    フロントに拡張子を書かないための口である。とくに ``target`` は
    ``yoshidacomp.compensation.pipeline.SUPPORTED_TARGET_IMAGE_SUFFIXES`` が
    唯一の出所で、対応形式が増えたときにファイラだけ古い集合を持つ状態を作らない。
    """
    return {key: list(value) for key, value in SUFFIX_TABLE.items()}
```

- [ ] **Step 4: `app.py` を書き換える**

`packages/web/yoshidacomp_web/app.py` の全体を次に置き換える。

```python
"""FastAPI application factory.

API は /api 配下、WebGL2 フロントエンドは / にマウントする。同一オリジンで
配信するため CORS は不要で、バイナリ応答の ``X-*`` ヘッダをブラウザから直接読める。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .routers import browse, generate, lut, session, target, view

STATIC_DIR = Path(__file__).parent / "static"

# ``Host`` ヘッダとして受け付ける名前。
#
# CORS が無いので他オリジンのページはこのサーバの応答を読めない -- が、その
# 保護は **DNS リバインディングで迂回できる**。攻撃サイトが自分のドメインを
# 127.0.0.1 に再解決させると、ブラウザから見て同一オリジンになり CORS の判定
# 自体が起きない。``/api/browse`` はマシン全体のファイル名を返すので、この
# 検査はその機能と一体である。
#
# ``testserver`` は ``TestClient`` の既定 ``base_url``。既存の ``TestClient(...)``
# は11ファイル85箇所にあり、全部に ``base_url`` を渡すのはこの変更に見合わない。
# 許して失うものは無い: DNS リバインディングは名前解決を要する攻撃であり、
# ``testserver`` は単一ラベルで登録可能なドメインではないので、ブラウザに
# この ``Host`` を送らせる経路が無い。``Host`` を自由に作れるローカルのプロセスは
# そもそもソケットへ直接届くので、この検査の対象ではない。
#
# IPv6 リテラルは入れない。``TrustedHostMiddleware`` はホスト部を
# ``host.split(":")[0]`` で取るので ``[::1]:8123`` は ``[`` になり、書いても
# 効かない。サーバは IPv4 ループバック束縛なので ``::1`` では到達もできない。
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]


def create_app() -> FastAPI:
    """Build the FastAPI app with all routers and the static mount."""
    app = FastAPI(title="Yoshida Compensation Matrix Viewer", version="0.1.0")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(session.router, prefix="/api", tags=["session"])
    app.include_router(target.router, prefix="/api", tags=["target"])
    app.include_router(lut.router, prefix="/api", tags=["lut"])
    app.include_router(view.router, prefix="/api", tags=["view"])
    app.include_router(generate.router, prefix="/api", tags=["generate"])
    app.include_router(browse.router, prefix="/api", tags=["browse"])

    # Mount the frontend last so /api/* wins. html=True serves index.html at /.
    # フロントは最後にマウントする。先にすると /api/* が奪われる。
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
```

- [ ] **Step 5: 通ることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_browse.py -q`
Expected: PASS（24件）

- [ ] **Step 6: 既存のテストを壊していないことを確認する**

`TrustedHostMiddleware` はアプリ全体に掛かるので、既存の85箇所の `TestClient` に
影響する。ここで全体を回す。

Run: `uv run --no-sync python -m pytest -q`
Expected: PASS（Task 1 以前の 394 に本タスクまでの追加分が乗った数）。約3分40秒。
**1件でも落ちたら `ALLOWED_HOSTS` の `testserver` を疑う。**

- [ ] **Step 7: ホスト検査が本当に効いていることを確認する**

`app.py` の `app.add_middleware(...)` の行を一時的にコメントアウトして実行する。

Run: `uv run --no-sync python -m pytest tests/test_web_browse.py -q`
Expected: FAIL — `test_a_foreign_host_header_is_refused` と
`test_the_host_check_also_covers_the_pre_existing_endpoints` の2件のみ。
確認したら**元に戻す**。

- [ ] **Step 8: コミット**

```bash
git add packages/web/yoshidacomp_web/routers/browse.py packages/web/yoshidacomp_web/app.py tests/test_web_browse.py
git commit -F - <<'EOF'
[add] 一覧エンドポイントと Host ヘッダ検査を追加

- packages/web/yoshidacomp_web/routers/browse.py
  - 変更点: 新規追加。GET /api/browse が1階層を返し、GET /api/browse/suffixes が
    4欄のフィルタを返す。例外の翻訳は 404/400/403 の3つだけ。
  - 理由/意図: PermissionError を 403 にするのは、マシン全体を辿れる以上、
    読めないディレクトリが例外的な事故ではなく通常の通り道になるため。
  - 影響/注意: 読み取り専用。作成・削除・改名の口は無い。
- packages/web/yoshidacomp_web/app.py
  - 変更点: browse ルータの登録と、TrustedHostMiddleware の追加。
  - 理由/意図: CORS が無いため他オリジンは応答を読めないが、DNS リバインディングは
    その保護を迂回する。マシン全体のファイル名を返す機能と一体の対策である。
  - 影響/注意: 全エンドポイントに掛かる。allowed_hosts に testserver を含めるのは
    TestClient の既定 base_url であるため（理由はコード内コメントに記載）。
- tests/test_web_browse.py
  - 変更点: HTTP層とホスト検査のテストを追記。合計24件。
  - 影響/注意: pytest tests/test_web_browse.py -q で 24 passed、
    pytest -q の全体も緑であることを確認済み。ミドルウェアを外すと
    ホスト検査の2件だけが落ちることを確認済み。
EOF
```

---

### Task 3: ファイラのモジュールと `path-input` への適用

ここでモーダルを1本作り、まず行列の欄だけに繋ぐ。残り3欄は Task 4。

**Files:**
- Create: `packages/web/yoshidacomp_web/static/js/filepicker.js`
- Modify: `packages/web/yoshidacomp_web/static/index.html`
- Modify: `packages/web/yoshidacomp_web/static/style.css`
- Modify: `packages/web/yoshidacomp_web/static/js/main.js`
- Modify: `tests/js/frontend_harness.mjs`
- Modify: `tests/test_web_frontend_js.py`

**Interfaces:**
- Consumes: `GET /api/browse`, `GET /api/browse/suffixes`（Task 2）
- Produces:
  - `filepicker.js`: `export async function pickPath({mode, suffixes, start, title}) -> Promise<string|null>`
    - `mode`: `"file"` | `"dir"` | `"file_or_dir"`
    - `suffixes`: `string[]`（空配列なら絞らない）
    - `start`: 出発点のパス（空文字ならサーバの作業ディレクトリ）
    - `title`: モーダルの見出し
    - 戻り値は選ばれたパス文字列。取消なら `null`
  - `index.html`: `btn-browse-matrix`、および `picker-*` の要素一式
  - `main.js`: `BROWSE_FIELDS` 表と `wireBrowseButtons()`

- [ ] **Step 1: `index.html` にモーダルの器と `参照` ボタンを足す**

`index.html:12-15` のツールバーを次に置き換える。

```html
    <input id="path-input" class="mono path" type="text" spellcheck="false"
           value="data/linear_pipeline_1080/matrices_linear_gamma.npy"
           placeholder="行列 .npy のパス (H, W, 4, 3)" />
    <button id="btn-browse-matrix">参照</button>
    <button id="btn-load" class="primary">読み込み</button>
```

同ファイルの `</main>` の直後（`<script type="module" src="js/main.js"></script>` の直前）に
モーダルを追加する。

```html
  <!-- The file picker, shared by all four path boxes.

       One dialog rather than four, because what differs between the boxes is
       three values (mode, suffix filter, title) and not any behaviour. Four
       copies would be four places to fix the next time a listing gets a column.

       It lives here, outside <main>, because it covers the whole page: nested
       inside the sidebar it would be clipped by that column's `overflow-y:
       auto` and appear as a scrollable sliver.

       Empty on purpose. Every row is written by js/filepicker.js from the
       `GET /api/browse` response -- names, sizes and the truncation notice are
       the server's facts, and a hard-coded row here would be a second source
       for one of them. The same argument as the class legend above.

       4つのパス欄で1つのダイアログを共有する。欄ごとに違うのは3つの値
       （モード・拡張子フィルタ・見出し）だけで、挙動は同じだからである。
       <main> の外に置くのは、サイドバーの中では `overflow-y: auto` に
       切り取られて細長い断片になるため。中身は空で、全行を filepicker.js が
       応答から書く。 -->
  <div id="picker-backdrop" class="picker-backdrop" hidden>
    <div id="picker" class="picker">
      <div class="picker-head">
        <span id="picker-title" class="lbl"></span>
        <select id="picker-roots" class="mono"></select>
        <span id="picker-cwd" class="mono path"></span>
      </div>
      <div class="picker-bar">
        <button id="picker-up">↑ 上へ</button>
        <span id="picker-count" class="muted mono"></span>
        <span id="picker-error" class="mono"></span>
      </div>
      <div id="picker-list" class="picker-list mono"></div>
      <div id="picker-newdir" class="picker-newdir" hidden>
        <label class="lbl" for="picker-newdir-name">新規フォルダ名（任意）</label>
        <input id="picker-newdir-name" class="mono" type="text" spellcheck="false" />
      </div>
      <div class="picker-foot">
        <!-- What 選択 will hand back, exactly. The three modes differ in what a
             press means -- a file in `file`, the current directory in `dir`,
             either in `file_or_dir` -- and this line is how that stops being
             something the user has to infer from the mode they did not choose.
             押したときに返る文字列そのもの。モードによって意味が変わるので、
             推測させずに出す。 -->
        <span id="picker-chosen" class="mono"></span>
        <div class="spacer"></div>
        <button id="picker-cancel">取消</button>
        <button id="picker-choose" class="primary">選択</button>
      </div>
    </div>
  </div>
```

- [ ] **Step 2: `style.css` にモーダルの見た目を足す**

`style.css` の末尾に追記する。

```css
/* --------------------------------------------------------------- file picker
   `position: fixed` on the backdrop, not `absolute`: the sidebar scrolls and
   an absolutely positioned overlay would scroll with it.
   背景は fixed。absolute だとサイドバーのスクロールに追従してしまう。 */
.picker-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.picker-backdrop[hidden] { display: none; }

.picker {
  display: flex;
  flex-direction: column;
  width: min(720px, 90vw);
  max-height: 80vh;
  gap: 8px;
  padding: 12px;
  background: var(--panel, #1b1b1f);
  border: 1px solid var(--line, #3a3a42);
  border-radius: 6px;
}

.picker-head, .picker-bar, .picker-foot {
  display: flex;
  align-items: center;
  gap: 8px;
}
.picker-head .path { flex: 1; overflow-x: auto; white-space: nowrap; }

/* The only element that scrolls. Fixed-height rows so a long listing does not
   reflow as sizes arrive. 唯一スクロールする要素。行高は固定。 */
.picker-list {
  flex: 1;
  min-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--line, #3a3a42);
}
.picker-row {
  display: flex;
  gap: 8px;
  padding: 2px 6px;
  cursor: pointer;
  white-space: nowrap;
}
.picker-row:hover { background: rgba(255, 255, 255, 0.06); }
.picker-row.selected { background: rgba(120, 170, 255, 0.22); }
.picker-row .size { margin-left: auto; opacity: 0.7; }

.picker-newdir { display: flex; align-items: center; gap: 8px; }
.picker-newdir[hidden] { display: none; }

/* Red only when there is something to say; `#picker-error` is always present
   and usually empty. 空のときは色を持たない。 */
#picker-error:not(:empty) { color: var(--err, #ff8080); }
```

- [ ] **Step 3: `filepicker.js` を書く**

`packages/web/yoshidacomp_web/static/js/filepicker.js` を新規作成する。

```js
// The file picker behind every 参照 button.
//
// `<input type="file">` cannot be used here: it hands back a name and the
// bytes, never the path, and this backend takes a path and reads the disk
// itself. So the listing is the server's (`GET /api/browse`) and this module
// only draws it.
//
// It fills a text box and nothing else. It does not load, it does not
// generate, and it never creates, deletes or renames anything on disk -- there
// is no endpoint for any of that. In `dir` mode the returned path may not
// exist yet; making it is `POST /api/generate`'s job, as it always was.
//
// `<input type="file">` は名前と中身しか返さずパスを返さないので使えない
// （バックエンドはパスを受け取って自分で読む）。一覧はサーバのもので、この
// モジュールは描くだけである。テキスト欄を埋める以外のことはしない。

const backdrop = () => document.getElementById("picker-backdrop");
const $ = (id) => document.getElementById(id);

// One dialog, so one live session at a time. Held here rather than passed
// around because the DOM listeners are wired once and have to reach whichever
// call is open. 開いているのは常に1つなのでここに持つ。
let session = null;

// Suffix filters come from the server (`GET /api/browse/suffixes`) so that the
// set of image extensions has one home. Fetched once and kept.
// 拡張子はサーバから取る（出所を1つにするため）。1度だけ取得して保持する。
let suffixTable = null;

export async function suffixesFor(key) {
  if (suffixTable === null) {
    const res = await fetch("/api/browse/suffixes");
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    suffixTable = await res.json();
  }
  return suffixTable[key] ?? [];
}

function humanSize(bytes) {
  if (bytes === null || bytes === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes, i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${i === 0 ? value : value.toFixed(1)} ${units[i]}`;
}

// The string 選択 would hand back right now, and the string `#picker-chosen`
// shows. The three modes differ only here.
//
// In `dir` mode the new-folder box is appended to the current directory, so an
// empty box means "this directory" -- the ordinary case -- and a filled one
// names a directory that does not exist yet. That is the whole of the
// new-folder affordance; nothing is created here.
// 3つのモードの違いはこの関数だけに閉じている。`dir` では新規フォルダ名を
// 現在地に連結する（空なら現在地そのもの）。作成はしない。
function chosenPath() {
  if (!session) return "";
  if (session.mode === "dir") {
    const name = $("picker-newdir-name").value.trim();
    return name ? `${session.cwd}/${name}` : session.cwd;
  }
  if (session.selected) return `${session.cwd}/${session.selected}`;
  return session.mode === "file_or_dir" ? session.cwd : "";
}

function renderFoot() {
  const chosen = chosenPath();
  $("picker-chosen").textContent = chosen;
  // `file` is the only mode with nothing to fall back on: a directory is not
  // an answer there. 「ファイルのみ」だけは既定の答えを持たない。
  $("picker-choose").disabled = chosen === "";
}

function renderList(body) {
  const host = $("picker-list");
  host.textContent = "";        // replaces the children, unregistering their ids
  body.entries.forEach((entry, i) => {
    const row = document.createElement("div");
    row.id = `picker-entry-${i}`;
    row.className = "picker-row";
    const name = document.createElement("span");
    name.textContent = entry.is_dir ? `${entry.name}/` : entry.name;
    row.appendChild(name);
    const size = document.createElement("span");
    size.className = "size";
    size.textContent = humanSize(entry.size);
    row.appendChild(size);
    row.addEventListener("click", () => {
      if (entry.is_dir) { navigate(`${body.path}/${entry.name}`); return; }
      session.selected = entry.name;
      for (const sibling of host.children) sibling.classList.remove("selected");
      row.classList.add("selected");
      renderFoot();
    });
    host.appendChild(row);
  });
  // The count is stated whenever it is not the whole story. A silent cut reads
  // as "this is everything in the directory".
  // 切り詰めたことは必ず言う。黙って切ると「これが全部」と読まれる。
  $("picker-count").textContent = body.truncated
    ? `${body.total}件中${body.entries.length}件を表示`
    : `${body.total}件`;
}

function renderRoots(roots, cwd) {
  const select = $("picker-roots");
  select.textContent = "";
  for (const root of roots) {
    const option = document.createElement("option");
    option.value = root;
    option.textContent = root;
    select.appendChild(option);
  }
  const match = roots.find((r) => cwd.toLowerCase().startsWith(r.toLowerCase()));
  if (match) select.value = match;
}

async function navigate(path) {
  const params = new URLSearchParams({ path });
  if (session.suffixes.length) params.set("suffixes", session.suffixes.join(","));
  const res = await fetch(`/api/browse?${params.toString()}`);
  if (!res.ok) {
    // Shown inside the dialog and the dialog stays where it was. Putting this
    // on the page's status bar would print it behind the backdrop, and a
    // refusal for a directory the user cannot see is a refusal they cannot act
    // on. モーダル内に出し、直前のディレクトリに留まる。
    let detail = await res.text();
    try {
      const parsed = JSON.parse(detail);
      if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
    } catch (_) { /* not JSON; show it as it arrived */ }
    $("picker-error").textContent = `${res.status}: ${detail}`;
    return;
  }
  const body = await res.json();
  $("picker-error").textContent = "";
  session.cwd = body.path;
  session.parent = body.parent;
  session.selected = null;      // a listing the selection does not belong to
  $("picker-cwd").textContent = body.path;
  $("picker-up").disabled = body.parent === null;
  renderRoots(body.roots, body.path);
  renderList(body);
  renderFoot();
}

function close(result) {
  const settle = session.settle;
  session = null;
  backdrop().hidden = true;
  settle(result);
}

// Wired once, at module load, because the elements are in index.html and the
// dialog is reused. 要素は index.html にあり使い回すので結線は1度きり。
function wireOnce() {
  $("picker-cancel").addEventListener("click", () => close(null));
  $("picker-choose").addEventListener("click", () => {
    const chosen = chosenPath();
    if (chosen !== "") close(chosen);
  });
  $("picker-up").addEventListener("click", () => {
    if (session && session.parent !== null) navigate(session.parent);
  });
  $("picker-roots").addEventListener("change", (e) => { navigate(e.target.value); });
  $("picker-newdir-name").addEventListener("input", () => { renderFoot(); });
  // A click on the backdrop is a cancel; a click inside the dialog is not.
  // 背景のクリックは取消。ダイアログ内は違う。
  backdrop().addEventListener("click", (e) => {
    if (session && e.target === backdrop()) close(null);
  });
}

let wired = false;

export async function pickPath({ mode, suffixes = [], start = "", title = "" }) {
  if (!wired) { wireOnce(); wired = true; }
  return new Promise((settle) => {
    session = { mode, suffixes, cwd: "", parent: null, selected: null, settle };
    $("picker-title").textContent = title;
    $("picker-error").textContent = "";
    $("picker-newdir-name").value = "";
    $("picker-newdir").hidden = mode !== "dir";
    backdrop().hidden = false;
    // A box holding a file's path should open in that file's directory, not at
    // the working directory -- the server answers 400 for a file, so the
    // directory part is what to ask for. Empty stays empty and the server picks.
    // ファイルのパスが入っている欄は、そのファイルのあるディレクトリで開く。
    const at = start.includes("/") || start.includes("\\")
      ? start.replace(/\\/g, "/").replace(/\/[^/]*$/, "")
      : start;
    navigate(at);
  });
}
```

- [ ] **Step 4: `main.js` に結線を足す**

`main.js` の import 群に追加する（既存の `import { drawLutCurve } ...` などの隣）。

```js
import { pickPath, suffixesFor } from "./filepicker.js";
```

`wireLoaders()` の直前（`main.js:1339` のあたり）に次を足す。

```js
// The four path boxes and what 参照 means for each of them.
//
// A table rather than four wirings, because the boxes differ only in these
// four values. `suffixKey` is a key into `GET /api/browse/suffixes` and not a
// list of extensions: the image formats are `pipeline.SUPPORTED_TARGET_IMAGE_
// SUFFIXES`'s to decide, and writing them here would be the second place they
// live.
//
// `gen-output-dir` is `dir` and the other write-related property follows from
// it: the picker returns a path that may not exist yet, and creating it stays
// `POST /api/generate`'s job.
//
// 4欄の違いはこの4つの値だけなので表にする。`suffixKey` が拡張子の並びでは
// なくキーなのは、画像形式の決定権が pipeline 側にあるためである。
const BROWSE_FIELDS = [
  { button: "btn-browse-matrix", input: "path-input", mode: "file",
    suffixKey: "matrix", title: "行列 .npy を選択" },
  { button: "btn-browse-target", input: "target-path", mode: "file_or_dir",
    suffixKey: "target", title: "ターゲット画像またはディレクトリを選択" },
  { button: "btn-browse-lut", input: "lut-path", mode: "file",
    suffixKey: "lut", title: "逆LUT .npz を選択" },
  { button: "btn-browse-output", input: "gen-output-dir", mode: "dir",
    suffixKey: null, title: "出力先ディレクトリを選択" },
];

function wireBrowseButtons() {
  for (const field of BROWSE_FIELDS) {
    $(field.button).addEventListener("click", () => {
      (async () => {
        const suffixes = field.suffixKey ? await suffixesFor(field.suffixKey) : [];
        const box = $(field.input);
        const chosen = await pickPath({
          mode: field.mode, suffixes, start: box.value.trim(), title: field.title,
        });
        if (chosen === null) return;      // cancelled: the box is not touched
        box.value = chosen;
        // Dispatched, not just assigned. `gen-output-dir`'s edit listeners
        // retire the 上書き agreement, and a value set without an event would
        // let an agreement about one directory be spent on another. The
        // press-time re-check catches it too, but a box whose listeners are
        // bypassed is a trap for the next listener added here.
        // 代入だけで済ませない。`gen-output-dir` の編集リスナは上書き合意を
        // 失効させる。押下時にも再確認しているが、リスナを迂回する欄は次に
        // リスナを足す人にとっての罠になる。
        box.dispatchEvent(new Event("input", { bubbles: true }));
        box.dispatchEvent(new Event("change", { bubbles: true }));
      })().catch((e) => status(errorText(e), true));
    });
  }
}
```

`main()` の `wireLoaders();` の直後に `wireBrowseButtons();` を足す（`main.js:2338` のあたり）。

```js
  wireDisplayControls();
  wireLoaders();
  wireBrowseButtons();
  wireGenerate();
```

- [ ] **Step 5: ハーネスの DOM スタブに `dispatchEvent` を足す**

`tests/js/frontend_harness.mjs` の `makeEl` 内、`__fire` の定義の直後に追加する。

```js
    // The page's own way to fire an event, as distinct from `__fire`, which is
    // the test's. `main.js` uses it to make a picked path indistinguishable
    // from a typed one, and without it here that call throws -- the stub must
    // not be the reason a behaviour cannot be checked.
    // ページ自身がイベントを出す口。`__fire` はテスト側の口で、別物である。
    dispatchEvent(event) { el.__fire(String(event.type), event); return true; },
```

同ファイルの `globalThis.ResizeObserver = ...` の隣に追加する。

```js
globalThis.Event = class { constructor(type, init = {}) { this.type = type; Object.assign(this, init); } };
```

- [ ] **Step 6: ハーネスに `/api/browse` の偽サーバを足す**

`globalThis.fetch` の中、`if (url === "/api/views") {` の直前に追加する。

```js
  // GET /api/browse と /api/browse/suffixes、routers/browse.py と同じ形で。
  //
  // `BROWSE_TREE` は絶対パスから一覧への写像。ディレクトリが先・casefold順に
  // 並んでいるのはサーバ側の性質で、ここではその形の応答を与えるだけである
  // （並べ替えそのものは tests/test_web_browse.py が見る）。
  if (String(url).startsWith("/api/browse?")) {
    const query = new URLSearchParams(String(url).slice("/api/browse?".length));
    const at = query.get("path") || "/repo";
    browseLog.push({ path: at, suffixes: query.get("suffixes") ?? "" });
    const node = BROWSE_TREE[at];
    if (!node) {
      return new Response(JSON.stringify({ detail: `no such directory: ${at}` }), {
        status: 404, headers: { "content-type": "application/json" },
      });
    }
    if (node.denied) {
      return new Response(JSON.stringify({ detail: `cannot list ${at}: Access is denied` }), {
        status: 403, headers: { "content-type": "application/json" },
      });
    }
    const wanted = (query.get("suffixes") || "").split(",").filter(Boolean);
    const entries = node.entries.filter(
      (e) => e.is_dir || !wanted.length || wanted.some((s) => e.name.toLowerCase().endsWith(s)),
    );
    return new Response(JSON.stringify({
      path: at, parent: node.parent, roots: ["/repo"],
      entries: entries.slice(0, node.cap ?? entries.length),
      total: node.total ?? entries.length,
      truncated: (node.cap ?? entries.length) < (node.total ?? entries.length),
    }), { headers: { "content-type": "application/json" } });
  }

  if (url === "/api/browse/suffixes") {
    return new Response(JSON.stringify({
      matrix: [".npy"], target: [".png", ".jpg"], lut: [".npz"],
    }), { headers: { "content-type": "application/json" } });
  }
```

同ファイルの `const requestLog = [];` の近くに追加する（無ければ他のログ配列の隣）。

```js
// Every /api/browse the picker issues, so "one navigation is one request" is
// observable. 移動1回が要求1回であることを外から見るための記録。
const browseLog = [];

// The fake filesystem the picker walks.
//
// `Delta` sorts before `alpha` bytewise and after it casefold, so a listing
// rendered in the order it arrives is distinguishable from one re-sorted in
// the client -- the client must not re-sort, because `LoadTargetRequest.index`
// is the *server's* casefold order.
// バイト順と casefold 順で並びが変わる名前を入れてある。クライアント側で
// 並べ替える実装と区別するため（index の基準はサーバ側の順序である）。
const BROWSE_TREE = {
  "/repo": {
    parent: null,
    entries: [
      { name: "data", is_dir: true, size: null, mtime: 1 },
      { name: "Delta", is_dir: true, size: null, mtime: 1 },
      { name: "notes.txt", is_dir: false, size: 12, mtime: 1 },
    ],
  },
  "/repo/data": {
    parent: "/repo",
    entries: [
      { name: "shots", is_dir: true, size: null, mtime: 1 },
      { name: "matrices_linear_gamma.npy", is_dir: false, size: 99532800, mtime: 1 },
      { name: "readme.txt", is_dir: false, size: 5, mtime: 1 },
      { name: "nature_1_sq.png", is_dir: false, size: 2048, mtime: 1 },
      { name: "inverse_lut.npz", is_dir: false, size: 4096, mtime: 1 },
    ],
  },
  "/repo/data/shots": { parent: "/repo/data", entries: [], cap: 2, total: 12800 },
  "/repo/Delta": { parent: "/repo", denied: true, entries: [] },
};
```

`/repo/data/shots` は `entries: []` のままだと `cap`/`total` の計算が空配列に対して
働くので、12800件のうち2件が返る形にするため次の2件を入れる。

```js
  "/repo/data/shots": {
    parent: "/repo/data",
    entries: [
      { name: "IMG_0001.png", is_dir: false, size: 10, mtime: 1 },
      { name: "IMG_0002.png", is_dir: false, size: 10, mtime: 1 },
    ],
    cap: 2, total: 12800,
  },
```

- [ ] **Step 7: `picker` シナリオを書く**

`tests/js/frontend_harness.mjs` の末尾、`if (!checks.length)` の直前に追加する。

```js
// ------------------------------------------------------------------- picker
//
// 参照 ボタンとモーダルのファイラ。ここで見るのは、画面写真からは見えず、
// Python 側からも見えないものだけである。
if (SCENARIO === "picker") {
  await import(pathToFileURL(path.join(STATIC_DIR, "js", "main.js")).href);
  await settled();

  // --- 1. 開く前は隠れている
  check("the picker starts hidden",
    el("picker-backdrop").hidden === true,
    `hidden=${el("picker-backdrop").hidden}`);

  // --- 2. 参照 が開き、欄の値のディレクトリ部分から始まる
  //
  // 出発点が「作業ディレクトリ」に固定されていると、`data/…/matrices…npy` が
  // 入っている欄から開いても毎回ルートに戻る。ファイルそのものを問い合わせると
  // サーバは 400 を返すので、ディレクトリ部分でなければならない。
  el("path-input").value = "/repo/data/matrices_linear_gamma.npy";
  el("btn-browse-matrix").__fire("click");
  await settled();
  check("参照 opens the dialog at the directory of the path in the box",
    el("picker-backdrop").hidden === false && el("picker-cwd").textContent === "/repo/data",
    `hidden=${el("picker-backdrop").hidden} cwd="${el("picker-cwd").textContent}"`);

  // --- 3. 行列の欄は .npy だけを絞って要求する
  //
  // フィルタはサーバに渡さなければ意味が無い。クライアント側で絞る実装は
  // 打ち切り件数を狂わせる（サーバが2000件を切ったあとで絞ることになる）。
  const matrixRequest = browseLog[browseLog.length - 1];
  check("the matrix box asks the server for .npy only",
    matrixRequest.suffixes === ".npy",
    `suffixes="${matrixRequest.suffixes}"`);

  // --- 4. 一覧はサーバの順序のまま描く
  //
  // `Delta` はバイト順では `alpha` の前、casefold 順では後。クライアントが
  // 並べ替え直すと `LoadTargetRequest.index` の基準とずれる。
  await settled();
  el("picker-up").__fire("click");
  await settled();
  check("the listing keeps the order the server sent",
    el("picker-list").children.map((row) => row.children[0].textContent)
      .join(",") === "data/,Delta/,notes.txt",
    el("picker-list").children.map((row) => row.children[0].textContent).join(","));

  // --- 5. ディレクトリのクリックは1回の要求で入る
  const before = browseLog.length;
  el("picker-entry-0").__fire("click");
  await settled();
  check("entering a directory costs exactly one /api/browse",
    browseLog.length === before + 1 && el("picker-cwd").textContent === "/repo/data",
    `requests=${browseLog.length - before} cwd="${el("picker-cwd").textContent}"`);

  // --- 6. ファイルモードでは、ファイルを選ぶまで 選択 が押せない
  //
  // `file` モードにはディレクトリという既定の答えが無い。押せてしまうと
  // 「何が返るのか」を利用者が推測することになる。
  const chooseBeforePick = el("picker-choose").disabled;
  el("picker-entry-1").__fire("click");     // matrices_linear_gamma.npy
  await settled();
  check("選択 is dead until a file is picked, and names exactly what it will return",
    chooseBeforePick === true
      && el("picker-choose").disabled === false
      && el("picker-chosen").textContent === "/repo/data/matrices_linear_gamma.npy",
    `before=${chooseBeforePick} after=${el("picker-choose").disabled} `
      + `chosen="${el("picker-chosen").textContent}"`);

  // --- 7. 確定は欄を書き換え、`change` を発火する
  //
  // 代入だけの実装と区別できない検査にしない。`gen-output-dir` の編集リスナは
  // 上書き合意を失効させるので、イベント無しの代入は「Aについての合意をBに
  // 使う」経路になる。ここで見るのは行列の欄だが、発火は4欄で共通の1箇所である。
  let sawChange = 0;
  el("path-input").addEventListener("change", () => { sawChange += 1; });
  el("picker-choose").__fire("click");
  await settled();
  check("choosing writes the box and fires change on it",
    el("path-input").value === "/repo/data/matrices_linear_gamma.npy"
      && sawChange === 1
      && el("picker-backdrop").hidden === true,
    `value="${el("path-input").value}" change=${sawChange} `
      + `hidden=${el("picker-backdrop").hidden}`);

  // --- 8. 取消は欄に触れない
  el("path-input").value = "untouched";
  el("btn-browse-matrix").__fire("click");
  await settled();
  el("picker-cancel").__fire("click");
  await settled();
  check("cancelling leaves the box exactly as it was",
    el("path-input").value === "untouched" && el("picker-backdrop").hidden === true,
    `value="${el("path-input").value}" hidden=${el("picker-backdrop").hidden}`);

  // --- 9. 拒否はモーダルの中に出て、モーダルは開いたまま
  //
  // ページのステータス行に出すと背景の裏に印字される。見えないディレクトリに
  // ついての拒否は、対処のしようが無い。
  el("path-input").value = "/repo/data/x.npy";
  el("btn-browse-matrix").__fire("click");
  await settled();
  el("picker-up").__fire("click");          // -> /repo
  await settled();
  const cwdBeforeDenial = el("picker-cwd").textContent;
  el("picker-entry-1").__fire("click");     // Delta -> 403
  await settled();
  check("a refusal is shown inside the dialog, which stays where it was",
    el("picker-backdrop").hidden === false
      && /403/.test(el("picker-error").textContent)
      && el("picker-cwd").textContent === cwdBeforeDenial,
    `hidden=${el("picker-backdrop").hidden} error="${el("picker-error").textContent}" `
      + `cwd="${el("picker-cwd").textContent}"`);

  // --- 10. 打ち切りは件数で言う
  //
  // 黙って切ると「これがディレクトリの全部」と読まれる。
  el("picker-cancel").__fire("click");
  await settled();
  el("path-input").value = "/repo/data/shots/x.npy";
  el("btn-browse-matrix").__fire("click");
  await settled();
  check("a truncated listing says how much of it is shown",
    el("picker-count").textContent === "12800件中2件を表示",
    `count="${el("picker-count").textContent}"`);
  el("picker-cancel").__fire("click");
  await settled();
}
```

- [ ] **Step 8: `test_web_frontend_js.py` にシナリオを足す**

`_MUTATIONS` の定義より前、既存のテスト関数群の末尾に追加する。

```python
def test_the_browse_button_fills_the_box_the_way_typing_would() -> None:
    """``参照`` がパス欄を埋め、手打ちと区別が付かない状態にすること。

    ファイラの3つの性質はどれも静かに壊れる。フィルタをサーバへ渡さずに
    クライアント側で絞れば、2000件で切ったあとに絞ることになり件数表示が嘘に
    なる。一覧をクライアントで並べ替え直せば、``LoadTargetRequest.index`` の
    基準である ``target.py::_listing`` の casefold 順とずれ、「3枚目」が別の
    画像を指す。そして確定を代入だけで済ませれば、``gen-output-dir`` の編集
    リスナ（上書き合意を失効させる）を迂回する欄が1つできる。

    拒否の出し先も同じ種類の問題である。ページのステータス行は背景の裏に
    あるので、そこへ出した 403 は読めない。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "picker"))
```

- [ ] **Step 9: 落ちることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_frontend_js.py -q -k browse_button`
Expected: FAIL。まだ実装が無い段階で書いた場合はここで落ちるが、Step 1〜7 を先に
済ませているので **PASS するはずである**。PASS しないなら、落ちたチェック名が
そのまま原因を指している（`_assert_all_passed` は失敗したチェックを全部並べる）。

- [ ] **Step 10: JS のテスト全体が通ることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_frontend_js.py -q`
Expected: PASS。既存の `index.html` 由来のチェック（要素の置き場所、id の一致検査）が
落ちたら、追加した要素の位置か id の綴りが原因である。

- [ ] **Step 11: `_MUTATIONS` に変異を4つ足す**

`test_web_frontend_js.py` の `_MUTATIONS` リストの末尾に追加する。

```python
    # ---- the picker's four silent failure modes
    #
    # None of these throws, and none of them is visible in a screenshot of a
    # working dialog. ここに挙げた4つはどれも例外を出さず、動いている
    # ダイアログの画面写真からも見えない。
    #
    # The filter, moved to the client: the server cuts at MAX_ENTRIES and *then*
    # the client would drop the non-matching files, so a directory of 12800
    # captures shows "12800件中2件を表示" over a list that is neither.
    # フィルタをクライアント側へ移す。サーバが2000件で切ったあとに絞ることに
    # なり、件数表示が実際の一覧と対応しなくなる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            '  if (session.suffixes.length) params.set("suffixes", session.suffixes.join(","));',
            "",
        )],
        ("asks the server for .npy only",),
        id="picker_filters_on_the_client_instead_of_the_server",
    ),
    # The listing, re-sorted locally. `LoadTargetRequest.index` is an index into
    # the *server's* casefold order (`target.py::_listing`), so a client-side
    # sort makes "the third image" name a different file in the box than in the
    # request. Bytewise sort puts `Delta` before `data`.
    # クライアント側で並べ替え直す。index の基準はサーバ側の casefold 順なので、
    # 「3枚目」が欄と要求とで別のファイルを指すようになる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            "  body.entries.forEach((entry, i) => {",
            "  [...body.entries].sort((a, b) => (a.name < b.name ? -1 : 1)).forEach((entry, i) => {",
        )],
        ("keeps the order the server sent",),
        id="picker_re_sorts_the_listing_on_the_client",
    ),
    # The confirmation, assigned instead of dispatched. `gen-output-dir`'s two
    # edit listeners retire the 上書き agreement, and this is the one way into
    # that box that would bypass them.
    # 確定を代入で済ませる。`gen-output-dir` の編集リスナ2つを迂回する唯一の
    # 経路になる。
    pytest.param(
        "picker",
        [(
            "js/main.js",
            '        box.dispatchEvent(new Event("input", { bubbles: true }));\n'
            '        box.dispatchEvent(new Event("change", { bubbles: true }));',
            "",
        )],
        ("fires change on it",),
        id="picker_sets_the_box_without_firing_its_events",
    ),
    # The cut, made silent. The list is right; the sentence over it is the lie.
    # 切り詰めを黙って行う。一覧そのものは正しく、その上の文だけが嘘になる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            "  $(\"picker-count\").textContent = body.truncated\n"
            "    ? `${body.total}件中${body.entries.length}件を表示`\n"
            "    : `${body.total}件`;",
            "  $(\"picker-count\").textContent = `${body.entries.length}件`;",
        )],
        ("says how much of it is shown",),
        id="picker_truncates_the_listing_silently",
    ),
```

- [ ] **Step 12: 変異が効いていることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_frontend_js.py -q`
Expected: PASS。`_MUTATIONS` を回すテストは、各変異で**指定したチェックだけ**が
落ちることを検証する。想定外のチェックが落ちたらそれ自体が発見であり、
チェックの範囲か変異の粒度のどちらかを直す。

- [ ] **Step 13: コミット**

```bash
git add packages/web/yoshidacomp_web/static/js/filepicker.js \
        packages/web/yoshidacomp_web/static/index.html \
        packages/web/yoshidacomp_web/static/style.css \
        packages/web/yoshidacomp_web/static/js/main.js \
        tests/js/frontend_harness.mjs tests/test_web_frontend_js.py
git commit -F - <<'EOF'
[add] モーダルのファイラを追加し、行列のパス欄に参照ボタンを付ける

- packages/web/yoshidacomp_web/static/js/filepicker.js
  - 変更点: 新規追加。pickPath() が GET /api/browse を辿るモーダルを開き、
    選ばれたパス（取消なら null）を返す。file / dir / file_or_dir の3モード。
  - 理由/意図: input type="file" はサーバ側のパスを返さず、バックエンドは
    パスを受け取って自分で読む設計であるため、一覧はサーバから取る必要がある。
  - 影響/注意: 読むだけ。作成・削除・改名はしない。拡張子はサーバの
    /api/browse/suffixes から取り、フロントには書かない。
- packages/web/yoshidacomp_web/static/index.html
  - 変更点: btn-browse-matrix と picker-* のモーダル一式を追加。
  - 理由/意図: モーダルを <main> の外に置くのは、サイドバー内では
    overflow-y: auto に切り取られるため。
- packages/web/yoshidacomp_web/static/style.css
  - 変更点: .picker-* の規則を追加。
- packages/web/yoshidacomp_web/static/js/main.js
  - 変更点: BROWSE_FIELDS 表と wireBrowseButtons() を追加。確定時に
    input と change を dispatchEvent する。
  - 理由/意図: gen-output-dir の編集リスナは上書き合意を失効させるので、
    代入だけではリスナを迂回する欄ができる。
  - 影響/注意: 参照ボタンの結線は行列の欄のみ。残る3欄は次のコミット。
- tests/js/frontend_harness.mjs
  - 変更点: DOMスタブに dispatchEvent と globalThis.Event を追加。
    /api/browse の偽サーバと picker シナリオ（10件）を追加。
- tests/test_web_frontend_js.py
  - 変更点: picker シナリオを起動するテストと、_MUTATIONS に変異4件を追加。
  - 影響/注意: pytest tests/test_web_frontend_js.py -q で緑を確認済み。
    4つの変異それぞれで、指定したチェックだけが落ちることを確認済み。
EOF
```

---

### Task 4: 残り3欄への適用（`dir` モードと新規フォルダ名）

**Files:**
- Modify: `packages/web/yoshidacomp_web/static/index.html`
- Modify: `tests/js/frontend_harness.mjs`
- Modify: `tests/test_web_frontend_js.py`

**Interfaces:**
- Consumes: `pickPath` / `suffixesFor` / `BROWSE_FIELDS`（Task 3。表には4欄すべてが既に入っている）
- Produces: `index.html` に `btn-browse-target` / `btn-browse-lut` / `btn-browse-output`

- [ ] **Step 1: 3つの `参照` ボタンを `index.html` に足す**

`index.html:146-149`（ターゲット）を次に置き換える。

```html
        <div class="group inline">
          <button id="btn-browse-target">参照</button>
          <button id="btn-load-target">読み込み</button>
          <span id="target-info" class="muted mono">未読み込み</span>
        </div>
```

`index.html:158-161`（逆LUT）を次に置き換える。

```html
        <div class="group inline">
          <button id="btn-browse-lut">参照</button>
          <button id="btn-load-lut">読み込み</button>
          <span id="lut-info" class="muted mono">未読み込み</span>
        </div>
```

`index.html:217-220`（出力先）を次に置き換える。

```html
        <div class="group inline">
          <button id="btn-browse-output">参照</button>
          <button id="btn-generate" class="primary">生成</button>
          <span id="gen-info" class="muted mono">未生成</span>
        </div>
```

- [ ] **Step 2: `picker` シナリオに残り3欄のチェックを足す**

`frontend_harness.mjs` の `picker` シナリオ末尾（`if (SCENARIO === "picker") { ... }` の
閉じ括弧の直前）に追加する。

```js
  // --- 11. ターゲットの欄は画像とディレクトリの両方を返せる
  //
  // `target-path` は「画像1枚」と「画像を並べたディレクトリ」のどちらも取る
  // 唯一の欄で、`POST /api/target` はその区別を自分で付ける。ファイラ側で
  // どちらかに寄せると、片方が選べない欄になる。
  el("target-path").value = "";
  el("btn-browse-target").__fire("click");
  await settled();
  const targetRequest = browseLog[browseLog.length - 1];
  el("picker-entry-0").__fire("click");            // data/
  await settled();
  const dirIsAnswerable = el("picker-choose").disabled === false
    && el("picker-chosen").textContent === "/repo/data";
  el("picker-entry-3").__fire("click");            // nature_1_sq.png
  await settled();
  check("the target box can answer with an image or with a directory",
    targetRequest.suffixes === ".png,.jpg"
      && dirIsAnswerable
      && el("picker-chosen").textContent === "/repo/data/nature_1_sq.png",
    `suffixes="${targetRequest.suffixes}" dirAnswerable=${dirIsAnswerable} `
      + `chosen="${el("picker-chosen").textContent}"`);
  el("picker-choose").__fire("click");
  await settled();
  check("the target box receives the picked image",
    el("target-path").value === "/repo/data/nature_1_sq.png",
    `value="${el("target-path").value}"`);

  // --- 12. LUT の欄は .npz を絞る
  el("lut-path").value = "";
  el("btn-browse-lut").__fire("click");
  await settled();
  check("the lut box asks the server for .npz only",
    browseLog[browseLog.length - 1].suffixes === ".npz",
    `suffixes="${browseLog[browseLog.length - 1].suffixes}"`);
  el("picker-cancel").__fire("click");
  await settled();

  // --- 13. 出力先はディレクトリモードで、新規フォルダ名を連結する
  //
  // 書き込み先なので「まだ存在しないディレクトリ」を指せなければならない。
  // ファイラは存在するものしか一覧できないので、現在地＋入力名という形になる。
  // **ここで作りはしない。** 作成は POST /api/generate の仕事のままである。
  el("gen-output-dir").value = "";
  el("btn-browse-output").__fire("click");
  await settled();
  const newdirShown = el("picker-newdir").hidden === false;
  const emptyMeansHere = el("picker-chosen").textContent === "/repo";
  el("picker-entry-0").__fire("click");            // data/
  await settled();
  el("picker-newdir-name").value = "proj_20260805";
  el("picker-newdir-name").__fire("input");
  await settled();
  check("the output box picks a directory that need not exist yet",
    newdirShown && emptyMeansHere
      && el("picker-chosen").textContent === "/repo/data/proj_20260805",
    `newdirShown=${newdirShown} emptyMeansHere=${emptyMeansHere} `
      + `chosen="${el("picker-chosen").textContent}"`);

  // --- 14. その欄では 上書き合意 が失効する
  //
  // 確定はイベントを発火するので、`gen-output-dir` の編集リスナに届く。
  // 届かなければ、A というディレクトリについて取った上書きの合意が B に
  // 使われうる（押下時にも再確認しているが、リスナを迂回する経路を残さない）。
  let sawOutputChange = 0;
  el("gen-output-dir").addEventListener("change", () => { sawOutputChange += 1; });
  el("picker-choose").__fire("click");
  await settled();
  check("the output box is written and its edit listeners are reached",
    el("gen-output-dir").value === "/repo/data/proj_20260805" && sawOutputChange === 1,
    `value="${el("gen-output-dir").value}" change=${sawOutputChange}`);

  // --- 15. 新規フォルダ名は次に開いたとき残らない
  //
  // 残ると、別の出力先を選びに来た押下が前回の名前を連れてくる。
  el("btn-browse-output").__fire("click");
  await settled();
  check("the new-folder name does not survive into the next dialog",
    el("picker-newdir-name").value === "" && el("picker-chosen").textContent === "/repo/data",
    `name="${el("picker-newdir-name").value}" chosen="${el("picker-chosen").textContent}"`);
  el("picker-cancel").__fire("click");
  await settled();

  // --- 16. ファイル用の欄には新規フォルダ名を出さない
  //
  // 読む側の欄で「まだ無いファイル」を名乗れてしまうと、必ず 404 になる
  // パスを組み立てる道具を渡すことになる。
  el("btn-browse-matrix").__fire("click");
  await settled();
  check("the new-folder box is absent where the path must already exist",
    el("picker-newdir").hidden === true,
    `hidden=${el("picker-newdir").hidden}`);
  el("picker-cancel").__fire("click");
  await settled();
```

- [ ] **Step 3: 通ることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_frontend_js.py -q -k browse_button`
Expected: PASS（`picker` シナリオが16件）

- [ ] **Step 4: `_MUTATIONS` に変異を3つ足す**

`test_web_frontend_js.py` の `_MUTATIONS` の末尾に追加する。

```python
    # ---- the three modes collapsed into one, one at a time
    #
    # The modes are one function (`chosenPath`) and each collapse leaves a
    # dialog that opens, lists and returns a plausible string.
    # 3つのモードは1つの関数に閉じており、どれに潰しても「開いて一覧して
    # それらしい文字列を返すダイアログ」のままである。
    #
    # `dir` folded into "the current directory, always": the new-folder name is
    # ignored, so 出力先 can only ever name a directory that already exists --
    # and `generate.py` refuses a non-empty one. The box becomes unable to
    # express its own ordinary case.
    # 新規フォルダ名を無視する。出力先は既存ディレクトリしか名乗れなくなり、
    # generate.py は空でないディレクトリを拒むので、通常の使い方ができなくなる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            "    const name = $(\"picker-newdir-name\").value.trim();\n"
            "    return name ? `${session.cwd}/${name}` : session.cwd;",
            "    return session.cwd;",
        )],
        ("picks a directory that need not exist yet",
         "the output box is written and its edit listeners are reached"),
        id="picker_dir_mode_ignores_the_new_folder_name",
    ),
    # `file_or_dir` narrowed to `file`: the target box can no longer answer with
    # a directory, and a directory of images is half of what that endpoint takes
    # (`LoadTargetRequest.index` exists for exactly that case).
    # ターゲットの欄がディレクトリを返せなくなる。画像を並べたディレクトリは
    # このエンドポイントの用途の半分である（index はそのために在る）。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            '    return session.mode === "file_or_dir" ? session.cwd : "";',
            '    return "";',
        )],
        ("can answer with an image or with a directory",),
        id="picker_target_mode_refuses_a_directory",
    ),
    # The new-folder box, shown everywhere: a reading box then offers to name a
    # file that does not exist, which can only ever be a 404.
    # 読む側の欄にも新規フォルダ名を出す。必ず404になるパスを組み立てる道具に
    # なる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            '    $("picker-newdir").hidden = mode !== "dir";',
            '    $("picker-newdir").hidden = false;',
        )],
        ("absent where the path must already exist",),
        id="picker_offers_a_new_folder_name_on_a_reading_box",
    ),
```

- [ ] **Step 5: 変異が効いていることを確認する**

Run: `uv run --no-sync python -m pytest tests/test_web_frontend_js.py -q`
Expected: PASS

- [ ] **Step 6: 全体が緑であることを確認する**

Run: `uv run --no-sync python -m pytest -q`
Expected: PASS。約3分40秒。

- [ ] **Step 7: コミット**

```bash
git add packages/web/yoshidacomp_web/static/index.html \
        tests/js/frontend_harness.mjs tests/test_web_frontend_js.py
git commit -F - <<'EOF'
[add] 残る3つのパス欄にも参照ボタンを付ける

- packages/web/yoshidacomp_web/static/index.html
  - 変更点: btn-browse-target / btn-browse-lut / btn-browse-output を追加。
  - 理由/意図: 挙動の分岐は main.js の BROWSE_FIELDS 表に既にあり、ここでは
    ボタンを置くだけで済む。
  - 影響/注意: 出力先だけがディレクトリモードで、新規フォルダ名の欄が出る。
    ファイラはディレクトリを作らない（作成は POST /api/generate のまま）。
- tests/js/frontend_harness.mjs
  - 変更点: picker シナリオに6件追加（計16件）。ターゲットが画像とディレクトリ
    の両方を返せること、LUT の .npz 絞り込み、出力先の新規フォルダ名の連結、
    上書き合意の失効に届くこと、名前が次回に残らないこと、読む側の欄には
    新規フォルダ名を出さないこと。
- tests/test_web_frontend_js.py
  - 変更点: _MUTATIONS に変異3件を追加（3つのモードをそれぞれ潰す）。
  - 影響/注意: pytest -q の全体が緑であることを確認済み。3つの変異それぞれで、
    指定したチェックだけが落ちることを確認済み。
EOF
```

---

### Task 5: 実機での確認と、打ち切り上限の実測

仕様書8節の未解決事項を閉じる。`MAX_ENTRIES = 2000` は実測に基づかない値だと
書いてあるので、実際のディレクトリで測る。

**Files:**
- Modify: `packages/web/yoshidacomp_web/browse.py`（実測の結果しだい）
- Modify: `docs/superpowers/specs/2026-08-05-file-picker-design.md`（8節を実測で置き換える）

- [ ] **Step 1: 実際のディレクトリの件数を数える**

```bash
uv run --no-sync python -c "import os; [print(len(os.listdir(d)), d) for d in ['data', 'data/graycode_recalibration_20260805_window4', 'data/linear_pipeline_1080', 'data/iso160_nonhdr_20260805_geom']]"
```

- [ ] **Step 2: 一番大きいディレクトリで応答時間を測る**

上で最大だったディレクトリを `<biggest>` として測る。

```bash
uv run --no-sync python -c "
import time, sys
sys.path.insert(0, 'packages/web')
from pathlib import Path
from yoshidacomp_web.browse import list_directory
t = time.perf_counter()
listing = list_directory(Path('<biggest>'))
print(f'{listing.total} entries, {(time.perf_counter()-t)*1000:.0f} ms, truncated={listing.truncated}')
"
```

- [ ] **Step 3: ビューアを起動する**

**ポート 8123。8000 は別プロセスが占有している。**

```bash
uv run --no-sync python -m uvicorn yoshidacomp_web.app:app --host 127.0.0.1 --port 8123
```

`http://127.0.0.1:8123/` を開く。

- [ ] **Step 4: 4つの `参照` を実際に押して確認する**

各項目を目で見て確かめる。**画面写真を1枚撮り、空でないことを確認する。**

1. ツールバーの `参照` → `data/` に入り `.npy` だけが並ぶ → 選ぶと欄が埋まる → `読み込み` が通る
2. ターゲットの `参照` → 画像1枚を選べる、ディレクトリも選べる → どちらも `読み込み` が通る
3. 逆LUT の `参照` → `.npz` だけが並ぶ → `読み込み` が通り曲線が出る
4. 出力先の `参照` → 新規フォルダ名を入れる → 欄に連結されたパスが入る
5. `C:/` に上がり、権限の無いディレクトリ（`C:/System Volume Information` など）を
   開いて **403 がモーダル内に出て、モーダルが閉じないこと**
6. 手打ちが今まで通りできること

- [ ] **Step 5: 実測に合わせて `MAX_ENTRIES` を決める**

Step 1〜2 の結果が `MAX_ENTRIES` を大きく下回るなら 2000 のままでよい。上回る、
または応答が 300ms を超えるなら値を見直し、`browse.py` の `MAX_ENTRIES` の
コメントを**実測値に書き換える**（「実測に基づかない暫定値」という記述を消す）。

- [ ] **Step 6: 仕様書の8節を実測で置き換える**

`docs/superpowers/specs/2026-08-05-file-picker-design.md` の「## 8. 未解決」を
実測した件数・応答時間・最終的な `MAX_ENTRIES` に書き換える。閉じられなかった
項目が残るならそれだけを残す。

- [ ] **Step 7: 全体が緑であることを最後にもう一度確認する**

Run: `uv run --no-sync python -m pytest -q`
Expected: PASS

- [ ] **Step 8: コミット**

```bash
git add packages/web/yoshidacomp_web/browse.py \
        docs/superpowers/specs/2026-08-05-file-picker-design.md
git commit -F - <<'EOF'
[update] 一覧の打ち切り上限を実測に基づいて確定する

- packages/web/yoshidacomp_web/browse.py
  - 変更点: MAX_ENTRIES を実測に基づく値に確定し、コメントから「暫定値」の
    記述を外した。
  - 理由/意図: 仕様書8節の未解決事項。data/ の実ディレクトリで件数と
    応答時間を測った。
  - 影響/注意: 実測値と応答時間はコメントとコミット本文に記載（<ここに実数>）。
- docs/superpowers/specs/2026-08-05-file-picker-design.md
  - 変更点: 8節を実測結果で置き換えた。
  - 影響/注意: ブラウザでの4欄の動作確認を実施済み（403 がモーダル内に出て
    閉じないことを含む）。pytest -q の全体が緑であることを確認済み。
EOF
```

---

## 自己レビュー結果

**仕様との対応**

| 仕様の節 | 実装するタスク |
| --- | --- |
| 2. スコープ（対象外の4項目） | Task 1 と Task 3 のコード内コメントで明示。作成・削除・改名の口を作らない |
| 3. パスを渡せない理由 | Task 3 の `filepicker.js` 冒頭コメント |
| 4. 閲覧範囲と Host 検査 | Task 2 |
| 5. 閲覧API（並び順・フィルタ・打ち切り・404/400/403・`os.listdir` の制約） | Task 1、Task 2 |
| 6. フロントエンド（4欄の表・構成・操作・確定時の副作用・書き込む形式・新規フォルダ・エラー表示） | Task 3、Task 4 |
| 7. テスト（Python / JS / `_MUTATIONS`） | 各タスクに分散。`_MUTATIONS` は Task 3 で4件、Task 4 で3件 |
| 8. 未解決（2000件の根拠） | Task 5 |

**仕様と計画が食い違う1点**

仕様6節は「CWD 配下なら相対パス、外なら絶対パス」と書いているが、計画は
**常にサーバが返した絶対パス**を欄へ書く。マシン全体を閲覧できるようにした以上、
相対/絶対の切り替えは「どちらの形で表示されているか」を利用者に推測させる規則に
なる。`#picker-chosen` が押す前に返る文字列そのものを出しているので、それと
欄の中身が一致するほうが読みやすい。**Task 5 の Step 4 で実際に見て、読みにくければ
そのとき相対表示を足す。**

**型と名前の一貫性**

`list_directory` / `Listing` / `Entry` / `SUFFIX_TABLE` / `MAX_ENTRIES`（Task 1）は
Task 2 の `routers/browse.py` から同じ名前で参照している。`pickPath` /
`suffixesFor`（Task 3）は Task 4 でも同じ名前。`BROWSE_FIELDS` は Task 3 で4欄すべてを
定義し、Task 4 はボタンを置くだけ ── 表を2回書かない。
