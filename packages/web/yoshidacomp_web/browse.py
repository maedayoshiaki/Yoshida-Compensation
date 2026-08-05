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

# 1回の応答に載せる最大件数。実測(2026-08-06、仕様書8節)。数値は各々そのまま
# 計測した範囲であり、別ディレクトリの値を混ぜていない。
#
# 実利用域: ``data/`` 直下(57件)は2.7〜4ms、``data/`` 配下の実測最大
# ``data/debug``(139件)は6〜9ms(いずれも複数回の計測)。
# 2000件に対して十分な余裕がある。
#
# 閲覧範囲はマシン全体（4節）なので外側も測った。``C:/Windows/System32``
# （4880件）は繰り返し計測で146〜170ms(6回)と300ms未満で安定する。
# ``C:/Windows/WinSxS``（29537件）は繰り返し計測でも1174〜1274ms(6回)と
# 一貫して300msを超える。どちらも**最初の1回**は遅く、System32が738ms、
# WinSxSが2400msだった。これを「冷キャッシュだから」とは書かない ── キャッシュ
# を意図的に破棄する手段を持たず（仕様書8節）、遅かった理由を確かめていない。
# 観測したのは「続けて測ったうちの最初の1回が遅かった」ことだけであり、
# どちらも1回のみで再現を試みていない。300ms超えという結論は複数回再現した値
# （WinSxS）に基づく。System32を「超える」側に含めない。
#
# ``MAX_ENTRIES`` を下げてもこれは速くならない ── ``list_directory`` は
# ``sorted()`` の後、全エントリに ``is_dir()``/``stat()`` を呼んでから
# ``ordered[:MAX_ENTRIES]`` で最後に切り詰めるため、時間のかかる処理（全件の
# stat）は打ち切り件数と無関係に走る。実測で確認済み: 同じ ``System32`` を
# ``MAX_ENTRIES=2000/200/50`` で計測し 152/146/154ms と差がなかった。
#
# 実利用域には十分な余裕があり、かつこの値自体は応答時間のボトルネックでは
# ないため、2000のままとする。巨大ディレクトリの高速化（全件 stat をやめる等）
# は ``list_directory`` の構造を変える話で、この値の見直しでは達成できない。
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
