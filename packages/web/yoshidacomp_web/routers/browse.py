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
