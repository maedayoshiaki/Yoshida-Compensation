"""Locate the vendored GrayCode sources and put them on ``sys.path``.

GrayCode (``graycode``) is a git submodule checked out at
``external/GrayCode/src``, not an installed distribution, so it is
importable only after that directory has been added to ``sys.path``.
Every consumer of :mod:`yoshidacomp.geometry.warp` therefore has to
bootstrap the same path; this module is the single place that knows how.

GrayCode (``graycode``) は ``external/GrayCode/src`` に配置された git
submodule であり、インストール済みディストリビューションではない。その
ため当該ディレクトリを ``sys.path`` に追加した後でしか import できない。
:mod:`yoshidacomp.geometry.warp` の利用者はいずれもこのブートストラップを
必要とするので、その方法を知る唯一の場所を本モジュールに集約する。

This module must **not** import ``graycode`` itself: ``warp.py``
deliberately defers all GrayCode imports into its functions so that
``import yoshidacomp.geometry.warp`` stays GrayCode-free, and importing it
here would undo that.
本モジュールは ``graycode`` 自体を import して**はならない**。``warp.py``
は ``import yoshidacomp.geometry.warp`` を GrayCode 非依存に保つために
GrayCode の import を関数内へ意図的に遅延させており、ここで import すると
それを台無しにする。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Path of the vendored sources relative to the repository root.
# リポジトリルートから見た vendored ソースの相対パス。
_RELATIVE_SRC = Path("external") / "GrayCode" / "src"

#: Environment variable that overrides the search entirely. Set it when the
#: repository layout cannot be discovered -- e.g. ``yoshidacomp`` installed
#: as a wheel with the submodule checked out elsewhere.
#: 探索を完全に上書きする環境変数。リポジトリ配置から発見できない場合
#: （wheel でインストールし submodule を別の場所に置いた場合など）に使う。
GRAYCODE_SRC_ENV_VAR = "YOSHIDACOMP_GRAYCODE_SRC"


def _candidate_start_dirs() -> list[Path]:
    """Directories whose ancestors are searched for the submodule.

    submodule を探すために祖先を辿る起点ディレクトリ。

    Two starting points, because neither alone covers every layout:
    this file's own location works for the editable workspace
    (``packages/core/yoshidacomp/geometry`` sits four levels below the
    repository root), and the working directory works when ``yoshidacomp``
    is installed into a virtualenv but a repository checkout is what the
    process is actually running against.

    起点は2つ。単独ではすべての配置を賄えないため。本ファイルの位置は
    editable なワークスペース配置（``packages/core/yoshidacomp/geometry``
    はリポジトリルートの4階層下）で機能し、カレントディレクトリは
    ``yoshidacomp`` が仮想環境へインストールされていて実際の実行対象が
    リポジトリのチェックアウトである場合に機能する。
    """
    return [Path(__file__).resolve().parent, Path.cwd().resolve()]


def find_graycode_src() -> Path:
    """Return the directory that makes ``import graycode`` work.

    ``import graycode`` を可能にするディレクトリを返す。

    Returns:
        Absolute path to ``external/GrayCode/src``.
        ``external/GrayCode/src`` への絶対パス。

    Raises:
        FileNotFoundError: If the directory cannot be located. The message
            names the relative path, every root that was searched, and the
            override environment variable -- failing here is far easier to
            act on than a later ``ModuleNotFoundError: No module named
            'graycode'`` raised from an unrelated call site.
            ディレクトリが見つからない場合。相対パス・探索したすべての
            ルート・上書き用環境変数をメッセージに含める。無関係な箇所から
            後で ``ModuleNotFoundError: No module named 'graycode'`` が出る
            よりも、ここで失敗した方が対処しやすい。
    """
    override = os.environ.get(GRAYCODE_SRC_ENV_VAR, "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"{GRAYCODE_SRC_ENV_VAR} is set to '{override}', which is not "
                "an existing directory. It must point at the GrayCode "
                f"'{_RELATIVE_SRC.as_posix()}' directory (the one containing "
                "the 'graycode' package)."
            )
        return candidate

    searched: list[Path] = []
    for start in _candidate_start_dirs():
        for root in (start, *start.parents):
            candidate = root / _RELATIVE_SRC
            searched.append(candidate)
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        f"Could not locate the vendored GrayCode sources "
        f"('{_RELATIVE_SRC.as_posix()}'). GrayCode is a git submodule, not an "
        "installed package, so it must be checked out "
        "(`git submodule update --init`) and its 'src' directory placed on "
        f"sys.path. Searched {len(searched)} location(s): "
        + ", ".join(f"'{path}'" for path in searched)
        + f". Set {GRAYCODE_SRC_ENV_VAR} to override the search."
    )


def ensure_on_syspath() -> Path:
    """Put the vendored GrayCode sources on ``sys.path`` and return them.

    vendored GrayCode ソースを ``sys.path`` に載せ、そのパスを返す。

    Idempotent: repeated calls -- which happen whenever more than one of the
    bootstrapping modules is loaded into the same interpreter -- neither
    duplicate the entry nor reorder ``sys.path``.

    冪等。ブートストラップを行うモジュールが同一インタプリタに複数読み込ま
    れると本関数は複数回呼ばれるが、エントリの重複も ``sys.path`` の並べ替え
    も起こさない。

    Returns:
        Absolute path to ``external/GrayCode/src``.
        ``external/GrayCode/src`` への絶対パス。
    """
    src = find_graycode_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
