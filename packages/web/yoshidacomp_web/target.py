"""Target image loading (path-only, no upload).

このGUIはオフライン専用でローカルの ``data/`` を直接読む前提であり、
アップロード経路（一時ファイルの寿命管理）は使われないため設けない。パスは
画像ファイル1枚、または画像を並べたディレクトリのどちらかを指す。
``decode_target`` はここでは呼ばない -- uint8 の生画素を保持するだけで、
線形化（``mode``/``gamma``）は補償を計算する側（計画B3 Task 3の
``compensation`` ビュー）が選ぶパラメータであり、保存時点では未定だから。

読み込みは2段に分かれている。``resolve_target`` がパスと ``index`` を検証して
**どのファイルを読むか**まで決め、``load_resolved`` がその1枚だけを復号する。
分けてあるのは、ディレクトリの一覧から別の画像を選び直すのが
``POST /api/target`` の投げ直しであり、1段だったころはそのたびに
``load_target_images`` がディレクトリ内の**全画像**を復号していたためである
（実測: 3枚のディレクトリを一巡すると ``cv2.imread`` が9回。いまは3回）。
``TargetSource.identity`` はその結果を店（``store.TargetStore``）が
使い回せるかどうかの鍵で、これが同じなら復号そのものを省ける。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from yoshidacomp.compensation.pipeline import SUPPORTED_TARGET_IMAGE_SUFFIXES


@dataclass(frozen=True)
class TargetContext:
    """A loaded target image plus what the viewer needs about its source.

    Attributes:
        image: The *currently selected* image, shape ``(H, W, 3)`` uint8, RGB.
            選択中の画像。
        path: The path that was posted -- a file or a directory. 元パス。
        is_directory: Whether ``path`` was a directory of images.
        names: Image file names in the deterministic (casefold-sorted) order,
            the same order ``pipeline.load_target_images`` reports. For a
            single-file ``path`` this is ``[path.name]``. ディレクトリの
            画像名一覧（casefold順）。単一ファイルの場合は ``[path.name]``。
        index: Which entry of ``names`` ``image`` corresponds to. Always
            ``0`` for a single-file ``path``. 選択中の index。
    """

    image: np.ndarray
    path: Path
    is_directory: bool
    names: list[str]
    index: int


@dataclass(frozen=True)
class TargetSource:
    """What a ``path``/``index`` pair resolves to, before any pixel is read.

    Attributes:
        file: The single image file to decode. 復号する1枚。
        path: The path that was posted (``file`` itself, or its directory).
        is_directory: Whether ``path`` was a directory of images.
        names: The listing, casefold-sorted. 一覧。
        index: Which entry of ``names`` is selected.
        identity: A hashable value that changes if and only if decoding this
            source could produce a different ``TargetContext``: the resolved
            path, the index, the listing, and the selected file's size and
            modification time. Used as the store's cache key, so re-selecting
            an image already loaded costs neither a decode nor a second
            retained copy of it.
            復号結果が変わりうるときにだけ変わる鍵。解決済みパス・index・
            一覧・選択中ファイルのサイズと更新時刻から作る。ディスク上で
            差し替えられたファイルは別物として扱われる。
    """

    file: Path
    path: Path
    is_directory: bool
    names: list[str]
    index: int
    identity: tuple


def _listing(directory: Path) -> list[str]:
    """Image file names in *directory*, casefold-sorted.

    ``os.listdir`` is called through the module rather than imported by name
    so that a test which needs genuinely unsorted input can intercept it --
    on NTFS the filesystem itself returns entries in name order, which is what
    made the original ordering test unable to fail (task-2-review.md
    Important 1). Same reason ``pipeline.load_target_images`` is written this
    way.

    Only the *names* are read here. ``load_target_images`` decodes every file
    to build the same list and additionally drops the ones ``cv2.imread``
    cannot read; this does not, so an unreadable file stays in the listing and
    says so when it is selected ("could not read ... as an image") instead of
    disappearing from a directory the user can see in Explorer. Silently
    omitting an entry is the failure this repository keeps closing elsewhere.
    """
    return [
        name
        for name in sorted(os.listdir(directory), key=str.casefold)
        if name.lower().endswith(SUPPORTED_TARGET_IMAGE_SUFFIXES)
    ]


def _identity(file: Path, path: Path, index: int, names: list[str]) -> tuple:
    """The cache key for a resolved source. See ``TargetSource.identity``."""
    stat = file.stat()
    return (
        str(path.resolve()),
        index,
        tuple(names),
        stat.st_size,
        stat.st_mtime_ns,
    )


def _load_single_image(path: Path) -> np.ndarray:
    """Read one image file as ``(H, W, 3)`` uint8 RGB.

    Raises:
        ValueError: the extension is not one of
            ``SUPPORTED_TARGET_IMAGE_SUFFIXES``, or the file exists but
            ``cv2.imread`` cannot decode it (corrupt / not actually an
            image). Both cases are routine mistakes (a pasted path with the
            wrong extension, or a text file), not exceptional.
    """
    if path.suffix.lower() not in SUPPORTED_TARGET_IMAGE_SUFFIXES:
        raise ValueError(
            f"not a supported image file: {path} -- expected one of "
            f"{SUPPORTED_TARGET_IMAGE_SUFFIXES}"
        )
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read {path} as an image (corrupt or not an image)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resolve_target(path: Path, index: int = 0) -> TargetSource:
    """Decide which single file ``path``/``index`` names, without reading it.

    Every refusal about the *request* is raised here -- ``load_resolved``
    below cannot reject anything the caller chose, only what the file turns
    out to contain -- so a caller may use this to look up an already loaded
    target and still get the same errors for a bad request.

    ディレクトリの場合は一覧だけを作り、``index`` 番目のファイルを選ぶ。
    単一ファイルには index が1つ（0）しか存在しないため、0 以外を渡すと
    範囲外のディレクトリindexと同じ扱いで拒否する（無視はしない -- 呼び出し
    側の状態不整合をサイレントに飲み込まないため）。

    Raises:
        FileNotFoundError: ``path`` does not exist. 存在しないパス。
        ValueError: ``path`` exists but cannot become a target image --
            an empty directory, an out-of-range ``index`` (for a directory:
            outside ``[0, len(entries))``; for a single file: anything other
            than ``0``, since a single file has exactly one selectable
            entry), or a path that is neither a file nor a directory. The
            message always names ``path`` since the router hands it straight
            to the user.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.is_dir():
        names = _listing(path)
        if not names:
            raise ValueError(
                f"no supported images ({', '.join(SUPPORTED_TARGET_IMAGE_SUFFIXES)}) "
                f"found in {path}"
            )
        if not (0 <= index < len(names)):
            raise ValueError(
                f"index {index} is out of range for {path} "
                f"({len(names)} image(s), expected 0 <= index < {len(names)})"
            )
        file = path / names[index]
        return TargetSource(
            file=file,
            path=path,
            is_directory=True,
            names=names,
            index=index,
            identity=_identity(file, path, index, names),
        )

    if not path.is_file():
        raise ValueError(
            f"not a file or directory: {path} -- expected an image file or a "
            "directory of images"
        )
    if index != 0:
        raise ValueError(
            f"index {index} is invalid for {path}: a single image file has "
            "only index 0"
        )
    names = [path.name]
    return TargetSource(
        file=path,
        path=path,
        is_directory=False,
        names=names,
        index=0,
        identity=_identity(path, path, 0, names),
    )


def load_resolved(source: TargetSource) -> TargetContext:
    """Decode the one file *source* names.

    Raises:
        ValueError: the extension is not supported, or ``cv2.imread`` cannot
            decode the file. 拡張子非対応、またはデコード不能。
    """
    return TargetContext(
        image=_load_single_image(source.file),
        path=source.path,
        is_directory=source.is_directory,
        names=source.names,
        index=source.index,
    )
