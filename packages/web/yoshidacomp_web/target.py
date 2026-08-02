"""Target image loading (path-only, no upload).

このGUIはオフライン専用でローカルの ``data/`` を直接読む前提であり、
アップロード経路（一時ファイルの寿命管理）は使われないため設けない。パスは
画像ファイル1枚、または画像を並べたディレクトリのどちらかを指す。

読み込み自体は ``yoshidacomp.compensation.pipeline.load_target_images`` に
委譲する（ディレクトリの場合）。単一ファイルはその関数が対象にしていない
ため、同じ読み方（``cv2.imread`` → BGR→RGB）をここで行う。
``decode_target`` はここでは呼ばない -- uint8 の生画素を保持するだけで、
線形化（``mode``/``gamma``）は補償を計算する側（計画B3 Task 3の
``compensation`` ビュー）が選ぶパラメータであり、保存時点では未定だから。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from yoshidacomp.compensation.pipeline import (
    SUPPORTED_TARGET_IMAGE_SUFFIXES,
    load_target_images,
)


@dataclass(frozen=True)
class TargetContext:
    """A loaded target image plus what the viewer needs about its source.

    Attributes:
        image: The *currently selected* image, shape ``(H, W, 3)`` uint8, RGB.
            選択中の画像。
        path: The path that was posted -- a file or a directory. 元パス。
        is_directory: Whether ``path`` was a directory of images.
        names: Image file names in the deterministic (casefold-sorted) order
            ``load_target_images`` returns them. For a single-file ``path``
            this is ``[path.name]``. ディレクトリの画像名一覧
            （casefold順）。単一ファイルの場合は ``[path.name]``。
        index: Which entry of ``names`` ``image`` corresponds to. Always
            ``0`` for a single-file ``path``. 選択中の index。
    """

    image: np.ndarray
    path: Path
    is_directory: bool
    names: list[str]
    index: int


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


def load_target(path: Path, index: int = 0) -> TargetContext:
    """Load a target image file or a directory of them.

    ディレクトリの場合は ``load_target_images`` で全画像を読み、``index``
    番目（既定 0 = 先頭）を選択中の画像とする。単一ファイルには index が
    1つ（0）しか存在しないため、0 以外を渡すと範囲外のディレクトリindex
    と同じ扱いで拒否する（無視はしない -- 呼び出し側の状態不整合を
    サイレントに飲み込まないため）。

    Raises:
        FileNotFoundError: ``path`` does not exist. 存在しないパス。
        ValueError: ``path`` exists but cannot become a target image --
            an empty directory, an out-of-range ``index`` (for a directory:
            outside ``[0, len(entries))``; for a single file: anything other
            than ``0``, since a single file has exactly one selectable
            entry), an unsupported file extension, or a file ``cv2.imread``
            cannot decode. The message always names ``path`` since the
            router hands it straight to the user.
            ディレクトリが空、``index`` が範囲外（ディレクトリなら
            ``[0, len(entries))`` の外、単一ファイルなら 0 以外すべて）、
            拡張子非対応、デコード不能のいずれか。メッセージには必ず
            パスを含める。
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.is_dir():
        entries = load_target_images(str(path))
        if not entries:
            raise ValueError(
                f"no supported images ({', '.join(SUPPORTED_TARGET_IMAGE_SUFFIXES)}) "
                f"found in {path}"
            )
        if not (0 <= index < len(entries)):
            raise ValueError(
                f"index {index} is out of range for {path} "
                f"({len(entries)} image(s), expected 0 <= index < {len(entries)})"
            )
        names = [name for name, _ in entries]
        image = entries[index][1]
        return TargetContext(
            image=image, path=path, is_directory=True, names=names, index=index
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
    image = _load_single_image(path)
    return TargetContext(
        image=image, path=path, is_directory=False, names=[path.name], index=0
    )
