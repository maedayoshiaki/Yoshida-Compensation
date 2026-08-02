"""Write the CLI's projection images to disk, from the GUI.

``POST /api/generate`` の実体。``examples/python/rebuild_compensation_with_map.py``
（以下 CLI）が書くのと**同じ並び・同じファイル名・同じ画素**を出力する。

    <output_dir>/
    ├── compensation_linear/compensation_image_<NN>.png      線形補償画像
    ├── compensation_projector/<prefix>_comp_image_<NN>.png  投影画像
    └── generate_metadata.json

ディレクトリ名は CLI のもの（``compensation_linear`` /
``compensation_projector``）である。計画のブリーフは図の中で ``linear/`` /
``projector/`` と略記しているが、同じ節の見出しが「CLI に合わせる」であり、
図中のファイル名は CLI のものそのままなので、略記のほうを図の意図と読んだ。
実利もこちらにある: GUI と CLI の出力を ``fc /b`` なり ``cmp`` なりで直接
突き合わせられる。メタデータのファイル名だけは ``generate_metadata.json``
とした（CLI の ``rebuild_metadata.json`` は CLI 自身の動詞を含んでいる）。

**このモジュールはこのプロジェクトで唯一ユーザのファイルシステムへ書く。**
その一点だけで、他のどのモジュールとも危険度が違う。ここの ``data/`` は
99GB の実測データで、撮り直しは機材と被写体と時間を要求する — つまり
取り違えたパスへの上書きは回復できない。したがって:

1. **既存があれば書かない。** ``output_dir`` に**何か1つでも**入っていれば
   ``OutputDirNotEmpty`` を上げる（router が 409 にする）。自分が書く2つの
   ファイル名だけを見るのでは足りない: 打ち間違えて実測ディレクトリを指した
   とき、そこに ``compensation_image_00.png`` は無いので衝突が検出できず、
   ``.CR3`` の隣に別物を撒くことになる。「空でなければ止まる」なら、
   打ち間違いは**必ず**止まる。
2. **``overwrite=True`` でも消さない。** 上書きとは「同名を置き換える」で
   あって「ディレクトリを作り直す」ではない。``shutil.rmtree`` はこの
   モジュールに存在しない。
3. **判定は書き込みの前に、1回だけ。** ``mkdir`` すら判定より先には
   呼ばない（``parents=True`` は打ち間違えたパスの木を丸ごと作る）。

**ディレクトリのターゲットは選択中の1枚だけを書く。** CLI は全部書くが、
GUI では以下の理由でそうしない。

- **店にあるのは1枚だけである。** ``TargetContext`` が持つ復号済み画素は
  選択中の1枚で、残りは名前しか無い（``target.py`` の2段構えはそのために
  ある）。全部書くにはここでディスクを読み直すことになり、それは
  「``POST /api/target`` で読み込んで確かめた画像」ではなく「いま
  ディレクトリに入っている画像」を書くという別の約束になる。
- **止められない。** HTTP のリクエストに中断も進捗も無い。100枚のディレクトリ
  を選んで押した瞬間に、100組の PNG が黙って書かれ始める。上限を設ける案も
  あったが、「51枚目から書かれない」は「全部書かれる」より分かりにくい。
- **画面に出ているものが保存される。** GUI の選択（``POST /api/target`` の
  ``index``）は1枚を選ぶためにあり、``compensation`` ビューはその1枚を描く。
  保存されるのが描かれているものと同じであれば、説明が要らない。
- 一括生成は CLI の仕事のままである。CLI には Ctrl+C がある。

連番は ``TargetContext.index`` を使う。ディレクトリの並び順は
``target.py::_listing`` と ``pipeline.load_target_images`` で同じ
（casefold 昇順）なので、GUI が3枚目を選んで書く ``..._02.png`` は、CLI が
同じディレクトリを一巡して書く ``..._02.png`` と同じ画像になる。
（例外: ``load_target_images`` は ``cv2.imread`` が読めないファイルを飛ばすが
``_listing`` は残すので、読めないファイルを含むディレクトリでは以降の番号が
1つずれる。読めないファイルは選んだ時点で 400 になるので GUI 側からは
気付ける。）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from yoshidacomp.compensation.pipeline import (
    LINEARIZER_FILENAME_PREFIXES,
    generate_projection_image,
)
from yoshidacomp.config import ProjectorConfig

from .lut import LutContext
from .matrix import MatrixContext
from .target import TargetContext

# The CLI's two subdirectories, spelled its way. Pinned by
# ``tests/test_web_generate.py::test_the_output_subdirectory_names_are_the_cli_ones``,
# which greps the CLI's source for these literals -- the two writers have no
# other reason to agree, and a divergence would be invisible (the comparison
# tool simply finds no file).
LINEAR_SUBDIR = "compensation_linear"
PROJECTOR_SUBDIR = "compensation_projector"

# Not the CLI's ``rebuild_metadata.json``: that name carries the CLI's own verb
# and the two documents do not describe the same run (there is no ``config``
# path, no ``p2c_map`` and no warp in this one -- the GUI refuses a camera-space
# matrix rather than warping it). Every key that *does* describe the same thing
# is spelled the CLI's way, so the two files can be diffed key by key.
METADATA_NAME = "generate_metadata.json"


class OutputDirNotEmpty(Exception):
    """``output_dir`` already holds something. Nothing has been written.

    Attributes:
        output_dir: The directory, resolved.
        entries: Up to ``limit + 1`` names found in it, so the caller can say
            "and more" without having listed a 99 GB directory.
    """

    def __init__(self, output_dir: Path, entries: list[str]) -> None:
        self.output_dir = output_dir
        self.entries = entries
        shown = ", ".join(entries[:_ENTRY_LIMIT])
        more = " and more" if len(entries) > _ENTRY_LIMIT else ""
        super().__init__(
            f"{output_dir} is not empty (contains {shown}{more}). Refusing to "
            "write into a directory that already holds files -- this endpoint "
            "writes to your local disk and a mistyped path is not recoverable. "
            "Choose an empty or new directory, or resend with overwrite=true "
            "to replace files of the same name (nothing else is deleted). "
            f"（{output_dir} は空ではありません。既存を壊さないため書き込みを"
            "中止しました。別のディレクトリを指定するか、同名ファイルの置換を"
            "許す場合のみ overwrite=true を付けて再送してください。）"
        )


_ENTRY_LIMIT = 5


def existing_entries(output_dir: Path, limit: int = _ENTRY_LIMIT) -> list[str]:
    """Names already in *output_dir*: at most ``limit + 1`` of them.

    Stops early on purpose. The one number the caller needs beyond the names is
    "are there more than I am showing", and that is ``len(result) > limit`` --
    listing a directory of 99 GB of RAW files to build a refusal message would
    be paying an unbounded cost to say the same sentence.
    途中で打ち切る。断り文に必要なのは「表示した以上にあるか」だけで、
    全件を数える理由が無い。

    Returns:
        ``[]`` when *output_dir* does not exist. A non-directory path is the
        caller's to reject (see ``write_projection_images``), not this
        function's -- ``os.scandir`` on a file raises ``NotADirectoryError``,
        which is an ``OSError`` and reaches the user as a 400 either way.
    """
    if not output_dir.exists():
        return []
    with os.scandir(output_dir) as entries:
        return [entry.name for _, entry in zip(range(limit + 1), entries)]


def _save_rgb(path: Path, image: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 **RGB** array as a PNG.

    The CLI's ``_save_rgb``, kept byte-identical in behaviour. The
    ``COLOR_RGB2BGR`` is not decoration: ``cv2.imwrite`` writes channel 0 as
    blue, so dropping it saves a file whose red and blue are swapped -- a
    perfectly valid PNG of the wrong image, which nothing downstream can
    detect. ``tests/test_web_generate.py::
    test_saved_images_match_generate_projection_image_exactly`` reads the file
    back and compares against ``generate_projection_image()``'s own array,
    using a fixture whose R and B differ (a grey or single-colour image would
    be identical either way).

    Raises:
        OSError: ``cv2.imwrite`` refused the path. It returns ``False`` rather
            than raising, and a silent ``False`` here means "the user asked for
            a report and got an empty directory".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path} (cv2.imwrite refused the path)")


def write_projection_images(
    *,
    matrix: MatrixContext,
    target: TargetContext,
    projector: ProjectorConfig,
    active_region: tuple[int, int, int, int],
    output_dir: Path,
    linearizer: str,
    lut: LutContext | None = None,
    target_decode: str = "srgb",
    target_gamma: float = 2.2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the selected target's images and save them under *output_dir*.

    Every argument is already resolved and validated by the router; this
    function makes exactly two decisions of its own -- whether the directory is
    safe to write into, and what to call the files.

    Args:
        matrix: The loaded projector-space matrix. Camera space is the
            router's to refuse (``views/target_view.py``).
        target: The loaded target. **Only ``target.image`` is written** -- see
            the module docstring for why a directory does not generate all of
            its images.
        active_region: ``(x, y, w, h)``, from ``resolve_active_region``.
        linearizer: A key of ``LINEARIZERS`` *and* of
            ``LINEARIZER_FILENAME_PREFIXES``; the router has already rejected
            anything else, so a ``KeyError`` here would be a bug.
        lut: Required for ``linearizer="inverse_lut"``. The context rather than
            the bare ``InverseLut`` so the metadata can record its path.
        overwrite: Replace files of the same name. Never deletes anything else.

    Returns:
        ``{"output_dir": str, "files": [str, ...], "meta": {...}}``. ``files``
        lists every path written, the metadata JSON included, so a caller can
        check that what was reported and what is on disk are the same set.

    Raises:
        OutputDirNotEmpty: *output_dir* holds files and *overwrite* is false.
            Nothing has been written and nothing has been created.
        OSError: the path could not be created or written -- a mistyped drive,
            a name Windows will not accept, a permission. The router reports
            these as 400: they are almost always the path the user typed, and
            the OS message names it.
    """
    started = time.perf_counter()
    output_dir = Path(output_dir).expanduser()

    # Before any mkdir, and before any pixel is computed. ``mkdir(parents=True)``
    # on a mistyped path creates the whole tree, so even the directory itself is
    # not created until this has passed.
    # 判定は ``mkdir`` より前に行う（``parents=True`` は打ち間違えた木を
    # まるごと作る）。画素の計算より前でもあるので、断るときは無駄な計算も
    # していない。
    if not overwrite:
        entries = existing_entries(output_dir)
        if entries:
            raise OutputDirNotEmpty(output_dir, entries)

    linear, projector_image = generate_projection_image(
        target.image,
        matrix.array,
        projector=projector,
        active_region=active_region,
        linearizer=linearizer,  # type: ignore[arg-type]
        lut=lut.lut if lut is not None else None,
        target_decode=target_decode,  # type: ignore[arg-type]
        target_gamma=target_gamma,
        # The CLI's dtype, deliberately, unlike ``views/compensation.py``: what
        # is being produced here is a PNG the projector will display, not a
        # float surface a viewer will tone-map. ``np.uint8`` is the same 8-bit
        # quantization the CLI's files carry.
        dtype=np.uint8,
    )

    # ``generate_projection_image`` returns ``(linear, projector)`` in that
    # order. Named on both sides rather than indexed, for the reason
    # ``views/compensation.py`` spells out at length: a swap still writes two
    # full, plausible PNGs, and the only thing that catches it is comparing
    # *both* files against the function's own return -- which
    # ``test_saved_images_match_generate_projection_image_exactly`` does.
    index = int(target.index)
    prefix = LINEARIZER_FILENAME_PREFIXES[linearizer]
    linear_path = output_dir / LINEAR_SUBDIR / f"compensation_image_{index:02d}.png"
    projector_path = (
        output_dir / PROJECTOR_SUBDIR / f"{prefix}_comp_image_{index:02d}.png"
    )
    _save_rgb(linear_path, linear)
    _save_rgb(projector_path, projector_image)

    meta: dict[str, Any] = {
        # The CLI's key names wherever the concept is the same, so the two
        # metadata documents can be diffed key by key. Absent here and present
        # there: ``config`` (the GUI has no config path to record --
        # ``AppConfig`` does not keep one), ``p2c_map`` /
        # ``matrix_warp_seconds`` (nothing is warped; a camera-space matrix is
        # refused), ``camera_half_size`` (no RAW is read).
        "matrix": str(matrix.path),
        "matrix_shape": [int(v) for v in matrix.array.shape],
        "linearizer": linearizer,
        "inverse_lut": str(lut.path) if lut is not None else None,
        "target_decode": target_decode,
        # Null unless it was used, exactly as the CLI writes it: recording an
        # exponent that did not run reads, later, as "this was decoded at 1.8".
        "target_gamma": target_gamma if target_decode == "gamma" else None,
        # The CLI's key is ``target_dir`` because its input is always a
        # directory; this one accepts a single file too, and calling a file a
        # ``_dir`` is the kind of quietly wrong name this codebase keeps
        # closing. ``target_is_directory`` says which it was.
        "target_path": str(target.path),
        "target_is_directory": bool(target.is_directory),
        "active_region": [int(v) for v in active_region],
        # The GUI has no config file path to record, so this block is the only
        # record of which projector these images were computed for. Without it
        # a folder of PNGs cannot be told from one generated against a
        # different frame -- and this repository's ``config.toml`` has eight
        # sibling variants in the working tree.
        "projector": {
            "width": int(projector.width),
            "height": int(projector.height),
            "gamma": float(projector.gamma),
        },
        "targets": [
            {
                "name": target.names[index],
                "index": index,
                "seconds": time.perf_counter() - started,
            }
        ],
        "total_seconds": time.perf_counter() - started,
    }

    metadata_path = output_dir / METADATA_NAME
    metadata_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "output_dir": str(output_dir),
        "files": [str(linear_path), str(projector_path), str(metadata_path)],
        "meta": meta,
    }


# ``LINEARIZER_FILENAME_PREFIXES`` is listed deliberately: it is imported for
# use above, and naming it here makes ``generate.LINEARIZER_FILENAME_PREFIXES``
# -- which ``test_the_cli_and_the_endpoint_share_one_filename_prefix_map``
# asserts is the core's own object -- a documented part of this module rather
# than an import that happens to be reachable through it.
__all__ = [
    "LINEARIZER_FILENAME_PREFIXES",
    "LINEAR_SUBDIR",
    "METADATA_NAME",
    "OutputDirNotEmpty",
    "PROJECTOR_SUBDIR",
    "existing_entries",
    "write_projection_images",
]
