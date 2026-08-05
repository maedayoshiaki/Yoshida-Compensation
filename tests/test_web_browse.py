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
