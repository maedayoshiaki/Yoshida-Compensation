"""``POST /api/generate``: the CLI's projection images, written from the GUI.

このファイルの中心は ``test_saved_images_match_generate_projection_image_exactly``
である。保存された PNG を読み戻し、``pipeline.generate_projection_image()`` の
戻り値と**バイト単位で一致**することを linearizer 3通りで確認する。期待値は
手書きの定数ではなく、テスト内で実関数を呼んで作る。

このエンドポイントは**このプロジェクトで唯一ユーザのファイルシステムへ書く
経路**である。``data/` は 99GB の再取得不能な実測データで、取り違えたパスに
書けば回復できない。したがって「既存があれば 409 で**何も書かない**」を
ステータスコードだけでなく**ディスクの中身**で確認する — 書いてから 409 を
返す実装はステータスだけ見るテストを通ってしまう。

判別力について（このリポジトリの事情）:

- **ターゲット画像の R と B を違う値にする。** ``_save_rgb`` は
  ``cv2.cvtColor(..., COLOR_RGB2BGR)`` を通すため、灰色や単色の画像では
  チャンネルの取り違え（BGR/RGB）が1件も落ちない。
- **縦横を違うサイズにする。** 正方形だと転置が見えない。
- **行列は非対称かつ画素ごとに変化させる。** 対称だと転置が、一様だと
  並べ替えが見えない。
- **メタデータの期待値に既定値を使わない。** ``target_decode="srgb"`` /
  ``target_gamma=2.2`` / 有効領域＝フレーム全体 はいずれも実環境の既定値で、
  それを期待値にすると「入力を読まず既定を書いた実装」と区別できない。
- **有効領域はフレーム全体ではない**（``resolve_active_region`` の既定と
  一致させない）。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp.compensation.inverse_lut import InverseLut, save_inverse_lut  # noqa: E402
from yoshidacomp.compensation.pipeline import (  # noqa: E402
    LINEARIZER_FILENAME_PREFIXES,
    LINEARIZERS,
    generate_projection_image,
)
from yoshidacomp.config import get_config  # noqa: E402
from yoshidacomp_web import generate as generate_module  # noqa: E402
from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import luts, store, targets  # noqa: E402

# The projector frame. Non-square, and deliberately not the same numbers as
# ``tests/test_web_compensation_view.py`` -- a transposed (H, W) anywhere lands
# on a different shape.
_FRAME_WIDTH, _FRAME_HEIGHT = 9, 6

# ``(x, y, w, h)``: off-centre and asymmetric (``x != y``, ``w != h``), and
# **not** the whole frame -- ``resolve_active_region`` falls back to
# ``(0, 0, 9, 6)`` when the configured rectangle is degenerate, so an active
# region equal to the frame would agree with an implementation that never read
# the config at all.
_REGION = (2, 1, 5, 4)

# The target's own size: different from the frame *and* from the active region
# in both dimensions, so ``embed_in_active_region``'s resize genuinely runs
# (7 -> 5 wide, 3 -> 4 tall).
_TARGET_HEIGHT, _TARGET_WIDTH = 3, 7

# A deliberately non-symmetric affine, scaled so ``[t, 1] @ M`` mostly lands
# inside [0, 1] -- a fixture that clipped everywhere would hide every
# difference this file is trying to see.
_A = np.array(
    [
        [0.60, 0.18, -0.12],
        [0.09, 0.54, 0.15],
        [-0.06, 0.12, 0.66],
    ],
    dtype=np.float32,
)
_B = np.array([0.08, 0.05, 0.10], dtype=np.float32)

# An inverse LUT whose three rows have three different shapes and three
# different endpoints: three identical curves cannot tell an R/B row swap from
# the real thing. Same reasoning (and the same numbers) as
# ``tests/test_web_lut.py``.
_T_GRID = [0.0, 0.2, 0.5, 0.9, 1.0]
_LUT_ROWS = [
    [0.0, 0.11, 0.275, 0.495, 0.55],
    [0.0, 0.22, 0.311, 0.417, 0.44],
    [0.0, 0.028, 0.172, 0.557, 0.688],
]


def _matrices(height: int, width: int) -> np.ndarray:
    """A per-pixel-varying, non-symmetric ``(H, W, 4, 3)`` matrix."""
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    modulation = (1.0 + 0.004 * (rows + 2.0 * cols)).astype(np.float32)

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, :3, :] = _A * modulation[:, :, None, None]
    matrices[:, :, 3, :] = _B
    return matrices


def _target_image(height: int, width: int) -> np.ndarray:
    """A multi-coloured ``(H, W, 3)`` uint8 RGB image, deterministic.

    R と B は別の式で、しかも互いに反転している（R は増加、B は減少）。
    ``_save_rgb`` の ``COLOR_RGB2BGR`` を落とした実装を捕まえるには、
    R と B が違う値を持っていなければならない。
    """
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (37 * cols + 11 * rows) % 256
    image[:, :, 1] = (17 * cols + 53 * rows + 90) % 256
    image[:, :, 2] = 255 - (29 * cols + 7 * rows) % 256
    return image


def _write_png(path: Path, image: np.ndarray) -> Path:
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok, f"failed to write test fixture image: {path}"
    return path


def _read_png(path: Path) -> np.ndarray:
    """Read a saved PNG back as ``(H, W, 3)`` uint8 **RGB**."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"could not read back {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_lut(path: Path) -> Path:
    save_inverse_lut(
        path,
        InverseLut(
            lut=np.array(_LUT_ROWS, dtype=np.float32),
            t_grid=np.array(_T_GRID, dtype=np.float32),
            meta={"source": "channel_ramp", "trust_low": 5, "trust_high": 245},
        ),
    )
    return path


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under *root*, as ``relative posix path -> bytes``.

    「何も書かれていない」を確認するのはこれである。ステータスコードだけを
    見ると、書いた上で 409 を返す実装でも通ってしまう。
    """
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _prepared(tmp_path, stand_in_projector_config):
    """Load the standard fixture matrix, target and LUT; return ids and arrays."""
    store.clear()
    targets.clear()
    luts.clear()
    x, y, w, h = _REGION
    stand_in_projector_config(
        width=_FRAME_WIDTH,
        height=_FRAME_HEIGHT,
        active_x=x,
        active_y=y,
        active_width=w,
        active_height=h,
    )
    client = TestClient(create_app())

    matrices = _matrices(_FRAME_HEIGHT, _FRAME_WIDTH)
    matrix_path = tmp_path / "m.npy"
    np.save(matrix_path, matrices)
    response = client.post("/api/session/matrix", json={"path": str(matrix_path)})
    assert response.status_code == 200, response.text
    matrix_id = response.json()["id"]

    image = _target_image(_TARGET_HEIGHT, _TARGET_WIDTH)
    image_path = _write_png(tmp_path / "target.png", image)
    response = client.post("/api/target", json={"path": str(image_path)})
    assert response.status_code == 200, response.text
    target_id = response.json()["id"]

    lut_path = _write_lut(tmp_path / "inverse_lut.npz")
    response = client.post("/api/lut", json={"path": str(lut_path)})
    assert response.status_code == 200, response.text
    lut_id = response.json()["id"]

    return {
        "client": client,
        "matrix_id": matrix_id,
        "target_id": target_id,
        "lut_id": lut_id,
        "matrices": matrices,
        "image": image,
        "matrix_path": matrix_path,
        "target_path": image_path,
        "lut_path": lut_path,
    }


def _body(prepared, output_dir: Path, **overrides) -> dict:
    body = {
        "matrix_id": prepared["matrix_id"],
        "target_id": prepared["target_id"],
        "output_dir": str(output_dir),
        "linearizer": "none",
    }
    body.update(overrides)
    return body


def _reference(
    prepared,
    *,
    linearizer: str = "none",
    target_decode: str = "srgb",
    target_gamma: float = 2.2,
    with_lut: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """What the CLI would compute, via the real pipeline function.

    期待値はここで作る。エンドポイントと同じ引数を同じ関数へ渡し、返ってきた
    2配列をそのまま返す。定数を書き写す余地を残さない。
    """
    lut = None
    if with_lut or linearizer == "inverse_lut":
        lut = luts.get(prepared["lut_id"]).lut
    return generate_projection_image(
        prepared["image"],
        prepared["matrices"],
        projector=get_config().projector,
        active_region=_REGION,
        linearizer=linearizer,  # type: ignore[arg-type]
        lut=lut,
        target_decode=target_decode,  # type: ignore[arg-type]
        target_gamma=target_gamma,
        dtype=np.uint8,
    )


# --------------------------------------------------------------------------
# 1. The central claim: the endpoint writes the pipeline's own arrays.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("linearizer", ["none", "gamma", "inverse_lut"])
def test_saved_images_match_generate_projection_image_exactly(
    tmp_path, stand_in_projector_config, linearizer: str
) -> None:
    """The two PNGs, read back, must equal ``generate_projection_image()``.

    PNG は可逆なので許容誤差は置かない。保存側に数式や再スケールを足せば、
    小数以下で合っていてもここで落ちる。

    ``linear`` と ``projector`` の**両方**を比較しているのが要点で、戻り値
    2つの取り違えはこの比較でしか捕まらない（``generate_projection_image``
    は ``(linear, projector)`` の順に返す）。片方だけを見るテストや、
    「2枚が互いに違う」だけを見るテストは、入れ替えても通ってしまう。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(
            prepared,
            output_dir,
            linearizer=linearizer,
            lut_id=prepared["lut_id"] if linearizer == "inverse_lut" else None,
        ),
    )
    assert response.status_code == 200, response.text

    expected_linear, expected_projector = _reference(prepared, linearizer=linearizer)
    prefix = LINEARIZER_FILENAME_PREFIXES[linearizer]
    saved_linear = _read_png(
        output_dir / "compensation_linear" / "compensation_image_00.png"
    )
    saved_projector = _read_png(
        output_dir / "compensation_projector" / f"{prefix}_comp_image_00.png"
    )

    np.testing.assert_array_equal(saved_linear, expected_linear)
    np.testing.assert_array_equal(saved_projector, expected_projector)

    # --- the fixture's own discriminating power, asserted rather than assumed.
    #
    # A channel-symmetric image (grey, or any single colour) cannot tell
    # ``cv2.cvtColor(..., COLOR_RGB2BGR)`` from its absence: writing RGB bytes
    # into a file read back as BGR is the identity when R == B. Assert that
    # this fixture is not such an image, for the array actually compared.
    assert not np.array_equal(expected_projector, expected_projector[:, :, ::-1]), (
        "the expected projector image is channel-palindromic (R == B "
        "everywhere), so this comparison cannot detect a dropped RGB->BGR "
        "conversion"
    )
    # A square image cannot tell a transposed (H, W) from the real one.
    assert expected_projector.shape[0] != expected_projector.shape[1], (
        "the expected image is square, so this comparison cannot detect an "
        "H/W transposition"
    )
    assert expected_projector.shape == (_FRAME_HEIGHT, _FRAME_WIDTH, 3)


def test_the_linear_and_projector_images_are_not_the_same_file(
    tmp_path, stand_in_projector_config
) -> None:
    """With a real linearizer the two saved images differ.

    ``linearizer="none"`` では両者は同一配列（``_linearize_none`` は受け取った
    オブジェクトをそのまま返す）なので、この確認は ``gamma`` で行う。上の
    完全一致テストと合わせて初めて「2枚とも正しい」が言える — こちらだけでは
    入れ替えを検出できない。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, linearizer="gamma")
    )
    assert response.status_code == 200, response.text

    linear = _read_png(
        output_dir / "compensation_linear" / "compensation_image_00.png"
    )
    projector = _read_png(
        output_dir / "compensation_projector" / "inv_gamma_comp_image_00.png"
    )
    assert not np.array_equal(linear, projector)


# --------------------------------------------------------------------------
# 2. The layout, the filenames, and the reported paths.
# --------------------------------------------------------------------------


def test_the_saved_layout_is_the_cli_layout(
    tmp_path, stand_in_projector_config
) -> None:
    """``compensation_linear/`` + ``compensation_projector/`` + metadata, and nothing else."""
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, linearizer="none")
    )
    assert response.status_code == 200, response.text

    assert set(_tree(output_dir)) == {
        "compensation_linear/compensation_image_00.png",
        "compensation_projector/no_gamma_comp_image_00.png",
        "generate_metadata.json",
    }


def test_the_response_lists_exactly_the_files_that_were_written(
    tmp_path, stand_in_projector_config
) -> None:
    """``files`` は実際に書いたものと過不足なく一致すること。

    「報告したのに書いていない」も「書いたのに報告していない」も、
    ユーザのディスクに対しては同じくらい悪い。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, linearizer="gamma")
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    reported = {Path(p).resolve() for p in payload["files"]}
    on_disk = {(output_dir / name).resolve() for name in _tree(output_dir)}
    assert reported == on_disk
    assert Path(payload["output_dir"]).resolve() == output_dir.resolve()


@pytest.mark.parametrize(
    "linearizer,prefix",
    [("none", "no_gamma"), ("gamma", "inv_gamma"), ("inverse_lut", "inv_lut")],
)
def test_the_projector_filename_prefix_follows_the_linearizer(
    tmp_path, stand_in_projector_config, linearizer: str, prefix: str
) -> None:
    """linearizer を変えると投影画像の接頭辞が変わること。

    期待値をここに直書きしているのは意図的で、``LINEARIZER_FILENAME_PREFIXES``
    から引くと「辞書を引き直しただけ」の恒真式になる。この3行が
    **ディスク上のファイル名**そのものである。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / f"out_{linearizer}"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(
            prepared,
            output_dir,
            linearizer=linearizer,
            lut_id=prepared["lut_id"] if linearizer == "inverse_lut" else None,
        ),
    )
    assert response.status_code == 200, response.text
    assert (
        output_dir / "compensation_projector" / f"{prefix}_comp_image_00.png"
    ).is_file()
    # The linear image's name does not carry the linearizer -- it is the same
    # picture whichever one runs.
    assert (
        output_dir / "compensation_linear" / "compensation_image_00.png"
    ).is_file()


# --------------------------------------------------------------------------
# 3. The filename prefixes are the CLI's, not a second copy of them.
# --------------------------------------------------------------------------


def test_the_cli_and_the_endpoint_share_one_filename_prefix_map() -> None:
    """CLI と ``generate.py`` が**同じ辞書オブジェクト**を読んでいること。

    ブリーフは「片方を輸入するか、同じであることをテストで固定する」を求めて
    いる。ここでは前者を採ったので、確認するのは等価ではなく**同一性**である
    （等価だけを見ると、片方に写しを置いてたまたま一致している状態を許す）。
    """
    from examples.python import rebuild_compensation_with_map as rebuild

    assert (
        rebuild.LINEARIZER_FILENAME_PREFIXES is LINEARIZER_FILENAME_PREFIXES
    ), "the CLI must read the core's mapping, not a copy of it"
    assert (
        generate_module.LINEARIZER_FILENAME_PREFIXES is LINEARIZER_FILENAME_PREFIXES
    ), "generate.py must read the core's mapping, not a copy of it"


def test_the_filename_prefixes_are_the_names_downstream_tools_glob() -> None:
    """接頭辞の**値**を固定する。

    ``examples/python/evaluate_compensation_conditions.py`` はこの名前を
    glob しており、変えると**静かに**何も見つけなくなる。単一の辞書にした
    以上、値の取り違えを捕まえられるのはこの1件だけである。
    """
    assert LINEARIZER_FILENAME_PREFIXES == {
        "none": "no_gamma",
        "gamma": "inv_gamma",
        "inverse_lut": "inv_lut",
    }


def test_every_linearizer_has_a_filename_prefix() -> None:
    """``LINEARIZERS`` に足して接頭辞を足し忘れると落ちること。

    ``LINEARIZERS`` は GUI のドロップダウンの選択肢そのものなので、接頭辞の
    無い linearizer は「選べるが保存で KeyError（＝500）になる」ものになる。
    """
    assert set(LINEARIZER_FILENAME_PREFIXES) == set(LINEARIZERS)


# --------------------------------------------------------------------------
# 4. The metadata.
# --------------------------------------------------------------------------


def test_the_metadata_records_the_actual_request(
    tmp_path, stand_in_projector_config
) -> None:
    """メタデータの値が**実際の入力**と一致すること。

    ここで使う値はどれも既定値ではない: ``linearizer="gamma"``（CLI に既定は
    無いが ``compensation`` ビューの既定は ``"none"``）、
    ``target_decode="gamma"``（既定は ``"srgb"``）、``target_gamma=1.8``
    （既定は 2.2）、有効領域は ``(2, 1, 5, 4)``（``resolve_active_region``
    の退化時の既定はフレーム全体 ``(0, 0, 9, 6)``）、``matrix_shape`` は
    ``[6, 9, 4, 3]``（H != W なので転置が見える）。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(
            prepared,
            output_dir,
            linearizer="gamma",
            target_decode="gamma",
            target_gamma=1.8,
        ),
    )
    assert response.status_code == 200, response.text

    written = json.loads(
        (output_dir / "generate_metadata.json").read_text(encoding="utf-8")
    )
    assert written == response.json()["meta"], (
        "the metadata on disk and the metadata in the response must be the "
        "same document"
    )

    assert written["linearizer"] == "gamma"
    assert written["target_decode"] == "gamma"
    assert written["target_gamma"] == 1.8
    assert written["active_region"] == [2, 1, 5, 4]
    assert written["matrix_shape"] == [_FRAME_HEIGHT, _FRAME_WIDTH, 4, 3]
    assert Path(written["matrix"]) == prepared["matrix_path"]
    assert Path(written["target_path"]) == prepared["target_path"]
    assert written["inverse_lut"] is None
    assert [t["name"] for t in written["targets"]] == ["target.png"]
    assert [t["index"] for t in written["targets"]] == [0]


def test_the_metadata_active_region_is_the_configured_one_not_the_frame(
    tmp_path, stand_in_projector_config
) -> None:
    """有効領域がフレーム全体と**違う**ことを明示的に確認する。

    ``active_region`` の期待値がフレーム全体だと、``resolve_active_region``
    を呼ばず ``(0, 0, width, height)`` を書いた実装と区別が付かない。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir)
    )
    assert response.status_code == 200, response.text

    region = response.json()["meta"]["active_region"]
    assert region == list(_REGION)
    assert region != [0, 0, _FRAME_WIDTH, _FRAME_HEIGHT]


def test_the_metadata_records_the_lut_path_when_one_is_used(
    tmp_path, stand_in_projector_config
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(
            prepared,
            output_dir,
            linearizer="inverse_lut",
            lut_id=prepared["lut_id"],
        ),
    )
    assert response.status_code == 200, response.text

    meta = response.json()["meta"]
    assert Path(meta["inverse_lut"]) == prepared["lut_path"]
    assert meta["linearizer"] == "inverse_lut"


def test_the_metadata_target_gamma_is_null_unless_the_decode_uses_it(
    tmp_path, stand_in_projector_config
) -> None:
    """CLI と同じ規約: ``target_decode != "gamma"`` なら ``target_gamma`` は null。

    使われなかった指数を記録すると、後から読んだ人が「1.8 で復号した」と
    読める。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(prepared, output_dir, target_decode="srgb", target_gamma=1.8),
    )
    assert response.status_code == 200, response.text
    assert response.json()["meta"]["target_gamma"] is None


# --------------------------------------------------------------------------
# 5. Never silently overwrite. This is the one that guards 99 GB.
# --------------------------------------------------------------------------


def test_an_existing_directory_is_refused_with_409_and_nothing_is_written(
    tmp_path, stand_in_projector_config
) -> None:
    """既存ファイルがあるディレクトリには 409 を返し、**1バイトも書かない**。

    ステータスコードだけを見ると、書いた上で 409 を返す実装でも通る。
    ここではディスクの中身を前後で比較する。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "real_measurements"
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "raw" / "IMG_0001.CR3").write_bytes(b"irreplaceable")
    (output_dir / "session.json").write_bytes(b'{"kept": true}')
    before = _tree(output_dir)

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir)
    )

    assert response.status_code == 409, response.text
    assert _tree(output_dir) == before, (
        "the 409 path wrote to disk; an implementation that writes and then "
        "reports the conflict passes a status-only assertion"
    )
    # The refusal has to name what is in the way, or the user cannot tell a
    # mistyped path from a rerun.
    assert "IMG_0001.CR3" in response.text or "session.json" in response.text


def test_a_directory_holding_our_own_output_is_also_refused(
    tmp_path, stand_in_projector_config
) -> None:
    """自分の出力が既にある場合も 409（上書き前提にしない）。"""
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    first = prepared["client"].post("/api/generate", json=_body(prepared, output_dir))
    assert first.status_code == 200, first.text
    before = _tree(output_dir)

    second = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, linearizer="gamma")
    )
    assert second.status_code == 409, second.text
    assert _tree(output_dir) == before


def test_overwrite_true_replaces_the_files(
    tmp_path, stand_in_projector_config
) -> None:
    """``overwrite: true`` のときだけ上書きされること。"""
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"
    linear_path = output_dir / "compensation_linear" / "compensation_image_00.png"
    linear_path.parent.mkdir(parents=True)
    _write_png(linear_path, np.zeros((2, 3, 3), dtype=np.uint8))
    stale = linear_path.read_bytes()

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, overwrite=True)
    )
    assert response.status_code == 200, response.text

    assert linear_path.read_bytes() != stale
    expected_linear, _ = _reference(prepared, linearizer="none")
    np.testing.assert_array_equal(_read_png(linear_path), expected_linear)


def test_overwrite_true_leaves_unrelated_files_alone(
    tmp_path, stand_in_projector_config
) -> None:
    """上書きしても、自分が書かないファイルは消さないこと。

    ``overwrite`` は「同名を置き換える」であって「ディレクトリを作り直す」
    ではない。``shutil.rmtree`` は取り違えたパスに対して回復不能である。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    keep = output_dir / "notes.txt"
    keep.write_bytes(b"keep me")

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, overwrite=True)
    )
    assert response.status_code == 200, response.text
    assert keep.read_bytes() == b"keep me"


def test_a_directory_that_does_not_exist_yet_is_created(
    tmp_path, stand_in_projector_config
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "brand" / "new" / "place"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir)
    )
    assert response.status_code == 200, response.text
    assert (output_dir / "generate_metadata.json").is_file()


def test_an_empty_directory_is_not_a_conflict(
    tmp_path, stand_in_projector_config
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir)
    )
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------
# 6. Bad requests. A mistyped path must never become a 500 or a write.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_output_dir_is_refused(
    tmp_path, stand_in_projector_config, blank: str
) -> None:
    """``Path("")`` は ``Path(".")`` -- つまりサーバの作業ディレクトリである。

    このリポジトリではそこがリポジトリのルートで、``data/`` も ``.git`` も
    その下にある。空欄をそのまま通すと、409 に救われるとはいえ「なぜ
    リポジトリのルートに書こうとしたのか」を誰も説明できない。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)

    body = _body(prepared, tmp_path / "unused")
    body["output_dir"] = blank
    response = prepared["client"].post("/api/generate", json=body)
    assert response.status_code == 400, response.text
    assert "output_dir" in response.json()["detail"]


def test_the_repository_root_is_refused_rather_than_written_into(
    tmp_path, stand_in_projector_config
) -> None:
    """``"."`` （＝サーバの作業ディレクトリ）は 409 で止まること。

    空欄チェックを外しても、非空ディレクトリの判定がもう一段の防壁として
    効いていることを固定する。**このテストは実際にリポジトリのルートを
    ``output_dir`` に指定する。** 通ってしまう実装だと、リポジトリに
    ``compensation_linear/`` と ``generate_metadata.json`` が現れる。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    # ``"."`` is resolved by the *server*, i.e. against the process working
    # directory -- the same one this test has to inspect for the assertion to
    # be about the directory that was at risk.
    cwd = Path.cwd()
    before = sorted(p.name for p in cwd.iterdir())
    assert before, "cwd is empty, so this test cannot detect a write into it"

    body = _body(prepared, tmp_path / "unused")
    body["output_dir"] = "."
    try:
        response = prepared["client"].post("/api/generate", json=body)
        assert response.status_code == 409, response.text
        assert sorted(p.name for p in cwd.iterdir()) == before
    finally:
        # A broken guard leaves its output in the working directory, which
        # here is a 99 GB repository someone has to clean by hand while
        # reading a red suite. Remove only the three names this endpoint
        # writes, and only if they were not there beforehand.
        # 壊れた実装はここ（＝リポジトリのルート）に出力を残す。書き得る3つの
        # 名前だけを、元から無かった場合に限って片付ける。
        for name in (
            generate_module.LINEAR_SUBDIR,
            generate_module.PROJECTOR_SUBDIR,
            generate_module.METADATA_NAME,
        ):
            stray = cwd / name
            if name not in before and stray.exists():
                shutil.rmtree(stray) if stray.is_dir() else stray.unlink()


def test_an_output_dir_that_is_a_file_is_refused(
    tmp_path, stand_in_projector_config
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    not_a_dir = tmp_path / "notes.txt"
    not_a_dir.write_bytes(b"keep me")

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, not_a_dir)
    )
    assert response.status_code == 400, response.text
    assert not_a_dir.read_bytes() == b"keep me"


def test_a_refused_write_is_reported_not_swallowed(
    tmp_path, stand_in_projector_config, monkeypatch
) -> None:
    """``cv2.imwrite`` が ``False`` を返したら 200 にしないこと。

    ``cv2.imwrite`` は失敗しても例外を投げず ``False`` を返す。戻り値を見ない
    実装は「200 OK と、保存したはずのパスの一覧」を返しつつディスクには
    何も無い状態を作る -- ユーザは保存できたと思って画面を閉じる。

    ``False`` を実際に起こさせるには権限や空き容量の細工が要るので、ここでは
    ``cv2.imwrite`` そのものを差し替えて「拒否された」状況だけを作る。
    見ているのは**戻り値を見ているかどうか**の一点である。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir)
    )
    assert response.status_code == 400, response.text
    assert "compensation_image_00.png" in response.json()["detail"]
    # And nothing may be reported as saved.
    assert "files" not in response.json()


def test_an_unknown_matrix_id_is_404(tmp_path, stand_in_projector_config) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, matrix_id="matrix999")
    )
    assert response.status_code == 404, response.text
    assert _tree(output_dir) == {}


def test_an_unknown_target_id_is_404(tmp_path, stand_in_projector_config) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, target_id="target999")
    )
    assert response.status_code == 404, response.text
    assert _tree(output_dir) == {}


def test_an_unknown_lut_id_is_404(tmp_path, stand_in_projector_config) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(
            prepared, output_dir, linearizer="inverse_lut", lut_id="lut999"
        ),
    )
    assert response.status_code == 404, response.text
    assert _tree(output_dir) == {}


def test_inverse_lut_without_a_lut_id_is_400(
    tmp_path, stand_in_projector_config
) -> None:
    """``linearizer="inverse_lut"`` で ``lut_id`` 未指定は 400。

    断らないと ``_linearize_inverse_lut`` の ``ValueError`` が 500 になり、
    「LUT を選び忘れた」だけの要求が「サーバが壊れた」として返る。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, linearizer="inverse_lut")
    )
    assert response.status_code == 400, response.text
    assert "lut_id" in response.json()["detail"]
    assert _tree(output_dir) == {}


def test_an_unknown_linearizer_is_400_not_500(
    tmp_path, stand_in_projector_config
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, linearizer="srgb")
    )
    assert response.status_code == 400, response.text
    assert _tree(output_dir) == {}


def test_an_unknown_target_decode_is_400_not_500(
    tmp_path, stand_in_projector_config
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate", json=_body(prepared, output_dir, target_decode="linear")
    )
    assert response.status_code == 400, response.text
    assert _tree(output_dir) == {}


@pytest.mark.parametrize("gamma", [0.0, -1.0])
def test_a_non_positive_target_gamma_is_400_not_500(
    tmp_path, stand_in_projector_config, gamma: float
) -> None:
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    response = prepared["client"].post(
        "/api/generate",
        json=_body(
            prepared, output_dir, target_decode="gamma", target_gamma=gamma
        ),
    )
    assert response.status_code == 400, response.text
    assert _tree(output_dir) == {}


def test_a_camera_space_matrix_is_refused_without_writing(
    tmp_path, stand_in_projector_config
) -> None:
    """カメラ座標の行列は 400（有効領域が存在しないため）。

    ``views/target_view.py::ViewParams.require_projector_space`` と同じ判定を
    共有している。断らないと ``calc_compensation_image`` が numpy の形状
    ``ValueError`` を投げ、それは 500 になる。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    output_dir = tmp_path / "out"

    camera = _matrices(_FRAME_HEIGHT + 3, _FRAME_WIDTH + 5)
    camera_path = tmp_path / "camera.npy"
    np.save(camera_path, camera)
    loaded = prepared["client"].post(
        "/api/session/matrix", json={"path": str(camera_path)}
    )
    assert loaded.status_code == 200, loaded.text

    response = prepared["client"].post(
        "/api/generate",
        json=_body(prepared, output_dir, matrix_id=loaded.json()["id"]),
    )
    assert response.status_code == 400, response.text
    assert _tree(output_dir) == {}


# --------------------------------------------------------------------------
# 7. A directory target generates the selected image only.
# --------------------------------------------------------------------------


def test_a_directory_target_generates_only_the_selected_image(
    tmp_path, stand_in_projector_config
) -> None:
    """ディレクトリを読み込んでも、書き出すのは**選択中の1枚**だけ。

    決定の理由は ``generate.py`` の冒頭にある。ここで固定するのは、
    (a) 出力が1組であること、(b) 番号が ``TargetContext.index``
    （＝CLI が同じディレクトリに対して付ける番号）であること の2つ。
    """
    prepared = _prepared(tmp_path, stand_in_projector_config)
    directory = tmp_path / "targets"
    directory.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _write_png(directory / name, _target_image(_TARGET_HEIGHT, _TARGET_WIDTH))

    loaded = prepared["client"].post(
        "/api/target", json={"path": str(directory), "index": 2}
    )
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["meta"]["images"] == ["a.png", "b.png", "c.png"]

    output_dir = tmp_path / "out"
    response = prepared["client"].post(
        "/api/generate",
        json=_body(prepared, output_dir, target_id=loaded.json()["id"]),
    )
    assert response.status_code == 200, response.text

    # One pair, numbered by the selection's index -- the same number the CLI
    # gives that file when it walks the whole directory.
    assert set(_tree(output_dir)) == {
        "compensation_linear/compensation_image_02.png",
        "compensation_projector/no_gamma_comp_image_02.png",
        "generate_metadata.json",
    }
    meta = response.json()["meta"]
    assert [t["name"] for t in meta["targets"]] == ["c.png"]
    assert [t["index"] for t in meta["targets"]] == [2]


# --------------------------------------------------------------------------
# 8. Module-level invariants.
# --------------------------------------------------------------------------


def test_existing_entries_does_not_walk_a_whole_directory(tmp_path) -> None:
    """既存の一覧は上限付きで打ち切ること。

    99GB のディレクトリを指されたとき、断り文のために全件を列挙する理由は
    無い。``os.scandir`` を途中で止めていることを、返る件数で固定する。
    """
    directory = tmp_path / "many"
    directory.mkdir()
    for i in range(50):
        (directory / f"f{i:03d}.bin").write_bytes(b"x")

    names = generate_module.existing_entries(directory, limit=5)
    assert len(names) == 6, "at most limit + 1 entries, so 'and more' is knowable"
    assert generate_module.existing_entries(tmp_path / "absent") == []


def test_generate_does_not_import_torch(tmp_path, stand_in_projector_config) -> None:
    """保存経路が torch を引かないこと。

    ``pipeline`` は軽い import だが ``prepare_target_image_for_compensation``
    は ``geometry.warp`` 経由で torch を引く。保存が偶然そちらを呼ぶように
    なると、初回リクエストが数秒止まる。担保しているのは所要時間ではなく
    **依存の不在**である。
    """
    import subprocess
    import sys

    code = (
        "import sys; import yoshidacomp_web.generate as g; "
        "print(int(any(m in sys.modules for m in "
        "('torch', 'psutil', 'yoshidacomp.compensation.color_mixing_matrix'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0", result.stdout


def test_the_output_subdirectory_names_are_the_cli_ones() -> None:
    """並びの名前を固定する。

    ``examples/python/rebuild_compensation_with_map.py`` が
    ``compensation_linear`` / ``compensation_projector`` へ書くので、GUI と
    CLI の出力を同じ道具で突き合わせられる。名前を変えるとその比較が
    静かに壊れる（ファイルが無いだけで、誰も落ちない）。
    """
    source = Path(
        os.fspath(Path(__file__).resolve().parents[1])
    ) / "examples" / "python" / "rebuild_compensation_with_map.py"
    text = source.read_text(encoding="utf-8")
    assert f'"{generate_module.LINEAR_SUBDIR}"' in text
    assert f'"{generate_module.PROJECTOR_SUBDIR}"' in text
