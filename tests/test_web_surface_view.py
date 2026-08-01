"""The forward-model view: what the camera sees for a given projector command."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402


def _load(client, tmp_path, matrices):
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    return client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]


def test_surface_inverts_a_known_affine(tmp_path, stand_in_projector_config) -> None:
    """M が [C,1]M = P を満たすとき、surface は C = (P - b)A^-1 を返す。

    A = 2I, b = (0.1, 0.2, 0.3) を置くと、P=(1,1,1) に対する答えは
    ((1-0.1)/2, (1-0.2)/2, (1-0.3)/2) = (0.45, 0.40, 0.35) と手計算できる。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 200
    assert response.headers["X-Kind"] == "rgb"
    assert response.headers["X-Channels"] == "3"
    assert response.headers["X-Layout"] == "planar"

    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    np.testing.assert_allclose(planar[0], 0.45, atol=1e-5)
    np.testing.assert_allclose(planar[1], 0.40, atol=1e-5)
    np.testing.assert_allclose(planar[2], 0.35, atol=1e-5)


def test_surface_honours_the_projection_command(tmp_path, stand_in_projector_config) -> None:
    """P=(0,0,0) は環境光のみの応答 -b·A^-1 を返す。"""
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(
        f"/api/matrix/{matrix_id}/view/surface?p_r=0&p_g=0&p_b=0"
    )
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    np.testing.assert_allclose(planar[0], -0.05, atol=1e-5)
    np.testing.assert_allclose(planar[1], -0.10, atol=1e-5)
    np.testing.assert_allclose(planar[2], -0.15, atol=1e-5)


def test_surface_marks_singular_pixels_nan(tmp_path, stand_in_projector_config) -> None:
    """A が特異な画素は逆行列を持たないので NaN にする。

    もっともらしい値を描くとユーザーを誤らせる。NaN ならシェーダが
    マスク色で塗る。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[2, 3, :3, :] = 0.0          # この画素だけ A = 0 で特異
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    assert np.isnan(planar[:, 2, 3]).all()
    assert np.isfinite(planar[:, 0, 0]).all()


def test_surface_marks_near_singular_pixel_nan_even_with_ordinary_magnitudes(
    tmp_path, stand_in_projector_config
) -> None:
    """A near-singular A must be NaN even though every entry is O(0.1-1).

    ``|det(A)|`` alone is a poor singularity test: it scales as the cube of
    A's entries, so a fixed absolute threshold changes meaning with the
    matrix's units. This pixel's A is the classic 3x3 "magic square" (rows
    are an arithmetic progression, so row2 = 2*row1 - row0 and det = 0
    exactly) scaled by 0.1 and then perturbed by 1e-6 in one entry:
    ``|det(A)| ~= 3e-8`` -- comfortably *above* the old fixed threshold of
    1e-12, so a version of this view using that absolute threshold would
    call this pixel invertible and paint a plausible-looking but
    meaningless number -- while ``|det(A)| / ||A||_F**3 ~= 6.2e-9`` correctly
    identifies it as singular in scale-invariant terms.

    ``|det(A)|`` 単体は特異性の判定として不十分。行列式は成分の3乗で
    スケールするため、固定の絶対しきい値は行列の単位が変わると意味が
    変わってしまう。この画素の A は古典的な「魔方陣」（行が等差数列で
    並ぶため row2 = 2*row1 - row0 となり det=0 になる行列）を0.1倍し、
    1成分だけ1e-6ずらしたもの。``|det(A)|`` は約3e-8で旧しきい値1e-12を
    優に超えるため、絶対しきい値版のこのビューなら正則と誤判定して
    もっともらしいが無意味な値を描いていたが、``||A||_F`` の3乗との比は
    約6.2e-9であり、スケール不変な基準では明確に特異である。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0  # everywhere else: a healthy, well-conditioned A

    near_singular = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9 + 1e-6],
        ],
        dtype=np.float32,
    )
    matrices[1, 2, :3, :] = near_singular
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    assert np.isnan(planar[:, 1, 2]).all(), (
        "near-singular pixel (|det(A)| ~= 3e-8 but |det(A)|/||A||_F**3 ~= 6.2e-9) "
        "must be NaN -- a fixed absolute |det| threshold lets it through"
    )
    assert np.isfinite(planar[:, 0, 0]).all()


def test_surface_masks_outside_active_region_and_composes_with_singularity(
    tmp_path, stand_in_projector_config
) -> None:
    """The NaN mask must honour ``(x, y, w, h)`` as-is, and compose with the
    per-pixel singularity check, not overwrite it.

    Mirrors ``test_coef_tiles_masks_outside_active_region`` in
    ``test_web_views.py``: every other test in this file uses a 5x4 frame
    where ``x != y`` and ``w != h`` only by accident, so this test picks an
    asymmetric, off-centre active rectangle on purpose -- a transposed
    ``x``/``y`` or ``w``/``h`` in the mask slice lands on a disjoint set of
    pixels and is caught explicitly below. It also plants one singular pixel
    *inside* the active region, alongside finite pixels both inside and
    outside, so all three cases (finite+inside, singular+inside,
    finite+outside) are asserted to land where they should -- proving the
    active-region mask and the singularity mask compose with AND rather
    than one silently overwriting the other.

    ``coef_tiles.py`` の ``test_coef_tiles_masks_outside_active_region`` に
    倣う。このファイルの他のテストは5x4という偶然 ``x != y`` かつ
    ``w != h`` になっているだけの frame を使っているため、ここでは意図的に
    非対称・非中央の有効領域を選ぶ。行列転置バグは別の画素集合に着地する
    ため以下で明示的に検出できる。さらに有効領域の内側に特異画素を1つ
    置き、内側・外側それぞれの正則画素とあわせて3パターン
    （正則+内側、特異+内側、正則+外側）が期待通りの場所に現れるかを
    確認する。これにより、有効領域マスクと特異性マスクが AND で合成され、
    片方がもう片方を黙って上書きしていないことを証明する。
    """
    store.clear()
    height, width = 6, 8
    x, y, w, h = 1, 2, 5, 3  # active rect occupies rows 2-4, cols 1-5.
    stand_in_projector_config(
        width=width,
        height=height,
        active_x=x,
        active_y=y,
        active_width=w,
        active_height=h,
    )
    client = TestClient(create_app())

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0  # well-conditioned everywhere...
    singular_row, singular_col = 3, 3
    matrices[singular_row, singular_col, :3, :] = 0.0  # ...except this one, inside the region

    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 200
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, height, width)
    finite_mask = np.all(np.isfinite(planar), axis=0)

    expected_mask = np.zeros((height, width), dtype=bool)
    expected_mask[y : y + h, x : x + w] = True
    expected_mask[singular_row, singular_col] = False
    assert np.array_equal(finite_mask, expected_mask), (
        "active-region mask does not match (x, y, w, h) as configured, or does "
        "not compose with the singularity mask -- check for a transposed x/y "
        "or w/h in the mask slice"
    )

    # finite + inside: the active rectangle's corner, away from the singular pixel.
    assert np.all(np.isfinite(planar[:, y, x]))
    # singular + inside: the planted singular pixel must stay NaN, not be
    # "rescued" by being inside the active region.
    assert np.all(np.isnan(planar[:, singular_row, singular_col]))
    # finite + outside: a well-conditioned pixel outside the region must still
    # be NaN -- the active-region mask must not be skipped just because A
    # itself is invertible.
    assert np.all(np.isnan(planar[:, 0, 0]))
    # The point a transposed (x, y) reading would have kept instead (row=x,
    # col=y) must be NaN, since x != y and w != h here.
    assert np.all(np.isnan(planar[:, x, y]))


def test_surface_keeps_zero_as_the_black_point_while_the_response_is_positive(
    tmp_path, stand_in_projector_config
) -> None:
    """A positive response must keep ``vmin == 0``, so the colour is the colour.

    The rgb branch of the shader maps ``[vmin, vmax]`` onto ``[0, 1]`` and then
    tone-maps, so ``vmin`` *is* the black point. Letting it float up to the
    data's own minimum would subtract a constant from all three channels, which
    is not a brightness change but a desaturation: a surface reading
    ``(0.25, 0.375, 0.5)`` would be drawn with a hue it does not have. This
    view's whole question is "what colour is the projection surface", so its
    zero has to stay at zero whenever the data allows it.

    rgb経路は ``[vmin, vmax]`` を ``[0, 1]`` へ写すので ``vmin`` が黒レベルに
    なる。データの最小値へ浮かせると3チャンネルから同じ定数を引くことになり、
    明るさではなく彩度が変わる。「投影面が何色か」を問うビューなので、
    データが許す限り 0 を黒に固定する。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.5, 0.25, 0.0], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 200
    # Response is ((1-0.5)/2, (1-0.25)/2, (1-0)/2) = (0.25, 0.375, 0.5) everywhere.
    assert float(response.headers["X-Vmin"]) == 0.0
    assert float(response.headers["X-Vmax"]) == pytest.approx(0.5)


def test_surface_lets_vmin_follow_the_data_when_the_response_goes_negative(
    tmp_path, stand_in_projector_config
) -> None:
    """An ambient-only command is all-negative and must not render as black.

    ``P = (0, 0, 0)`` gives ``-b A^-1``, which is negative wherever the ambient
    term is positive -- the case this view answers "how much light is on the
    surface with the projector dark". With ``vmin`` pinned at 0 every one of
    those values clips to the bottom and the whole active region renders as a
    flat black frame, indistinguishable from "no data" and from "exactly zero".
    So the floor drops to the data's own 1st percentile (the mirror of the 99th
    used for ``vmax``, and equally robust to a single wild pixel) *when, and
    only when, that is below zero*.

    ``P=(0,0,0)`` の応答 ``-b A^-1`` は負であり、``vmin`` を0に固定すると
    有効領域が一様な黒になって「データ無し」とも「ちょうど0」とも区別が
    つかない。負値があるときだけ下端を1パーセンタイル（``vmax`` の99の鏡像）
    まで下げる。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.5, 0.25, 0.0], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(
        f"/api/matrix/{matrix_id}/view/surface?p_r=0&p_g=0&p_b=0"
    )
    assert response.status_code == 200
    # Response is -(0.5, 0.25, 0)/2 = (-0.25, -0.125, 0.0) everywhere.
    assert float(response.headers["X-Vmin"]) == pytest.approx(-0.25)
    # The span still has to be positive, or the shader's max(vmax - vmin, 1e-9)
    # would be doing the work instead of the range.
    assert float(response.headers["X-Vmax"]) > float(response.headers["X-Vmin"])


# A deliberately non-symmetric affine, and a projection command whose three
# components are all different.
#
# Every other fixture in this file (and in ``test_web_golden.py``) uses
# ``A = 2I`` with a command of ``(1,1,1)`` or ``(0,0,0)``. That combination
# cannot see two real mistakes, and this was measured rather than guessed:
# with the registry patched to compute ``A**-T`` instead of ``A**-1``
# (``"hwk,hwkc->hwc"`` -> ``"hwc,hwkc->hwk"``) the whole file still passed, and
# so did swapping ``p_r`` with ``p_g`` in the command vector. A diagonal matrix
# is its own transpose and three equal components survive any permutation, so
# neither mutation had anything to change. The near-singular fixture does not
# help either: both ``det`` and ``||A||_F`` are transpose-invariant.
#
# That matters because the matrices this view actually renders are strongly
# non-symmetric -- ``data/comp_test_20260724/matrices_gamma.npy`` has
# ``max|offdiag| / max|diag| = 0.251`` at its centre -- so a transposed inverse
# would return a *different, plausible* surface colour on every real input, in
# the one view whose entire output is a colour. ``A`` below sits at 0.273 by the
# same measure, and is well-conditioned (``|det|/||A||_F**3 = 0.144``, five
# orders of magnitude above ``_DET_REL_EPS``).
#
# 他のフィクスチャは A = 2I かつコマンドが (1,1,1)/(0,0,0) であり、これでは
# 逆行列の転置（A^-T）も p_r と p_g の取り違えも検出できない（実際に両方の
# 変異を入れてもこのファイルは全て通った）。対角行列は自身の転置に等しく、
# 3成分が等しいベクトルはどう入れ替えても同じだからである。det も ||A||_F も
# 転置不変なので特異性のフィクスチャも役に立たない。実データの行列は明確に
# 非対称（実測で非対角/対角 = 0.251）であり、転置した逆行列は「違うが
# もっともらしい」色を返す。以下の A はこの比が 0.273 で、条件数も十分よい。
_ASYMMETRIC_A = np.array(
    [
        [1.00, 0.30, -0.20],
        [0.15, 0.90, 0.25],
        [-0.10, 0.20, 1.10],
    ],
    dtype=np.float32,
)
_ASYMMETRIC_B = np.array([0.05, -0.12, 0.20], dtype=np.float32)
_ASYMMETRIC_COMMAND = (0.2, 0.6, 0.9)


def _asymmetric_matrices(height: int, width: int) -> np.ndarray:
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, :3, :] = _ASYMMETRIC_A
    matrices[:, :, 3, :] = _ASYMMETRIC_B
    return matrices


def test_surface_round_trips_a_non_symmetric_affine_under_an_asymmetric_command(
    tmp_path, stand_in_projector_config
) -> None:
    """``[C, 1] @ M`` must reproduce ``P`` for a non-diagonal ``M``.

    The expectation is derived by an *independent round trip* rather than from
    hand-computed literals. ``photometric_compensation.calc_compensation_image``
    documents the convention as ``[R, G, B, 1] @ M``; this view claims to invert
    exactly that. So feeding the view's own answer back through the forward
    model and requiring the projection command to come out pins both
    derivations at once -- the einsum's index order and the order of the
    components in the command vector -- with no literal for a future edit to
    quietly agree with.

    期待値は手計算の定数ではなく、順方向モデルによる往復で導く。
    ``[R,G,B,1] @ M`` という規約は
    ``photometric_compensation.calc_compensation_image`` が定めており、この
    ビューはその逆を主張している。ビューの答えを順方向へ戻して投影コマンドが
    再現されることを要求すれば、einsum の添字順とコマンドベクトルの成分順の
    両方を、書き間違えようのある定数を一切置かずに固定できる。
    """
    store.clear()
    height, width = 4, 5
    stand_in_projector_config(width=width, height=height)
    client = TestClient(create_app())

    matrices = _asymmetric_matrices(height, width)
    matrix_id = _load(client, tmp_path, matrices)

    p_r, p_g, p_b = _ASYMMETRIC_COMMAND
    response = client.get(
        f"/api/matrix/{matrix_id}/view/surface?p_r={p_r}&p_g={p_g}&p_b={p_b}"
    )
    assert response.status_code == 200
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, height, width)

    forward = matrices[0, 0].astype(np.float64)  # (4, 3), the same M everywhere
    command = np.array(_ASYMMETRIC_COMMAND, dtype=np.float64)

    # (3, H, W) -> (N, 3), then [C, 1] @ M for every pixel at once.
    surface = planar.reshape(3, -1).T.astype(np.float64)
    homogeneous = np.concatenate([surface, np.ones((surface.shape[0], 1))], axis=1)
    np.testing.assert_allclose(
        homogeneous @ forward,
        np.broadcast_to(command, (surface.shape[0], 3)),
        atol=1e-5,
    )

    # The fixture's discriminating power, asserted rather than assumed.
    #
    # Both of these are what the two mutations above actually produce, and both
    # are *finite and plausible* -- the failure this test exists for is a wrong
    # colour, not an exception. If a later edit "simplifies" A back to something
    # diagonal, or makes the command's components equal, these two lines fail
    # first and say why, instead of the file silently going back to proving
    # nothing.
    # フィクスチャの判別力そのものを検証する。以下はどちらも実際の変異が返す
    # 値で、いずれも有限でもっともらしい（このテストが守っているのは例外では
    # なく「色が違う」こと）。将来 A を対角に戻したりコマンドの成分を揃えたり
    # したら、まずここが落ちて理由を告げる。
    a_mat = forward[:3, :]
    b_vec = forward[3, :]
    transposed = (command - b_vec) @ np.linalg.inv(a_mat).T
    assert np.max(np.abs(transposed @ a_mat + b_vec - command)) > 1e-3, (
        "A is too close to symmetric for this test to tell A**-1 from A**-T"
    )
    swapped = np.array([p_g, p_r, p_b], dtype=np.float64)
    assert np.max(np.abs(swapped - command)) > 1e-3, (
        "the command's components are too close together to tell p_r from p_g"
    )


def test_surface_refuses_a_camera_space_matrix(tmp_path, stand_in_projector_config) -> None:
    """A camera-space matrix must be refused, not rendered at whatever cost.

    ``surface`` inverts one 3x3 per pixel, and that price is set by the pixel
    count alone. Measured on the real 1080p projector-space matrix (95 MB):
    2.77 s and 651 MB peak to produce a 24 MB response. Spec section 9 lists
    ``data/color_mixing_matrices.npy`` -- 4660x6984, 1.56 GB, camera space --
    as a first-class input, and the view selector offers ``surface`` for it.
    Extrapolated by pixel count that is ~10 GB peak and ~40 s, with no guard in
    front of it: the machine thrashes and the user gets a bare 500 at the end.

    It is refused rather than merely slowed because the answer would be wrong
    anyway. ``active_region`` is ``None`` outside projector space, so the mask
    that keeps this view honest does not run, and every camera pixel that never
    saw the projector would be painted a "surface colour" derived from an
    affine fitted to nothing.

    ``surface`` は画素ごとに3x3の逆問題を解くため、代金は画素数だけで決まる。
    1080p のプロジェクタ座標行列（95MB）で実測 2.77 秒・ピーク 651 MB。
    仕様書 9 節が第一級の入力として挙げるカメラ座標行列は 4660x6984 =
    1.56 GB であり、画素数で外挿するとピーク約 10 GB・約 40 秒になる。
    しかも ``active_region`` はプロジェクタ座標でなければ ``None`` なので、
    このビューを正直に保っているマスクが走らず、投影を一度も受けていない
    カメラ画素にまで「投影面の色」を描いてしまう。遅いだけでなく誤りである。
    """
    store.clear()
    # The matrix is 4x5; the projector frame is not, so it classifies as camera.
    stand_in_projector_config(width=64, height=48)
    client = TestClient(create_app())

    matrices = _asymmetric_matrices(4, 5)
    matrix_id = _load(client, tmp_path, matrices)
    assert client.get(f"/api/matrix/{matrix_id}/meta").json()["space"] == "camera"

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "camera" in detail
    assert "surface" in detail
    # The message has to say the refusal is a "not yet", not a "never" --
    # otherwise the only route from here is to assume the view is broken.
    assert "B4" in detail or "計画" in detail

    # The scalar view over the same matrix is unaffected: this guard belongs to
    # ``surface``, not to camera-space matrices in general.
    # 同じ行列でも係数タイルは今まで通り描ける。これは surface 固有の制限である。
    assert client.get(f"/api/matrix/{matrix_id}/view/coef_tiles").status_code == 200


def test_surface_is_listed_with_its_params() -> None:
    store.clear()
    client = TestClient(create_app())
    views = client.get("/api/views").json()["views"]
    spec = next(v for v in views if v["key"] == "surface")
    assert spec["kind"] == "rgb"
    assert spec["channels"] == 3
    # The picture here is the projected colour, not the matrix, so the readout's
    # second panel says something the 4x3 table beside it cannot -- and has to
    # stay. This is the other half of the same declaration ``coef_tiles`` uses
    # to have it dropped.
    # ここで描いているのは投影面の色であって行列ではないので、読み取り欄の
    # 2枚目の板は隣の4x3が言えないことを述べており、残さねばならない。
    assert spec["values_are_matrix"] is False
    assert {p["name"] for p in spec["params"]} == {"p_r", "p_g", "p_b"}
    assert all(p["type"] == "number" for p in spec["params"])
