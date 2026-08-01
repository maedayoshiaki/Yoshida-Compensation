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


def test_surface_is_listed_with_its_params() -> None:
    store.clear()
    client = TestClient(create_app())
    views = client.get("/api/views").json()["views"]
    spec = next(v for v in views if v["key"] == "surface")
    assert spec["kind"] == "rgb"
    assert spec["channels"] == 3
    assert {p["name"] for p in spec["params"]} == {"p_r", "p_g", "p_b"}
    assert all(p["type"] == "number" for p in spec["params"])
