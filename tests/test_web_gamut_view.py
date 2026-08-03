"""The ``gamut`` view: which pixels the compensation cannot represent.

このファイルの中心は2件ある。

- ``test_every_pixel_is_classified_from_the_unclipped_compensation``:
  クラス地図が ``generate_projection_image(dtype=np.float32, clip=False)`` の
  戻り値から導いた分類と**画素単位で完全一致**すること。期待値は手書きの
  定数ではなく、テスト内で実関数を呼んで作る。
- ``test_pixels_at_exactly_zero_and_exactly_one_are_in_range``:
  境界が閉区間であること。``<`` と ``<=`` の取り違えを検出する唯一の手段。
  こちらは上の ``np.where`` 連鎖を**使わずに**書いてある。同じ式を写すと、
  実装と同じ誤りを写したときに一緒に通ってしまうためである。

判別力について（このリポジトリの事情。計画B3までに「通るが落ちない」
テストが22件出ている）:

- 行列は**非対称**（``max|offdiag| / max|diag|`` を実測してアサートする）。
  対称行列だと転置が見えない。
- ターゲットは**多色**かつ**縦横が違う**。フレーム・有効領域とも縦横を
  違える。単色だと画素の入れ替えが、正方形だと転置が見えない。
- **4クラスの画素数をすべて違える**。負クリップと1超クリップが同数だと、
  クラス1と2の入れ替えが画素数では見えない（画素単位の一致では見えるが、
  片方だけに頼らない）。
- **境界値は実データの既定値と一致する**。分類の境界は 0 と 1 であり、
  それはターゲットとクリップの自然な両端でもある。明らかに範囲外の値しか
  作らないフィクスチャは境界を一度も踏まない。そこでフィクスチャは
  ちょうど 0.0 とちょうど 1.0 を**実際に出す**ように組み、出ていることを
  アサートしている（``_assert_boundary_pixels_exist``）。

フィクスチャの作り: ターゲットの上半分（6行のうち上3行）を黒にしてある。
有効領域の上2行はその黒が縮小されて落ちる場所なので、そこでは
``[0,0,0] @ M + b`` すなわち**バイアス行そのもの**が補償値になり、float の
丸めが一切入らない。この14画素のバイアスを1つずつ置くことで、ちょうど 0.0・
ちょうど 1.0・負のみ・1超のみ・両方を意図した位置に正確に作れる。残りの
21画素は多色ターゲットが行列を通った本物の補償値であり、クラス地図が
ターゲットの中身に依存すること（＝転置や復号方式の取り違えが見えること）を
担保している。
"""

from __future__ import annotations

import base64
import json
from typing import get_args

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp.compensation.pipeline import (  # noqa: E402
    TargetDecodeMode,
    generate_projection_image,
)
from yoshidacomp.config import get_config  # noqa: E402
from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store, targets  # noqa: E402

# The projector frame. Non-square, and neither side equals the other, so a
# transposed (H, W) anywhere lands on a different shape.
_FRAME_WIDTH, _FRAME_HEIGHT = 10, 7

# ``(x, y, w, h)``: off-centre and asymmetric (``x != y``, ``w != h``), so a
# transposed slice in the NaN mask hits a disjoint set of pixels.
# Rows 1-5, columns 2-8.
_REGION = (2, 1, 7, 5)

# The target's own size, different from the frame *and* from the active region
# in both dimensions, so ``embed_in_active_region``'s resize genuinely runs
# (9 -> 7 wide, 6 -> 5 tall, i.e. the INTER_AREA branch).
_TARGET_HEIGHT, _TARGET_WIDTH = 6, 9

# The target's top half is black. After the 6 -> 5 row resize, destination rows
# 0 and 1 draw only on source rows 0-2, so those two rows are exactly 0.0 --
# which is what makes the compensation there exactly the matrix's bias row.
_BLACK_ROWS = 3

# The frame rows that land on the black half of the target. Not derived from
# ``_BLACK_ROWS`` by arithmetic here: the mapping is cv2's, and
# ``test_the_black_band_makes_the_compensation_exactly_the_bias_row`` measures
# it rather than trusting this constant.
_BIAS_ROWS = (1, 2)

# A deliberately non-symmetric affine, scaled so the coloured half of the
# target lands mostly -- but not entirely -- inside [0, 1]. A fixture that
# clipped nowhere could not show that the class map depends on the target at
# all; one that clipped everywhere would hide every other difference.
_A = np.array(
    [
        [1.05, 0.30, -0.18],
        [0.16, 0.95, 0.24],
        [-0.09, 0.22, 1.15],
    ],
    dtype=np.float32,
)
_BIAS = np.array([0.10, 0.07, 0.12], dtype=np.float32)

# Bias rows for the pixels whose target is exactly black, keyed by frame
# ``(y, x)``. There the compensation *is* this triple, exactly, so each line
# below is the compensation value of one pixel and the class it must get.
#
# 黒ターゲットの画素では補償値がこのバイアス行そのものになる（丸め無し）。
# したがって右のコメントがその画素の期待クラスである。
_BIAS_OVERRIDES: dict[tuple[int, int], tuple[float, float, float]] = {
    # --- the closed-interval boundary. These are the pixels that tell `<`
    # from `<=`, and they are the reason this fixture exists at all.
    (1, 2): (0.0, 0.40, 0.70),    # class 0 -- exactly 0.0 in R
    (1, 3): (1.0, 0.40, 0.70),    # class 0 -- exactly 1.0 in R
    (1, 4): (0.0, 1.00, 0.50),    # class 0 -- both bounds at once
    # --- below 0 only
    (1, 5): (-0.30, 0.40, 0.70),  # class 1
    (1, 6): (0.20, -0.15, 0.60),  # class 1 -- in G, not R
    (2, 3): (-0.60, 0.30, 0.90),  # class 1
    (2, 7): (-0.10, 0.50, 0.50),  # class 1
    # --- above 1 only
    (1, 7): (0.20, 1.40, 0.60),   # class 2
    (2, 4): (0.30, 0.50, 1.20),   # class 2 -- in B
    (2, 8): (0.60, 1.10, 0.80),   # class 2
    # --- both
    (1, 8): (-0.25, 1.30, 0.50),  # class 3
    (2, 5): (1.50, -0.40, 0.30),  # class 3 -- above in R, below in G
    # --- plainly inside
    (2, 2): (0.10, 0.20, 0.30),   # class 0
    (2, 6): (0.05, 0.95, 0.45),   # class 0
}

# Class index -> how many pixels of it this fixture produces. All four differ,
# so swapping any two classes changes the histogram and not only the map.
# 4クラスの画素数はすべて違う（入れ替えが画素数だけでも見える）。
_EXPECTED_COUNTS = {0: 24, 1: 4, 2: 5, 3: 2}

_EXPECTED_LABELS = ["範囲内", "負クリップ", "1超クリップ", "両方"]

# The fixture's shape constraints, checked once at import rather than inside a
# test. These compare constants against constants -- they can never fail
# because of anything the view does, so asserting them inside a behaviour test
# only inflates the count of tests that "cover" the view. They belong with the
# fixture they constrain.
# 定数どうしの比較なので、ビューの振る舞いでは絶対に落ちない。振る舞いの
# テストの中に置くと「ビューを検証しているテスト」の数だけが水増しされる。
assert _TARGET_HEIGHT != _TARGET_WIDTH, "a square target hides a transpose"
assert (_FRAME_HEIGHT, _FRAME_WIDTH) != (_TARGET_HEIGHT, _TARGET_WIDTH), (
    "the frame and the target must differ or the resize is an identity"
)
assert (_FRAME_HEIGHT, _FRAME_WIDTH) != (_REGION[3], _REGION[2]), (
    "the frame and the active region must differ or the mask covers everything"
)
assert (_REGION[3], _REGION[2]) != (_TARGET_HEIGHT, _TARGET_WIDTH), (
    "the region and the target must differ or embed_in_active_region's resize "
    "never runs"
)
assert sum(_EXPECTED_COUNTS.values()) == _REGION[2] * _REGION[3]
assert len(set(_EXPECTED_COUNTS.values())) == 4, (
    "two classes have the same pixel count -- swapping them would be invisible "
    "to a histogram"
)


def _matrices(height: int, width: int) -> np.ndarray:
    """A per-pixel-varying, non-symmetric ``(H, W, 4, 3)`` matrix.

    行列を画素ごとに変える。全画素同一だと、ビューが行列を上下反転・転置して
    渡しても同じ答えになる。バイアス行だけは黒帯の画素で差し替える。
    """
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    modulation = (1.0 + 0.004 * (rows + 2.0 * cols)).astype(np.float32)

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, :3, :] = _A * modulation[:, :, None, None]
    matrices[:, :, 3, :] = _BIAS
    for (y, x), bias in _BIAS_OVERRIDES.items():
        matrices[y, x, 3, :] = np.array(bias, dtype=np.float32)
    return matrices


def _target_image(height: int, width: int) -> np.ndarray:
    """A multi-coloured ``(H, W, 3)`` uint8 RGB image with a black top half."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (37 * cols + 11 * rows) % 256
    image[:, :, 1] = (17 * cols + 53 * rows + 90) % 256
    image[:, :, 2] = 255 - (29 * cols + 7 * rows) % 256
    image[:_BLACK_ROWS] = 0
    return image


def _write_target(tmp_path, image: np.ndarray, name: str = "target.png"):
    path = tmp_path / name
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok, f"failed to write test fixture image: {path}"
    return path


def _load_matrix(client, tmp_path, matrices: np.ndarray, name: str = "m.npy") -> str:
    path = tmp_path / name
    np.save(path, matrices)
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _load_target(client, path) -> str:
    response = client.post("/api/target", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _hwc(response) -> np.ndarray:
    """Decode the planar ``(C, H, W)`` body back into ``(H, W, C)``."""
    height = int(response.headers["X-Height"])
    width = int(response.headers["X-Width"])
    channels = int(response.headers["X-Channels"])
    assert response.headers["X-Layout"] == "planar"
    planar = np.frombuffer(response.content, dtype="<f4").reshape(
        channels, height, width
    )
    return np.moveaxis(planar, 0, 2)


def _inside_mask() -> np.ndarray:
    x, y, w, h = _REGION
    mask = np.zeros((_FRAME_HEIGHT, _FRAME_WIDTH), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def _linear(
    target: np.ndarray,
    matrices: np.ndarray,
    *,
    target_decode: str = "srgb",
    target_gamma: float = 2.2,
    clip: bool = False,
) -> np.ndarray:
    """The compensation the CLI would compute, via the real pipeline function.

    ``clip=False`` が既定なのは、それがこのビューの存在理由だからである
    （クリップは、このビューが見せようとしている情報そのものを消す）。
    """
    linear, _ = generate_projection_image(
        target,
        matrices,
        projector=get_config().projector,
        active_region=_REGION,
        linearizer="none",
        lut=None,
        target_decode=target_decode,  # type: ignore[arg-type]
        target_gamma=target_gamma,
        dtype=np.float32,
        clip=clip,
    )
    return linear


def _classify(linear: np.ndarray) -> np.ndarray:
    """The brief's table, restated over a whole frame.

    分類そのものはここで書き下す（期待値は実関数の戻り値から導く）。この式を
    実装から写すと、同じ誤りを写したときに一緒に通ってしまうため、境界の検査
    （``test_pixels_at_exactly_zero_and_exactly_one_are_in_range``）だけは
    この関数を使わずに書いてある。
    """
    below = (linear < 0.0).any(axis=2)
    above = (linear > 1.0).any(axis=2)
    return np.where(
        below & above, 3.0, np.where(above, 2.0, np.where(below, 1.0, 0.0))
    ).astype(np.float32)


def _prepared(tmp_path, stand_in_projector_config):
    """Load the standard fixture matrix and target; return ids and arrays."""
    store.clear()
    targets.clear()
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
    image = _target_image(_TARGET_HEIGHT, _TARGET_WIDTH)
    matrix_id = _load_matrix(client, tmp_path, matrices)
    target_id = _load_target(client, _write_target(tmp_path, image))
    return client, matrix_id, target_id, matrices, image


def _url(matrix_id: str, **params) -> str:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"/api/matrix/{matrix_id}/view/gamut?{query}"


# --------------------------------------------------------------------------
# 1. The classification.
# --------------------------------------------------------------------------


def test_every_pixel_is_classified_from_the_unclipped_compensation(
    tmp_path, stand_in_projector_config
) -> None:
    """クラス地図が実関数の**クリップ前**の戻り値から導いた分類と一致すること。

    4クラスすべてが1枚の中に現れ、かつ画素数がすべて違うフィクスチャで見る。
    画素単位の一致はクラスの入れ替えを捕らえるが、画素数がすべて違うことも
    別に固定しておく（片方だけに頼らない）。
    """
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)[:, :, 0]

    expected = _classify(_linear(image, matrices))
    inside = _inside_mask()
    np.testing.assert_array_equal(data[inside], expected[inside])

    # --- all four classes are present, and no two of them are the same size.
    # (That the four *expected* counts differ is asserted at import; this is
    # the claim that the view actually produced them.)
    counts = {c: int(((data == c) & inside).sum()) for c in range(4)}
    assert counts == _EXPECTED_COUNTS, counts
    assert all(n > 0 for n in counts.values()), counts

    # --- the fixture's own discriminating power, measured rather than assumed.
    a_mat = matrices[3, 3, :3, :]
    asymmetry = float(
        np.max(np.abs(a_mat - np.diag(np.diag(a_mat))))
        / np.max(np.abs(np.diag(a_mat)))
    )
    assert asymmetry > 0.2, (
        f"A is too close to diagonal ({asymmetry:.3f}) for this file to tell a "
        "transposed matrix from the real one; real matrices sit near 0.25"
    )
    assert not np.array_equal(image[:, :, 0], image[:, :, 1]), (
        "the target's channels must differ or a channel swap is invisible"
    )


def test_the_class_map_depends_on_the_target_image_itself(
    tmp_path, stand_in_projector_config
) -> None:
    """ターゲットを転置すると別のクラス地図になること。

    ``embed_in_active_region`` はリサイズして有効領域へ収めるだけなので、
    転置したターゲットを渡してももっともらしいクラス地図が返る。上のテストに
    それを検出する力があるのは、転置版が**別の地図になる**からであり、それは
    フィクスチャの性質であってビューの性質ではない。ここで固定しておく。
    """
    _, _, _, matrices, image = _prepared(tmp_path, stand_in_projector_config)
    straight = _classify(_linear(image, matrices))
    transposed = _classify(
        _linear(np.ascontiguousarray(image.transpose(1, 0, 2)), matrices)
    )
    inside = _inside_mask()
    differing = int((straight != transposed)[inside].sum())
    assert differing >= 2, (
        f"only {differing} pixel(s) change when the target is transposed -- "
        "the class map barely depends on the target's content"
    )


# --------------------------------------------------------------------------
# 2. The boundary. Exactly 0 and exactly 1 are 範囲内.
# --------------------------------------------------------------------------


def _assert_boundary_pixels_exist(linear: np.ndarray, inside: np.ndarray) -> None:
    """Fail unless the fixture really puts a channel on each closed bound.

    This is the single most load-bearing assertion in the file, because
    **nothing outside this fixture ever reaches the boundary.** See
    ``test_pixels_at_exactly_zero_and_exactly_one_are_in_range`` for the
    measurement. If the fixture ever stops constructing the bounds exactly,
    the boundary test does not start failing -- it starts passing vacuously,
    and the closed-interval rule loses its only check anywhere.

    このアサートが無いと、境界を一度も踏まないフィクスチャに縮退しても
    誰も気付けない（境界テストは落ちるのではなく、空虚に通るようになる）。
    実データは境界を踏まないので、本番でも気付けない。
    """
    assert int((linear[inside] == 0.0).sum()) > 0, (
        "no channel is exactly 0.0 inside the region: this file's boundary "
        "test would pass with `<` or `<=` alike"
    )
    assert int((linear[inside] == 1.0).sum()) > 0, (
        "no channel is exactly 1.0 inside the region: this file's boundary "
        "test would pass with `>` or `>=` alike"
    )


def test_pixels_at_exactly_zero_and_exactly_one_are_in_range(
    tmp_path, stand_in_projector_config
) -> None:
    """境界は閉区間。ちょうど 0.0 とちょうど 1.0 は ``範囲内``（クラス0）。

    **この1件が落ちなければ、閉区間の規則はどこでも検査されていない。**
    直感に反するが、実データはこのビューの経路で境界を踏まない。「白飛びが
    1.0 に張り付く」現象を作っているのは ``calc_compensation_image`` の
    クリップであり、``gamut`` は ``clip=False`` でそれを止めているからである。
    実測（実行列4種 × 実写ターゲット3枚 = 12組、1組あたり2,764,800
    チャンネル値。``data/comp_test_20260724`` と ``data/comp_test_targets``）:

    | | ちょうど 0.0 | ちょうど 1.0 |
    | --- | ---: | ---: |
    | ``clip=True``（``compensation`` の経路） | 1,149,991 | 649,022 |
    | ``clip=False``（**このビューの経路**） | **0** | **0**（12組中1組だけ1件） |

    つまり ``<`` を ``<=`` に取り違えても実データでは1画素も誤分類されず、
    GUI を眺めても永久に気付けない。境界を踏ませているのは
    ``_BIAS_OVERRIDES`` の3画素だけである。フィクスチャを緩めてはならない
    （``_assert_boundary_pixels_exist`` がその番人）。

    規則自体は正しい。0.0 と 1.0 はプロジェクタが厳密に再現できる2つの値で
    あり、定義により色域内である。``<=`` にすると「最も暗い／最も明るい
    再現可能な画素」を「再現不能」と報告することになる。

    ここは ``_classify`` を**使わない**。同じ ``np.where`` 連鎖を写すと、
    実装が ``<=`` になったときに期待値も一緒に ``<=`` になり、この1件が
    黙って通ってしまう。境界にある画素を値そのものから選び出し、その画素の
    クラスが 0 であることだけを見る。
    """
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )
    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)[:, :, 0]

    linear = _linear(image, matrices)
    inside = _inside_mask()
    _assert_boundary_pixels_exist(linear, inside)

    within = ((linear >= 0.0) & (linear <= 1.0)).all(axis=2)
    at_zero = (linear == 0.0).any(axis=2) & within & inside
    at_one = (linear == 1.0).any(axis=2) & within & inside
    # Not a restatement of ``_assert_boundary_pixels_exist``: that one allows
    # the boundary channel to sit in a pixel whose *other* channels are out of
    # range, and such a pixel is legitimately class 1/2/3. This is the guard
    # against the two assertions below being vacuously true over an empty
    # selection, which is how a boundary test stops testing the boundary
    # without ever going red.
    # 上の存在確認とは別物。こちらは下の2つが空集合に対して空虚に真になるのを
    # 防ぐためのもので、境界テストが「赤くならないまま検査をやめる」経路を塞ぐ。
    assert at_zero.any() and at_one.any()

    assert (data[at_zero] == 0.0).all(), (
        "a pixel whose channel is exactly 0.0, with every channel inside "
        "[0, 1], was not classified 範囲内 -- the lower bound is closed"
    )
    assert (data[at_one] == 0.0).all(), (
        "a pixel whose channel is exactly 1.0, with every channel inside "
        "[0, 1], was not classified 範囲内 -- the upper bound is closed"
    )

    # The three named fixture pixels, by coordinate, so a failure says which
    # bound moved rather than only that one did.
    # どちらの境界が動いたのかが分かるように、座標を名指しでも見る。
    for (y, x), expected_value in (
        ((1, 2), 0.0),  # exactly 0.0 in R
        ((1, 3), 1.0),  # exactly 1.0 in R
    ):
        assert linear[y, x, 0] == expected_value, (linear[y, x], (y, x))
        assert data[y, x] == 0.0, ((y, x), data[y, x])
    # The pixel that sits on both bounds at once: one `<`/`<=` slip either way
    # moves it out of 範囲内.
    assert linear[1, 4, 0] == 0.0 and linear[1, 4, 1] == 1.0, linear[1, 4]
    assert data[1, 4] == 0.0


def test_the_black_band_makes_the_compensation_exactly_the_bias_row(
    tmp_path, stand_in_projector_config
) -> None:
    """境界の画素が**厳密に** 0.0/1.0 になる仕組みを固定する。

    これはビューではなくフィクスチャの検査である。上の境界テストは、狙った
    画素の補償値が丸め無しでバイアス行そのものになることに乗っている。
    cv2 のリサイズが変わる・黒帯の行数を触るなどでその前提が崩れると、境界
    テストは「境界にある画素が1つも無いまま」通りかねない。前提が崩れたら
    ここが落ちる。
    """
    _, _, _, matrices, image = _prepared(tmp_path, stand_in_projector_config)
    linear = _linear(image, matrices)
    x, _, w, _ = _REGION
    for y in _BIAS_ROWS:
        for column in range(x, x + w):
            np.testing.assert_array_equal(
                linear[y, column],
                matrices[y, column, 3, :],
                err_msg=f"pixel {(y, column)} is not exactly the bias row",
            )
    # And the rest of the region is not the bias row, or the "coloured" half
    # would be contributing nothing.
    coloured = [
        y for y in range(_REGION[1], _REGION[1] + _REGION[3]) if y not in _BIAS_ROWS
    ]
    assert coloured
    assert not np.allclose(
        linear[coloured, x : x + w], matrices[coloured, x : x + w, 3, :]
    )


# --------------------------------------------------------------------------
# 3. Outside the active region.
# --------------------------------------------------------------------------


def test_outside_the_active_region_is_nan_not_class_zero(
    tmp_path, stand_in_projector_config
) -> None:
    """有効領域の外は NaN であってクラス番号 0 ではない。

    NaN の比較は常に False なので、マスクを分類の**前**に置くと領域外は
    「どのチャンネルも 0 未満でも 1 超でもない」＝``範囲内`` に落ちる。
    もっともらしく、しかも最も安心できる誤答である。

    ``(x, y, w, h)`` は非対称なので、スライスの x/y や w/h を取り違えると
    別の画素集合に着地し、下の集合一致で落ちる。
    """
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)[:, :, 0]

    inside = _inside_mask()
    assert np.array_equal(np.isfinite(data), inside), (
        "the NaN mask does not match (x, y, w, h) as configured -- check for a "
        "transposed x/y or w/h in the mask slice"
    )

    x, y, w, h = _REGION
    assert np.isfinite(data[y, x])   # inside: the region's corner
    assert np.isnan(data[0, 0])      # outside: the frame's corner
    # The point a transposed (x, y) reading would have kept instead.
    assert np.isnan(data[x, y])

    # What makes the NaN a real difference rather than a restatement: the
    # pipeline writes 0.0 outside the region (mask_to_active_region), and 0.0
    # classifies as 範囲内. Masking before classifying would paint exactly
    # that, over three quarters of this frame.
    # パイプライン自身は領域外に 0.0 を書き、0.0 は ``範囲内`` に分類される。
    linear = _linear(image, matrices)
    assert (linear[~inside] == 0.0).all()
    assert (_classify(linear)[~inside] == 0.0).all()


def test_a_non_finite_compensation_inside_the_region_is_nan_not_in_range(
    tmp_path, stand_in_projector_config
) -> None:
    """有効領域の**内側**の非有限値もクラス0にしてはならない。

    領域外について同ファイルが名指しで警戒している「最も安心できる誤答」は、
    領域内でも同じ形で起こる: NaN との比較はすべて False なので、``< 0`` も
    ``> 1`` も成立せず、そのまま ``範囲内``（＝補償できている）と描かれる。

    経路は実在する。``calc_compensation_image`` は前進積なので有限入力から
    NaN を作れないが、行列は ``POST /api/session/matrix`` が指した
    ``.npy`` そのものであり、係数の NaN/inf はそのまま伝播する。

    ``+inf`` も NaN と同じ扱いにする。``+inf > 1`` は「1超クリップ」に落ちる
    が、それは測定値の無い画素の測定値であり、``surface`` が特異な画素を NaN
    にするのと同じ理由で答えにしてはならない。

    黒帯の画素を使うのは、そこでは補償値がバイアス行そのものになり、
    どの画素が汚染されるかを厳密に決められるからである。
    """
    store.clear()
    targets.clear()
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
    # (1, 2) was the exactly-0.0 boundary pixel and (1, 7) was class 2.
    matrices[1, 2, 3, 0] = np.nan
    matrices[1, 7, 3, 1] = np.inf
    image = _target_image(_TARGET_HEIGHT, _TARGET_WIDTH)
    matrix_id = _load_matrix(client, tmp_path, matrices, name="non_finite.npy")
    target_id = _load_target(client, _write_target(tmp_path, image))

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)[:, :, 0]

    # The contamination really did reach the compensation, and only there.
    linear = _linear(image, matrices)
    assert np.isnan(linear[1, 2, 0]), linear[1, 2]
    assert np.isinf(linear[1, 7, 1]), linear[1, 7]

    assert np.isnan(data[1, 2]), (
        "a NaN compensation was classified as 範囲内 -- every comparison "
        "against NaN is False, so it fell through to class 0"
    )
    assert np.isnan(data[1, 7]), (
        "an infinite compensation was classified as 1超クリップ -- inf > 1 is "
        "True, so it reads as a measurement of a pixel that has none"
    )

    # Every other in-region pixel is still classified normally: the guard must
    # narrow to the contaminated pixels, not blank the region.
    inside = _inside_mask()
    untouched = inside.copy()
    untouched[1, 2] = False
    untouched[1, 7] = False
    assert np.isfinite(data[untouched]).all()
    assert int((data[untouched] != 0.0).sum()) > 0


# --------------------------------------------------------------------------
# 4. clip=False is the whole point.
# --------------------------------------------------------------------------


def test_the_view_classifies_the_unclipped_values_not_the_clipped_ones(
    tmp_path, stand_in_projector_config
) -> None:
    """``clip=True`` なら色域外は1画素も残らない。ビューはそちらではない。

    ``generate_projection_image`` の既定は ``clip=True`` であり、それは
    このビューが見せようとしている情報そのものを消す。前半はフィクスチャの
    性質（クリップすれば全画素がクラス0になる）を、後半はビューが実際に
    どちらを見ているかを固定する。名前のとおりビューを呼ぶ -- 呼ばないと
    ``clip`` を冠したテストが ``clip`` の変異で落ちないことになる。
    """
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )
    inside = _inside_mask()

    clipped = _classify(_linear(image, matrices, clip=True))
    unclipped = _classify(_linear(image, matrices, clip=False))

    assert (clipped[inside] == 0.0).all(), (
        "clipping left an out-of-range value behind; this test cannot detect "
        "clip=True in the view"
    )
    assert int((unclipped[inside] != 0.0).sum()) == sum(
        _EXPECTED_COUNTS[c] for c in (1, 2, 3)
    )

    # ...and the view agrees with the unclipped one, not the clipped one.
    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)[:, :, 0]
    np.testing.assert_array_equal(data[inside], unclipped[inside])
    assert not np.array_equal(data[inside], clipped[inside]), (
        "the view returned the clipped classification: every out-of-gamut "
        "pixel it exists to show has been rounded away"
    )


def _uniform_class_matrices(height: int, width: int, bias) -> np.ndarray:
    """A matrix mild enough that *bias* alone decides the class, everywhere.

    ``_matrices`` に対する第2・第3のフィクスチャ。``_A`` を 0.4 倍に縮めて
    ``_BIAS_OVERRIDES`` を適用しないので、有効領域35画素すべてが同じクラスに
    なる。どのクラスになるかは *bias* だけで決まる。
    """
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    modulation = (1.0 + 0.004 * (rows + 2.0 * cols)).astype(np.float32)

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, :3, :] = (_A * 0.4) * modulation[:, :, None, None]
    matrices[:, :, 3, :] = np.array(bias, dtype=np.float32)
    return matrices


# ``(name, bias, the single class every in-region pixel gets)``.
#
# 2枚要る理由は下のテストの docstring にある。片方だけでは、レンジの片端しか
# データ由来の実装と区別できない。
_UNIFORM_FRAMES = [
    ("in_gamut", (0.20, 0.30, 0.25), 0.0),
    ("all_below_zero", (-0.60, 0.20, 0.25), 1.0),
]


@pytest.mark.parametrize("name, bias, expected_class", _UNIFORM_FRAMES)
def test_the_display_range_is_the_class_range_not_the_frame_s_own(
    tmp_path, stand_in_projector_config, name: str, bias, expected_class: float
) -> None:
    """1クラスしか出ないフレームでも ``vmin``/``vmax`` は 0.0/3.0 のまま。

    **主フィクスチャではこの主張を検証できない。** あちらは有効領域35画素に
    4クラス全部を含むため ``nanmin(data) == 0.0``、``nanmax(data) == 3.0``、
    ``nanpercentile(data, [1, 99]) == [0.0, 3.0]`` となり、クラス番号の範囲と
    データ自身の min/max が**たまたま一致する**。実装をデータ由来のレンジに
    変えても主フィクスチャは1件も落ちない（レビューが実測した「23件目」）。

    2枚とも要る。片方では片端しか動かない:

    - ``in_gamut``（全画素クラス0）: データ由来だと ``vmax`` が 0.0 に潰れる。
      これが実際に危ないケースで、``routers/view.py`` の categorical 規則が
      ``vmin == vmax`` を 500 で弾くため、「全部表現できています」という最も
      出したい答えのときだけビューが壊れる。ただし ``vmin`` の側は 0.0 の
      ままなので、``vmin`` だけをデータ由来にする退行はここでは見えない。
    - ``all_below_zero``（全画素クラス1）: ``nanmin`` が 1.0 になるので
      ``vmin`` 側の退行がここで見える。``vmax`` も 1.0 に潰れる。

    どちらのフレームも特殊ではない。暗めのターゲットは前者を、環境光の強い
    実測行列は後者に近いものを普通に生む（実データ12組で色域外は47〜85%で
    あり、「全部範囲内が普通」も「全部色域外が普通」も正しくない）。
    """
    store.clear()
    targets.clear()
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
    matrices = _uniform_class_matrices(_FRAME_HEIGHT, _FRAME_WIDTH, bias)
    image = _target_image(_TARGET_HEIGHT, _TARGET_WIDTH)
    matrix_id = _load_matrix(client, tmp_path, matrices, name=f"{name}.npy")
    target_id = _load_target(client, _write_target(tmp_path, image))

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)[:, :, 0]
    inside = _inside_mask()

    # The frame really is uniform -- measured from the view's own output, not
    # assumed, because a fixture that quietly grew a second class would make
    # the range assertions below coincidental again.
    assert (data[inside] == expected_class).all(), (
        f"{name} is meant to be entirely class {expected_class}, got "
        f"{sorted(set(data[inside].tolist()))}"
    )

    # ...so the data's own extent is a single point, and is not the class range.
    assert float(np.nanmin(data)) == expected_class
    assert float(np.nanmax(data)) == expected_class

    assert float(response.headers["X-Vmin"]) == 0.0
    assert float(response.headers["X-Vmax"]) == 3.0
    assert float(response.headers["X-Vmax"]) == len(_EXPECTED_LABELS) - 1


# --------------------------------------------------------------------------
# 5. The parameters this view has -- and the ones it deliberately has not.
# --------------------------------------------------------------------------


def test_target_decode_changes_the_classification(
    tmp_path, stand_in_projector_config
) -> None:
    """``target_decode`` が効くこと。既定は ``srgb``。"""
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )
    inside = _inside_mask()

    maps = {}
    for mode in ("srgb", "none"):
        response = client.get(_url(matrix_id, target=target_id, target_decode=mode))
        assert response.status_code == 200, response.text
        maps[mode] = _hwc(response)[:, :, 0]
        np.testing.assert_array_equal(
            maps[mode][inside],
            _classify(_linear(image, matrices, target_decode=mode))[inside],
        )
    differing = int((maps["srgb"] != maps["none"])[inside].sum())
    assert differing >= 2, f"only {differing} pixel(s) changed with the decode"

    default = _hwc(client.get(_url(matrix_id, target=target_id)))[:, :, 0]
    np.testing.assert_array_equal(default[inside], maps["srgb"][inside])


def test_target_gamma_is_honoured_when_decoding_with_gamma(
    tmp_path, stand_in_projector_config
) -> None:
    """``target_gamma`` が ``target_decode="gamma"`` のときに効くこと。"""
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )
    inside = _inside_mask()

    def _fetch(gamma: float) -> np.ndarray:
        response = client.get(
            _url(
                matrix_id,
                target=target_id,
                target_decode="gamma",
                target_gamma=gamma,
            )
        )
        assert response.status_code == 200, response.text
        return _hwc(response)[:, :, 0]

    default_gamma = _fetch(2.2)
    other_gamma = _fetch(1.0)
    np.testing.assert_array_equal(
        default_gamma[inside],
        _classify(
            _linear(image, matrices, target_decode="gamma", target_gamma=2.2)
        )[inside],
    )
    differing = int((default_gamma != other_gamma)[inside].sum())
    assert differing >= 2, f"only {differing} pixel(s) changed with the exponent"


@pytest.mark.parametrize("bad_gamma", ["0", "-2.2", "nan"])
def test_a_non_positive_target_gamma_is_refused(
    tmp_path, stand_in_projector_config, bad_gamma: str
) -> None:
    """``target_gamma <= 0`` は入力の誤りであってサーバのバグではない。"""
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(
        _url(
            matrix_id,
            target=target_id,
            target_decode="gamma",
            target_gamma=bad_gamma,
        )
    )
    assert response.status_code == 400, response.text
    assert "target_gamma" in response.json()["detail"]


def test_the_view_is_listed_with_its_params_and_without_the_others() -> None:
    """``/api/views`` の宣言。

    ``linearizer``/``lut``/``output`` を**持たない**ことが宣言の要点である。
    色域外かどうかは線形補償値の性質であり、その後の線形化とは無関係なので、
    それらを持たせると「LUTを変えると色域外が変わる」という誤った印象を与える。
    """
    store.clear()
    targets.clear()  # every other test in this file clears both, via _prepared
    client = TestClient(create_app())
    views = client.get("/api/views").json()["views"]
    spec = next(v for v in views if v["key"] == "gamut")

    assert spec["kind"] == "categorical"
    assert spec["channels"] == 1
    assert spec["values_are_matrix"] is False

    params = {p["name"]: p for p in spec["params"]}
    assert set(params) == {"target", "target_decode", "target_gamma"}
    for absent in ("linearizer", "lut", "output"):
        assert absent not in params, (
            f"{absent} would suggest the out-of-gamut classification depends "
            "on linearization, which it does not"
        )

    assert params["target"]["type"] == "text"
    assert params["target"]["choices"] == []
    assert params["target"]["default"] == ""
    assert set(params["target_decode"]["choices"]) == set(get_args(TargetDecodeMode))
    assert params["target_decode"]["default"] == "srgb"
    assert params["target_gamma"]["type"] == "number"
    assert params["target_gamma"]["default"] == 2.2

    for p in spec["params"]:
        if p["type"] == "select":
            assert p["default"] in p["choices"], p["name"]


# --------------------------------------------------------------------------
# 6. Refusals.
# --------------------------------------------------------------------------


def test_a_camera_space_matrix_is_refused(tmp_path, stand_in_projector_config) -> None:
    """カメラ座標の行列は ``ViewNotApplicable``（router 経由で 400）。

    ``compensation`` と同じ理由で断る。カメラ座標では ``active_region`` が
    ``None`` になり、``generate_projection_image`` に渡す領域が無い。
    """
    store.clear()
    targets.clear()
    stand_in_projector_config(width=64, height=48)
    client = TestClient(create_app())
    matrix_id = _load_matrix(client, tmp_path, _matrices(_FRAME_HEIGHT, _FRAME_WIDTH))
    target_id = _load_target(
        client, _write_target(tmp_path, _target_image(_TARGET_HEIGHT, _TARGET_WIDTH))
    )
    assert client.get(f"/api/matrix/{matrix_id}/meta").json()["space"] == "camera"

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "camera" in detail
    assert "gamut" in detail
    assert "B4" in detail or "計画" in detail

    # The refusal belongs to this view, not to camera-space matrices at large.
    assert client.get(f"/api/matrix/{matrix_id}/view/coef_tiles").status_code == 200


def test_an_unknown_target_id_is_refused(tmp_path, stand_in_projector_config) -> None:
    client, matrix_id, _, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target="target-does-not-exist"))
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "target-does-not-exist" in detail
    assert "gamut" in detail


def test_a_missing_target_is_refused(tmp_path, stand_in_projector_config) -> None:
    """未指定・空欄・空白のみはすべて同じ「未指定」であること。"""
    client, matrix_id, _, _, _ = _prepared(tmp_path, stand_in_projector_config)
    for url in (
        f"/api/matrix/{matrix_id}/view/gamut",
        _url(matrix_id, target=""),
        _url(matrix_id, target="%20%20"),
    ):
        response = client.get(url)
        assert response.status_code == 400, (url, response.text)
        detail = response.json()["detail"]
        assert "target" in detail
        assert "POST /api/target" in detail, (url, detail)
        assert "no target image with id" not in detail, (url, detail)


def test_a_matrix_loaded_under_another_projector_config_is_refused_not_500(
    tmp_path, stand_in_projector_config
) -> None:
    """読み込み後に config が変わった行列は 400 で断る（500 にしない）。

    ``compensation`` と同じ理由。``MatrixContext.space`` は読み込み時の判定で
    あり、フレームが食い違ったまま進むと numpy の形状 ``ValueError``（＝500）
    になる。これが ``_render`` の ``classify_space`` 再判定を検証している。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    assert client.get(f"/api/matrix/{matrix_id}/meta").json()["space"] == "projector"
    assert client.get(_url(matrix_id, target=target_id)).status_code == 200

    stand_in_projector_config(width=_FRAME_WIDTH + 2, height=_FRAME_HEIGHT + 1)
    assert client.get(f"/api/matrix/{matrix_id}/meta").json()["space"] == "projector"

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 400, response.text
    assert "gamut" in response.json()["detail"]


# --------------------------------------------------------------------------
# 7. Shape, dtype, labels, display range.
# --------------------------------------------------------------------------


def test_the_body_is_one_float32_channel_of_class_indices(
    tmp_path, stand_in_projector_config
) -> None:
    """``(H, W, 1) float32``、大きさは投影フレーム。値はクラス番号。"""
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    assert response.headers["X-Kind"] == "categorical"
    assert response.headers["X-Channels"] == "1"
    assert int(response.headers["X-Height"]) == _FRAME_HEIGHT
    assert int(response.headers["X-Width"]) == _FRAME_WIDTH
    assert (_FRAME_HEIGHT, _FRAME_WIDTH) != (_TARGET_HEIGHT, _TARGET_WIDTH)
    assert (_FRAME_HEIGHT, _FRAME_WIDTH) != (_REGION[3], _REGION[2])

    data = _hwc(response)
    assert data.shape == (_FRAME_HEIGHT, _FRAME_WIDTH, 1)
    assert data.dtype == np.float32
    finite = data[np.isfinite(data)]
    np.testing.assert_array_equal(finite, np.rint(finite))
    assert finite.min() >= 0.0 and finite.max() <= 3.0


def test_the_four_labels_are_in_class_order(
    tmp_path, stand_in_projector_config
) -> None:
    """``labels`` は4要素で、並びがクラス番号と一致すること。

    ``categorical`` のラベルはチャンネル名ではなく**クラス名**である。日本語が
    そのままヘッダに載ると latin-1 で 500 になるので、base64 で往復することも
    ここで確かめる（``routers/view.py::encode_labels``）。
    """
    from yoshidacomp_web.views.gamut import LABELS

    assert LABELS == _EXPECTED_LABELS
    assert len(LABELS) == 4

    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    labels = json.loads(
        base64.b64decode(response.headers["X-Labels-B64"]).decode("utf-8")
    )
    assert labels == _EXPECTED_LABELS


def test_the_display_range_is_the_class_index_range(
    tmp_path, stand_in_projector_config
) -> None:
    """主フィクスチャでの ``vmin``/``vmax`` が 0.0/3.0 であること。

    **この1件は「レンジがデータ由来でない」ことを検証していない。** 主
    フィクスチャは4クラス全部を含むので ``nanmin``/``nanmax``/1-99パーセンタイル
    のいずれもちょうど 0.0/3.0 になり、データ由来の実装と直書きの実装を
    区別できない（レビューが3種の変異で実測し、いずれも全件緑だった）。
    区別しているのは
    ``test_the_display_range_is_the_class_range_not_the_frame_s_own`` の
    1クラスだけのフレーム2枚である。

    ここに残す価値は2つある: 主フィクスチャ（＝このファイルの他のすべての
    テストが使う経路）でヘッダが実際に 0.0/3.0 で出ること、そして ``LABELS``
    の要素数とレンジの結合（4→3 に減らすとここも落ちる）。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    assert float(response.headers["X-Vmin"]) == 0.0
    assert float(response.headers["X-Vmax"]) == 3.0
    assert float(response.headers["X-Vmax"]) == len(_EXPECTED_LABELS) - 1


def test_the_view_result_itself_is_hw1_float32(
    tmp_path, stand_in_projector_config
) -> None:
    """``ViewResult`` を直接見る。

    router を通すと dtype はワイヤ形式に寄せられるため、``(H, W, 1) float32``
    という契約はここでしか見えない。直接呼び出しは関数内フォールバック
    （既定値の二重定義のずれ）が見える唯一の場所でもある。
    """
    from yoshidacomp_web.matrix import load_matrix
    from yoshidacomp_web.views import REGISTRY

    _, _, target_id, matrices, image = _prepared(tmp_path, stand_in_projector_config)
    context = load_matrix(tmp_path / "m.npy", get_config().projector)
    result = REGISTRY["gamut"].render(context, {"target": target_id})

    assert result.data.shape == (_FRAME_HEIGHT, _FRAME_WIDTH, 1)
    assert result.data.dtype == np.float32
    assert result.labels == _EXPECTED_LABELS
    assert (result.vmin, result.vmax) == (0.0, 3.0)

    inside = _inside_mask()
    np.testing.assert_array_equal(
        result.data[:, :, 0][inside],
        _classify(_linear(image, matrices, target_decode="srgb", target_gamma=2.2))[
            inside
        ],
    )
