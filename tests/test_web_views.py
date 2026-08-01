"""View registry and the coefficient-tile view."""

from __future__ import annotations

import base64
import contextlib
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402
from yoshidacomp_web.views.base import (  # noqa: E402
    REGISTRY,
    ParamSpec,
    ViewNotApplicable,
    ViewResult,
    ViewSpec,
    register,
)
from yoshidacomp_web.views.coef_tiles import LABELS, SPEC  # noqa: E402

# Every test here loads its own stand-in projector config
# (``stand_in_projector_config``, tests/conftest.py) rather than relying on the
# developer's live config.toml: whether a 6x8 matrix classifies as "camera"
# space -- and therefore whether the active-region NaN mask runs at all -- is
# decided by that file, which is edited constantly. The autouse fixture in
# conftest.py still restores the real config afterwards.
# 6x8 の行列が "camera" と判定されるか（＝NaNマスクが走るか）は config.toml
# 次第なので、各テストは代用configを自前で読み込む。

# A stand-in frame that is neither 6x8 nor its transpose, so the small test
# matrices below always classify as camera space.
_CAMERA_SAFE_FRAME = {"width": 64, "height": 48}


def _load(client, tmp_path, *, height: int, width: int):
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)[None, :]
    # 空間的に変化させる。一様だと座標の取り違えを検出できない。
    matrices[:, :, 0, 0] = xs
    matrices[:, :, 3, 1] = -1.0
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    response = client.post("/api/session/matrix", json={"path": str(path)})
    return response.json()["id"]


def _decode_labels(response) -> list[str]:
    """Decode ``X-Labels-B64`` exactly the way ``static/js/api.js`` does."""
    return json.loads(base64.b64decode(response.headers["X-Labels-B64"]).decode("utf-8"))


@contextlib.contextmanager
def _temporary_view(key: str, labels: list[str], *, channels=None, params=None, kind="scalar"):
    """Register a throwaway view for the duration of a ``with`` block.

    Registering rather than adding a real view keeps these contract tests
    independent of which views happen to ship.
    実ビューを増やさずに契約だけを検証するため、使い捨てビューを登録する。
    """
    declared = len(labels) if channels is None else channels

    def _render(context, params_: dict) -> ViewResult:
        height, width = context.array.shape[0], context.array.shape[1]
        return ViewResult(
            data=np.zeros((height, width, len(labels)), dtype=np.float32),
            vmin=0.0,
            vmax=1.0,
            labels=labels,
        )

    spec = ViewSpec(
        key=key,
        label="test-only",
        kind=kind,
        channels=declared,
        params=list(params or []),
        render=_render,
    )
    register(spec)
    try:
        yield spec
    finally:
        REGISTRY.pop(key, None)


@contextlib.contextmanager
def _raising_view(key: str, exception: Exception):
    """Register a throwaway view whose render raises *exception*."""

    def _render(context, params_: dict) -> ViewResult:
        raise exception

    spec = ViewSpec(
        key=key,
        label="test-only",
        kind="scalar",
        channels=1,
        params=[],
        render=_render,
    )
    register(spec)
    try:
        yield spec
    finally:
        REGISTRY.pop(key, None)


def test_a_view_refusing_its_input_is_a_400_and_a_bug_in_one_is_a_500(
    tmp_path, stand_in_projector_config
) -> None:
    """``ViewNotApplicable`` means 400; every other exception stays a 500.

    The two are not the same event and must not get the same status. A view
    that declines the matrix it was handed is answering the request -- the
    client asked for something this view does not do, and the message says
    which matrix to load instead. A view that raises anything else has a bug,
    and a bug is not the client's fault.

    The distinction is easy to lose because views are numpy code and numpy
    raises ``ValueError`` for a broadcast or reshape mismatch. Catching
    ``ValueError`` at the router -- which is what shipped first -- turned every
    such bug into a 400 telling the user their request was wrong, with a raw
    numpy message as the explanation, and made it a 4xx in whatever logs or
    monitoring watch the difference.

    「対象外」と「壊れた」は別の出来事であり、同じステータスにしてはならない。
    前者は要求への回答である（このビューはその入力を扱わない、と述べ、どの行列を
    読めばよいかを示す）。後者はバグであり、利用者の落ち度ではない。numpy は
    形状の不整合にも ``ValueError`` を使うため、ルータで ``ValueError`` を
    受けるとバグまで400になり、「あなたの要求が誤りです」と言いながら numpy の
    生メッセージを説明として渡すことになる。監視上も 4xx に化ける。
    """
    stand_in_projector_config(**_CAMERA_SAFE_FRAME)
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)

    refusal = ViewNotApplicable("this view does not accept that matrix")
    with _raising_view("refusing_view", refusal):
        response = client.get(f"/api/matrix/{matrix_id}/view/refusing_view")
    assert response.status_code == 400
    assert response.json()["detail"] == "this view does not accept that matrix"

    # A shape mistake inside a view is a server bug. It must not be dressed up
    # as a client error, so it is allowed to propagate.
    # ビュー内部の形状ミスはサーバのバグである。クライアントエラーに化けさせない。
    bug = ValueError("operands could not be broadcast together with shapes (3,) (4,)")
    with _raising_view("buggy_view", bug):
        with pytest.raises(ValueError, match="could not be broadcast"):
            client.get(f"/api/matrix/{matrix_id}/view/buggy_view")


def test_views_endpoint_lists_coef_tiles() -> None:
    store.clear()
    client = TestClient(create_app())
    body = client.get("/api/views").json()
    keys = {view["key"] for view in body["views"]}
    assert "coef_tiles" in keys
    spec = next(v for v in body["views"] if v["key"] == "coef_tiles")
    assert spec["kind"] == "scalar"
    assert spec["channels"] == 12


def test_coef_tiles_returns_float32_binary(tmp_path, stand_in_projector_config) -> None:
    stand_in_projector_config(**_CAMERA_SAFE_FRAME)
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Height"] == "6"
    assert response.headers["X-Width"] == "8"
    assert response.headers["X-Channels"] == "12"
    assert response.headers["X-Kind"] == "scalar"

    array = np.frombuffer(response.content, dtype="<f4").reshape(12, 6, 8)
    # 本体は planar (C, H, W)。チャンネル順は (i, j) の行優先:
    # 0=in_R->out_R, 1=in_R->out_G, ... 11=bias->out_B
    assert np.allclose(array[0], np.arange(8, dtype=np.float32)[None, :])
    assert np.allclose(array[10], -1.0)


def test_view_body_is_planar_and_declares_its_layout(tmp_path, stand_in_projector_config) -> None:
    """本体は (C, H, W)。ヘッダ X-Layout がそれを申告する。

    インタリーブ (H, W, C) だとクライアントがチャンネルを取り出すのに
    ストライドコピーが要る。実測ではGPUアップロード時間の93%がこの
    デインターリーブだった。planar なら subarray でゼロコピーになる。
    """
    import numpy as np

    stand_in_projector_config(width=8, height=6)
    client = TestClient(create_app())

    matrices = np.zeros((6, 8, 4, 3), dtype=np.float32)
    xs = np.arange(8, dtype=np.float32)[None, :]
    ys = np.arange(6, dtype=np.float32)[:, None]
    matrices[:, :, 0, 0] = xs     # channel 0
    matrices[:, :, 3, 1] = ys     # channel 10
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    matrix_id = client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200
    assert response.headers["X-Layout"] == "planar"
    assert response.headers["X-Channels"] == "12"

    planar = np.frombuffer(response.content, dtype="<f4").reshape(12, 6, 8)
    np.testing.assert_allclose(planar[0], np.broadcast_to(xs, (6, 8)))
    np.testing.assert_allclose(planar[10], np.broadcast_to(ys, (6, 8)))
    # 転置していないことの確認: 幅8・高さ6の非正方なので取り違えれば形が合わない
    assert planar.shape == (12, 6, 8)


def test_coef_tiles_label_format_is_the_4x3_grid_contract(
    tmp_path, stand_in_projector_config
) -> None:
    """``"<入力> -> <出力>"`` is what lays the 12 tiles out as a 4x3 matrix.

    ``static/js/main.js`` (``axesFromLabels``) splits every label on ``"->"``
    and derives the row and column axes from the two halves. A label with a
    different separator, or with more than one ``"->"``, silently falls through
    to ``fallbackAxes`` and collapses the grid to a single row of 12 -- the
    view still renders, so nothing fails, it just stops looking like a matrix.
    Nothing else on either side of the wire pinned this format, which is why it
    is pinned here, at the point where the labels are produced, and again over
    the wire so an encoding change cannot quietly reshape them.

    ラベルの ``"<入力> -> <出力>"`` 形式が4x3グリッドの唯一の根拠であり、
    崩れても描画は成功して1行12枚になるだけなので気付けない。生成箇所と
    ヘッダ経由の両方で形式を固定する。
    """
    assert LABELS == [
        "in_R -> out_R",
        "in_R -> out_G",
        "in_R -> out_B",
        "in_G -> out_R",
        "in_G -> out_G",
        "in_G -> out_B",
        "in_B -> out_R",
        "in_B -> out_G",
        "in_B -> out_B",
        "bias -> out_R",
        "bias -> out_G",
        "bias -> out_B",
    ]
    assert len(LABELS) == SPEC.channels
    for label in LABELS:
        assert label.count("->") == 1, label
        left, right = label.split("->")
        assert left.strip() and right.strip(), label

    stand_in_projector_config(**_CAMERA_SAFE_FRAME)
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)
    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert _decode_labels(response) == LABELS


def test_labels_survive_non_ascii_and_embedded_commas(
    tmp_path, stand_in_projector_config
) -> None:
    """Header labels must round-trip Japanese text and commas untouched.

    Two independent failures of the old comma-joined ``X-Labels`` header:

    - Starlette encodes header values as latin-1, so a Japanese label -- e.g.
      the planned ``gamut`` view's 「色域外」 -- raised on the way out and the
      whole response became a 500. This codebase's UI strings are Japanese
      already (``coef_tiles`` is labelled 「12係数タイル」), so this was the
      next view's problem, not a remote one.
    - A label containing a comma split into two on the client, yielding more
      labels than channels and shifting every tile caption after it.

    latin-1 のヘッダに日本語ラベルを載せると500になり、カンマ入りラベルは
    分割されて以降のタイル名がすべてずれる。base64+UTF-8 JSON で両方を防ぐ。
    """
    labels = ["色域外", "範囲内", "a, b", "in_R -> out_R"]
    with _temporary_view("_test_labels", labels):
        stand_in_projector_config(**_CAMERA_SAFE_FRAME)
        store.clear()
        client = TestClient(create_app())
        matrix_id = _load(client, tmp_path, height=6, width=8)

        response = client.get(f"/api/matrix/{matrix_id}/view/_test_labels")
        assert response.status_code == 200
        assert _decode_labels(response) == labels
        # The comma-joined form is gone: nothing may fall back to splitting.
        assert "X-Labels" not in response.headers


def test_declared_channels_must_match_the_rendered_array(
    tmp_path, stand_in_projector_config
) -> None:
    """``/api/views`` advertises ``spec.channels``; the array must agree.

    The frontend builds its tiles and its hover readout from the advertised
    count, so a view that declares 99 and renders 3 would have produced a
    200 OK and a silently wrong grid.
    宣言チャンネル数と実際の配列が食い違えば200のまま表示が壊れるため、
    サーバ側で明示的に失敗させる。
    """
    with _temporary_view("_test_channel_mismatch", ["a", "b", "c"], channels=99):
        stand_in_projector_config(**_CAMERA_SAFE_FRAME)
        store.clear()
        client = TestClient(create_app(), raise_server_exceptions=False)
        matrix_id = _load(client, tmp_path, height=6, width=8)

        response = client.get(f"/api/matrix/{matrix_id}/view/_test_channel_mismatch")
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "99" in detail and "3" in detail


def test_unknown_view_is_404(tmp_path, stand_in_projector_config) -> None:
    stand_in_projector_config(**_CAMERA_SAFE_FRAME)
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)
    assert client.get(f"/api/matrix/{matrix_id}/view/nosuch").status_code == 404


def test_coef_tiles_masks_outside_active_region(tmp_path, stand_in_projector_config) -> None:
    """The NaN mask must honour ``(x, y, w, h)`` as-is, not a transposed read.

    Every other test matrix in this file is 6x8 against a 64x48 stand-in
    frame, which ``classify_space`` puts in "camera" space, so
    ``active_region`` is ``None`` and the masking branch in
    ``coef_tiles._render`` never runs -- a transposed ``x``/``y`` (or
    ``w``/``h``) in ``mask[y : y + h, x : x + w]``, or a dropped ``.copy()``
    before ``data[~mask] = np.nan``, would go undetected.

    This test asks ``stand_in_projector_config`` for a small frame (rather
    than allocating a real 1920x1080x12 float array) so the matrix classifies
    as "projector" and ``active_region`` is set. The frame is non-square
    (height != width) and the active rectangle sits off-centre near a
    corner with unequal width/height, so a row/column transposition lands
    on a disjoint set of pixels: the full finite/NaN mask is compared
    against the expected rectangle with ``np.array_equal``, and two single
    points are checked explicitly -- one inside the true rectangle, one
    inside where a swapped reading would have placed it instead.

    既存の行列はすべて6x8で "camera" 空間に分類され、``active_region`` は
    ``None`` のままマスク分岐が一度も実行されない。``x``/``y``（あるいは
    ``w``/``h``）の取り違えや ``.copy()`` の欠落は検出されない。実寸の
    1920x1080x12 floatを確保する代わりに、小さな代用プロジェクタconfigを
    読み込んで "projector" 空間に分類させる。縦横比が非対称かつ矩形も
    非対称・非中央に置くことで、行列転置バグが別の画素集合に着地し、
    このテストが確実に失敗するようにしている。
    """
    height, width = 6, 10
    x, y, w, h = 3, 1, 4, 2  # active rect occupies rows 1-2, cols 3-6.

    stand_in_projector_config(
        width=width,
        height=height,
        active_x=x,
        active_y=y,
        active_width=w,
        active_height=h,
    )

    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=height, width=width)

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200

    data = np.frombuffer(response.content, dtype="<f4").reshape(12, height, width)
    finite_mask = np.all(np.isfinite(data), axis=0)

    expected_mask = np.zeros((height, width), dtype=bool)
    expected_mask[y : y + h, x : x + w] = True
    assert np.array_equal(finite_mask, expected_mask), (
        "active-region mask does not match (x, y, w, h) as configured -- "
        "check for a transposed x/y or w/h in the mask slice"
    )

    # A point inside the true rectangle (row=y, col=x) must be finite; the
    # point a transposed reading (row=x, col=y) would have kept instead must
    # be NaN, since x != y and w != h here.
    assert np.all(np.isfinite(data[:, y, x]))
    assert np.all(np.isnan(data[:, x, y]))


def test_bad_number_param_is_400_not_500(tmp_path, stand_in_projector_config) -> None:
    """A malformed ``number`` query param must 400, not crash into a 500.

    Registers a throwaway view with a ``number`` param rather than adding a
    real view — no shipped view uses ``type="number"`` today.
    """
    params = [ParamSpec(name="scale", label="scale", type="number", default=1.0)]
    with _temporary_view("_test_number_param", ["x"], params=params):
        stand_in_projector_config(**_CAMERA_SAFE_FRAME)
        store.clear()
        client = TestClient(create_app())
        matrix_id = _load(client, tmp_path, height=6, width=8)

        response = client.get(
            f"/api/matrix/{matrix_id}/view/_test_number_param",
            params={"scale": "abc"},
        )
        assert response.status_code == 400
        assert "scale" in response.json()["detail"]
        assert "abc" in response.json()["detail"]
