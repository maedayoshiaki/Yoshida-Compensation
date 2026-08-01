"""GET /api/matrix/{id}/probe: point query for one pixel's 4x3 coefficients.

行列全体(H, W, 4, 3)を送らず、1画素分の12個のfloatだけを返す点クエリの
テスト。``coef_tiles`` のような係数ビューが無いビュー(``surface``)でも、
クライアントは行列そのものを1画素分だけ読める必要がある。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.routers.session import _finite_or_none  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402


def _load(client: TestClient, tmp_path, array: np.ndarray) -> str:
    """Save *array* as a ``.npy`` and register it; return the new matrix id."""
    path = tmp_path / "m.npy"
    np.save(path, array.astype(np.float32))
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_probe_returns_the_four_by_three_at_that_pixel(tmp_path) -> None:
    # 全画素で違う値にする。定数行列だと row/col の取り違えが見えない。
    store.clear()
    array = np.arange(4 * 5 * 4 * 3, dtype=np.float32).reshape(4, 5, 4, 3)
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, array)

    response = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 2, "col": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["row"] == 2
    assert body["col"] == 3
    assert body["values"] == array[2, 3].tolist()


def test_probe_does_not_transpose_row_and_col(tmp_path) -> None:
    # 正方形でない形にする。row/col を取り違えれば別画素の値になる。
    store.clear()
    array = np.arange(3 * 7 * 4 * 3, dtype=np.float32).reshape(3, 7, 4, 3)
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, array)

    response = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 1, "col": 6})
    assert response.status_code == 200, response.text
    assert response.json()["values"] == array[1, 6].tolist()


def test_probe_turns_non_finite_into_null(tmp_path) -> None:
    # NaN/Inf 係数は null としてこのエンドポイントの契約になる（理由は
    # session.py::_finite_or_none のdocstringを参照。Starlette の
    # allow_nan=False と Pydantic の既定変換のどちらも「JSON.parse が壊れた
    # 本文を受け取る」事態を防いではいるが、それはこのエンドポイント自身の
    # 契約ではなく、変換を怠れば有効領域外への素の500に変わる）。
    store.clear()
    array = np.zeros((2, 2, 4, 3), dtype=np.float32)
    array[0, 0, 1, 2] = np.nan
    array[0, 0, 2, 0] = np.inf
    array[0, 0, 2, 1] = -np.inf
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, array)

    response = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 0, "col": 0})
    assert response.status_code == 200, response.text
    # These two cannot fail against this stack today -- Starlette's
    # ``allow_nan=False`` guarantees a stray non-finite value 500s rather than
    # reaching the wire as "NaN"/"Infinity". Kept as regression guards against
    # a future hand-rolled ``json.dumps(allow_nan=True)`` or similar, not as
    # evidence that the conversion works; that is
    # ``test_finite_or_none_maps_nan_and_infinities_to_none_directly`` below.
    assert "NaN" not in response.text
    assert "Infinity" not in response.text
    body = json.loads(response.text)
    assert body["values"][1][2] is None
    assert body["values"][2][0] is None
    assert body["values"][2][1] is None
    assert body["values"][0][0] == 0.0


def test_finite_or_none_maps_nan_and_infinities_to_none_directly() -> None:
    """Unit-check ``_finite_or_none`` itself, not only through the HTTP round trip.

    FastAPI (with a return-type-annotated route, as ``probe_matrix`` is) here
    serialises the body through a Pydantic ``TypeAdapter``, whose JSON encoder
    already maps NaN/Infinity to ``null`` by default (Pydantic 2's
    ``ser_json_inf_nan``) -- independent of anything ``_finite_or_none`` does.
    That means ``test_probe_turns_non_finite_into_null`` above would see the
    same ``null`` even if ``_finite_or_none`` were replaced with a bare
    ``float(value)``: the framework launders the exact defect that function
    exists to prevent. Calling it directly is the only way this repository's
    test suite holds it accountable.

    ``probe_matrix`` はレスポンス型注釈つきのルートであり、本文は Pydantic の
    ``TypeAdapter`` を通って直列化される。そのJSONエンコーダは既定で
    NaN/Infinity を ``null`` に変換する（Pydantic 2 の ``ser_json_inf_nan``）
    ため、``_finite_or_none`` を素の ``float(value)`` に差し替えても HTTP
    経由では同じ ``null`` が返り、上のテストは検出できない。直接呼び出すこと
    でのみ、この関数自身の契約を検証できる。
    """
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(float("-inf")) is None
    assert _finite_or_none(0.0) == 0.0
    assert _finite_or_none(1.5) == 1.5


@pytest.mark.parametrize(
    ("row", "col"),
    [(-1, 0), (0, -1), (2, 0), (0, 2)],
)
def test_probe_refuses_out_of_bounds(tmp_path, row, col) -> None:
    store.clear()
    array = np.zeros((2, 2, 4, 3), dtype=np.float32)
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, array)

    response = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": row, "col": col})
    assert response.status_code == 400, response.text


def test_probe_on_unknown_matrix_is_404() -> None:
    store.clear()
    client = TestClient(create_app())
    response = client.get("/api/matrix/nosuch/probe", params={"row": 0, "col": 0})
    assert response.status_code == 404


def test_probe_reports_whether_a_pixel_is_inside_the_active_region(
    tmp_path, stand_in_projector_config
) -> None:
    """``inside_active_region`` must honour ``(x, y, w, h)`` as given.

    The real ``config.toml`` is an experiment log edited constantly (per
    ``tests/conftest.py``'s own docstring), and its active region may or may
    not overlap whatever test matrix is loaded, which would make this
    assertion depend on which experiment happens to be checked in. Using
    ``stand_in_projector_config`` (the same fixture
    ``test_web_views.py::test_coef_tiles_masks_outside_active_region`` uses
    for exactly this reason) fixes a non-square frame and an off-centre,
    non-square (``w != h``) active rectangle, so both the True and False
    branches are exercised deterministically.

    A non-square rectangle only makes a defect detectable if some assertion
    actually depends on ``w`` and ``h`` separately, or on the boundary's
    exclusive edge -- neither followed from non-squareness alone, so both are
    probed explicitly below: a transposed row/col <-> x/y read lands on a
    provably different pixel (the "trap" point), a transposed w/h read is
    separated by probing the rectangle's own last column against one past it
    (``last_col`` / ``past_col``), the far edge's exclusivity is checked
    directly (``past_col``, ``past_row``), and the ``and``/``or`` between the
    row and column conditions is separated by two partial-overlap points
    (``row_only``, ``col_only``).

    実 ``config.toml`` は実験のたびに編集されるログであり、その有効領域が
    テスト行列と重なるかどうかは「たまたま何がチェックインされているか」に
    依存してしまう。``stand_in_projector_config``（``test_web_views.py`` の
    同種テストと同じ理由で使う代用config）で非正方形の枠と非対称・非正方形
    （``w != h``）の矩形を固定し、True/False 両方の分岐を決定的に検証する。

    非正方形の矩形だけでは、``w`` と ``h`` を別々に、あるいは境界の排他的な
    端を実際に使うアサーションが無ければ欠陥は検出できない。そのため以下を
    個別に確認する: 行/列と x/y の取り違え（trap点）、w/h の取り違え
    （矩形自身の最終列とその1つ先を比較する ``last_col``/``past_col``）、
    境界の排他性（``past_col``、``past_row``）、行/列条件間の ``and``/``or``
    の取り違え（``row_only``、``col_only`` の部分重なり2点）。
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
    array = np.zeros((height, width, 4, 3), dtype=np.float32)
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, array)

    inside = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": y, "col": x})
    assert inside.status_code == 200, inside.text
    assert inside.json()["inside_active_region"] is True

    outside = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 0, "col": 0})
    assert outside.status_code == 200, outside.text
    assert outside.json()["inside_active_region"] is False

    # The point a transposed row/col <-> x/y reading would have kept "inside"
    # instead (x != y and w != h here, so this is a provably different pixel
    # from the true rectangle above).
    trap = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": x, "col": y})
    assert trap.status_code == 200, trap.text
    assert trap.json()["inside_active_region"] is False

    # Partial overlaps: row inside the rect's row range but col outside it,
    # and vice versa. Both (0, 0) and the trap point above satisfy neither
    # the row nor the column condition, so an "and" mistakenly written as
    # "or" in ``_inside_active_region`` would still pass every assertion so
    # far -- these two points each satisfy exactly one of the two conditions
    # and must still be False.
    row_only = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": y, "col": 0})
    assert row_only.status_code == 200, row_only.text
    assert row_only.json()["inside_active_region"] is False

    col_only = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 0, "col": x})
    assert col_only.status_code == 200, col_only.text
    assert col_only.json()["inside_active_region"] is False

    # None of the points above separate the rectangle's own width from its
    # own height, or touch its boundary -- a w/h transposition in
    # ``_inside_active_region`` (``col < x + h``, ``row < y + w``) and a
    # far-edge off-by-one (using ``<=`` instead of ``<``) both pass every
    # assertion above unnoticed. w=4 and h=2 here are chosen unequal
    # specifically so a swap lands on a different pixel.
    #
    # Far edge is exclusive, and w is the column extent (not h).
    last_col = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": y, "col": x + w - 1})
    assert last_col.status_code == 200, last_col.text
    assert last_col.json()["inside_active_region"] is True

    past_col = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": y, "col": x + w})
    assert past_col.status_code == 200, past_col.text
    assert past_col.json()["inside_active_region"] is False

    past_row = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": y + h, "col": x})
    assert past_row.status_code == 200, past_row.text
    assert past_row.json()["inside_active_region"] is False


def test_probe_is_true_everywhere_when_the_matrix_has_no_active_region(
    tmp_path, stand_in_projector_config
) -> None:
    """A camera-space matrix has no active region, so every pixel is "inside".

    Fixes a stand-in projector frame (rather than trusting that the live
    ``config.toml``'s 1920x1080 happens not to match the 9x11 test matrix)
    so this stays unconditional regardless of what the live config says.
    """
    stand_in_projector_config(width=64, height=48)
    store.clear()
    # 9x11 matches neither the stand-in frame nor its transpose, so this
    # always classifies as camera space.
    array = np.zeros((9, 11, 4, 3), dtype=np.float32)
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, array)

    response = client.get(f"/api/matrix/{matrix_id}/probe", params={"row": 0, "col": 0})
    assert response.status_code == 200, response.text
    assert response.json()["inside_active_region"] is True
