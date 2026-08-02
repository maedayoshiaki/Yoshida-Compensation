"""Measured inverse LUT loading and the curve endpoint.

``tests/test_web_target.py`` の作法（クライアントの作り方、``clear()`` の
扱い、一時ファイルの置き方）に合わせる。

判別力について（このリポジトリの事情）:

- **3本のカーブは形も終端も全部違える。** 3本とも同じ形だと、チャンネルの
  入れ替え（R行とB行の取り違え）が1件も落ちない。ここでは R を直線、G を
  上に凸、B を下に凸にし、終端も 0.55 / 0.44 / 0.688 と全て違える。
- **終端は 1.0 ではない。** 実測LUTの終端は 1.0 ではなく、
  ``data/full_recalibration_20260730_iso400/inverse_lut/inverse_lut.npz`` は
  0.813 / 0.876 / 0.751 で止まっている。1.0 で終わるフィクスチャを使うと、
  縦軸を 1.0 に固定した実装と区別が付かない。
- **サンプル数は 256 ではない。** 実ファイルは 256 点だが、それをそのまま
  使うと「``t_grid`` を作り直している実装」（``np.linspace(0, 1, 256)``）と
  「保存された値を返している実装」が区別できない。ここでは 5 点にし、しかも
  ``t_grid`` を一様にしない。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp.compensation.inverse_lut import InverseLut, save_inverse_lut  # noqa: E402
from yoshidacomp.config import get_config  # noqa: E402
from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import luts  # noqa: E402

# Deliberately not 256, and deliberately not uniform: a uniform 256-point grid
# is exactly what ``np.linspace(0, 1, 256)`` produces, so an endpoint that
# rebuilt the grid instead of reading the file would agree with it everywhere.
# 一様でも 256 点でもないグリッド。``np.linspace`` で作り直す実装と区別する。
_T_GRID = [0.0, 0.2, 0.5, 0.9, 1.0]

# Three curves with three different shapes and three different endpoints.
# R is a straight line, G bulges up (concave), B bulges down (convex) and stops
# at 0.688 -- the largest of the three, and the value the plot's vertical axis
# has to come from.
# R は直線、G は上に凸、B は下に凸で 0.688 で止まる（3本の最大値）。
_LUT_ROWS = [
    [0.0, 0.11, 0.275, 0.495, 0.55],   # R: straight, ends at 0.55
    [0.0, 0.22, 0.311, 0.417, 0.44],   # G: concave,  ends at 0.44
    [0.0, 0.028, 0.172, 0.557, 0.688],  # B: convex,   ends at 0.688
]

_META = {"source": "channel_ramp", "trust_low": 5, "trust_high": 245}


def _write_lut(path, rows=None, t_grid=None, meta=None):
    """Write a real ``.npz`` through ``save_inverse_lut``, not by hand.

    The file format is the core's, so the fixture has to be produced by the
    core's own writer -- otherwise this file would pin the web layer against a
    transcription of the format rather than against the format.
    フィクスチャはコア側の書き出し関数で作る。手書きすると、検証しているのが
    「形式」ではなく「形式の写し」になる。
    """
    lut = InverseLut(
        lut=np.array(_LUT_ROWS if rows is None else rows, dtype=np.float32),
        t_grid=np.array(_T_GRID if t_grid is None else t_grid, dtype=np.float32),
        meta=_META if meta is None else meta,
    )
    save_inverse_lut(path, lut)
    return path


def _load(client, path):
    return client.post("/api/lut", json={"path": str(path)})


def test_curve_returns_the_saved_values_unchanged(tmp_path) -> None:
    """``t_grid`` と ``lut`` が保存した値のまま返ること。

    float32 で保存されるため、期待値も float32 に丸めてから比べる（丸めを
    許容誤差で隠すと、値を作り直している実装が通ってしまう）。
    """
    luts.clear()
    path = _write_lut(tmp_path / "inverse_lut.npz")

    client = TestClient(create_app())
    body = _load(client, path).json()
    curve = client.get(f"/api/lut/{body['id']}/curve")
    assert curve.status_code == 200, curve.text
    payload = curve.json()

    expected_t = np.array(_T_GRID, dtype=np.float32).tolist()
    expected_lut = np.array(_LUT_ROWS, dtype=np.float32).tolist()
    assert payload["t_grid"] == expected_t
    assert payload["lut"] == expected_lut


def test_curve_is_three_rows_in_r_g_b_order(tmp_path) -> None:
    """``lut`` は ``[R行, G行, B行]`` の入れ子リストであること。

    3本の形と終端が全て違うので、行の入れ替えはここで見える。同じ形の3本を
    置くフィクスチャでは、R と B を入れ替えても全件通る。
    """
    luts.clear()
    path = _write_lut(tmp_path / "inverse_lut.npz")

    client = TestClient(create_app())
    body = _load(client, path).json()
    payload = client.get(f"/api/lut/{body['id']}/curve").json()

    assert len(payload["lut"]) == 3
    assert all(len(row) == len(_T_GRID) for row in payload["lut"])
    assert body["meta"]["shape"] == [3, len(_T_GRID)]

    # Each row identified by a property only it has, so no two rows can be
    # swapped without one of these failing.
    # 各行を「その行にしか無い性質」で見分ける。
    r_row, g_row, b_row = payload["lut"]
    assert r_row[-1] == pytest.approx(0.55, abs=1e-6)
    assert g_row[-1] == pytest.approx(0.44, abs=1e-6)
    assert b_row[-1] == pytest.approx(0.688, abs=1e-6)

    # ...and by shape, not only by endpoint: R is straight, G is above the
    # straight line joining its own ends, B is below its own.
    # 終端だけでなく形でも見分ける。
    def _midpoint_offset(row: list[float]) -> float:
        """How far the middle sample sits above the chord of its own ends."""
        t_mid = (_T_GRID[2] - _T_GRID[0]) / (_T_GRID[-1] - _T_GRID[0])
        chord = row[0] + t_mid * (row[-1] - row[0])
        return row[2] - chord

    assert abs(_midpoint_offset(r_row)) < 1e-6, "R must be straight"
    assert _midpoint_offset(g_row) > 0.05, "G must bulge up"
    assert _midpoint_offset(b_row) < -0.05, "B must bulge down"


def test_the_terminal_values_are_not_1_and_differ_per_channel(tmp_path) -> None:
    """このフィクスチャ自身の判別力を固定する。

    実測LUTの終端は 1.0 ではない（ISO400 の実ファイルは 0.813 / 0.876 /
    0.751）。3本とも 1.0 で終わるフィクスチャにすると、縦軸を 1.0 に固定した
    プロットと、データ由来の上限を持つプロットが区別できなくなる。
    ``tests/js/frontend_harness.mjs`` の ``lut`` シナリオが同じ理由で同じ形の
    データを使っており、ここはその前提を Python 側にも書き留めておくもの。
    """
    ends = [row[-1] for row in _LUT_ROWS]
    assert all(end < 1.0 for end in ends), ends
    assert len(set(ends)) == 3, ends
    assert max(ends) == pytest.approx(0.688)


def test_meta_is_parsed_into_a_dict(tmp_path) -> None:
    """``.npz`` の ``meta`` は JSON 文字列で保存される。辞書として返すこと。"""
    luts.clear()
    path = _write_lut(tmp_path / "inverse_lut.npz")

    client = TestClient(create_app())
    body = _load(client, path).json()
    payload = client.get(f"/api/lut/{body['id']}/curve").json()

    assert payload["meta"] == _META
    assert body["meta"]["meta"] == _META
    # The file really does hold it as a string, so "returns a dict" is a
    # transformation this endpoint performs rather than a property of the file.
    # ファイル側は本当に文字列で持っている（＝辞書化はこの層の仕事である）。
    with np.load(path, allow_pickle=False) as data:
        assert isinstance(str(data["meta"]), str)
        assert json.loads(str(data["meta"])) == _META


def test_the_curve_carries_the_configured_projector_gamma(tmp_path) -> None:
    """解析ガンマを重ねるための指数は**プロジェクタ設定**の値であること。

    フロントには他に知る手段が無く、違う指数を重ねれば「定量的に見えるが
    そうではない比較」になる。

    **判別力について。** 実 ``config.toml`` の ``gamma`` は 2.2 であり、それは
    ちょうど「よくある指数」でもある。設定を読まず ``2.2`` を直書きした実装で
    も、実configのまま比べる限り区別が付かない（実測: ``"gamma":
    float(get_config().projector.gamma)`` を ``2.2`` に置き換えても
    このファイルの19件すべてが通った）。2.2 でない値を設定してから確認する。
    """
    from yoshidacomp.config import reload_config

    luts.clear()
    path = _write_lut(tmp_path / "inverse_lut.npz")

    # Deliberately neither 2.2 nor 1.0: the first is the default a hard-coded
    # implementation would use, the second is the value an implementation that
    # forgot the exponent entirely would produce.
    # 2.2 でも 1.0 でもない値にする（前者は直書き、後者は指数を忘れた実装）。
    unusual_gamma = 1.7
    config_path = tmp_path / "unusual_gamma.toml"
    config_path.write_text(
        f"[projector]\nwidth = 64\nheight = 48\ngamma = {unusual_gamma}\n",
        encoding="utf-8",
    )
    # The autouse ``_restore_real_config`` fixture puts the real config back
    # after this test, on both sides -- see tests/conftest.py.
    reload_config(config_path)
    assert float(get_config().projector.gamma) == pytest.approx(unusual_gamma)

    client = TestClient(create_app())
    body = _load(client, path).json()
    payload = client.get(f"/api/lut/{body['id']}/curve").json()

    assert payload["gamma"] == pytest.approx(unusual_gamma)
    assert payload["gamma"] != pytest.approx(2.2)


def test_post_reports_the_path_and_shape(tmp_path) -> None:
    luts.clear()
    path = _write_lut(tmp_path / "inverse_lut.npz")

    client = TestClient(create_app())
    response = _load(client, path)
    assert response.status_code == 200, response.text
    body = response.json()
    # Not a literal "lut1": the counter is a process-wide singleton shared
    # with every other test in this file and never rewound by ``clear()``.
    # ``tests/test_web_store.py`` pins the numbering rule on its own instance.
    assert body["id"].startswith("lut")
    assert body["id"] == body["meta"]["id"]
    assert body["meta"]["path"] == str(path)
    assert body["meta"]["shape"] == [3, len(_T_GRID)]
    # This one pins the field's *presence*, not its value, and is kept knowing
    # that: ``save_inverse_lut`` writes float32 and ``load_inverse_lut`` reads
    # float32, so no fixture this loader accepts can make it say anything else
    # (task-4 final review, Minor 13). What it still catches is the key going
    # missing or being renamed -- the meta is what the sidebar's info line and
    # any future caller read.
    assert body["meta"]["dtype"] == "float32"


def test_missing_path_is_404(tmp_path) -> None:
    luts.clear()
    missing = tmp_path / "nope.npz"
    client = TestClient(create_app())
    response = _load(client, missing)
    assert response.status_code == 404
    assert str(missing) in response.json()["detail"]


def test_empty_path_is_400() -> None:
    """``Path("")`` is ``Path(".")``; say "empty" rather than "'.' is not a file"."""
    luts.clear()
    client = TestClient(create_app())
    for value in ("", "   "):
        response = client.post("/api/lut", json={"path": value})
        assert response.status_code == 400, value
        assert "empty" in response.json()["detail"]


def test_unknown_id_is_404() -> None:
    luts.clear()
    client = TestClient(create_app())
    response = client.get("/api/lut/nosuch/curve")
    assert response.status_code == 404
    # Not just "some 404": ``StaticFiles(html=True)`` mounted at "/" would also
    # 404 an unmatched path if this route were never registered at all. The
    # message is what proves *this* route answered.
    assert "lut not found" in response.json()["detail"]
    assert "nosuch" in response.json()["detail"]


def test_a_directory_is_400(tmp_path) -> None:
    luts.clear()
    client = TestClient(create_app())
    response = _load(client, tmp_path)
    assert response.status_code == 400
    assert str(tmp_path) in response.json()["detail"]


def test_a_non_npz_extension_is_400_with_a_readable_message(tmp_path) -> None:
    """``.npy`` を渡すと ``np.load`` は素の ndarray を返し、``with`` が
    ``TypeError`` になる。それは「利用者の誤り」を説明していないので、
    拡張子の段階で断る。
    """
    luts.clear()
    path = tmp_path / "matrices.npy"
    np.save(path, np.zeros((2, 2), dtype=np.float32))

    client = TestClient(create_app())
    response = _load(client, path)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert str(path) in detail
    assert ".npz" in detail
    # This is the extension branch, before any attempt to read: the message
    # must not be the read-failure one.
    assert "could not read" not in detail.lower()


def test_a_corrupt_npz_is_400_with_a_readable_message(tmp_path) -> None:
    """拡張子は正しいが中身が壊れている場合。上のテストとは別の枝であり、
    片方の文言がもう片方に出ないことまで見る（``test_web_target.py`` の
    拡張子/デコード失敗の2件と同じ作法）。
    """
    luts.clear()
    path = tmp_path / "broken.npz"
    path.write_bytes(b"this is not actually a numpy archive")

    client = TestClient(create_app())
    response = _load(client, path)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert str(path) in detail
    assert "could not read" in detail.lower()
    assert "expected .npz" not in detail.lower()  # not the extension message


def test_an_npz_without_a_lut_entry_is_400(tmp_path) -> None:
    """``lut`` の無い ``.npz`` は ``KeyError`` になる。500 にしない。"""
    luts.clear()
    path = tmp_path / "other.npz"
    np.savez(path, something_else=np.zeros(4, dtype=np.float32))

    client = TestClient(create_app())
    response = _load(client, path)
    assert response.status_code == 400
    assert str(path) in response.json()["detail"]


@pytest.mark.parametrize(
    "rows,t_grid,expected_substring",
    [
        # Two channels instead of three: ``apply_inverse_lut`` indexes
        # ``lut.lut[2]`` and would raise ``IndexError`` *inside the view*,
        # i.e. a 500, at render time rather than at load time.
        ([[0.0, 1.0], [0.0, 1.0]], [0.0, 1.0], "one row per channel"),
        # Mismatched lengths: ``np.interp`` raises there, again at render time.
        ([[0.0, 0.5, 1.0]] * 3, [0.0, 1.0], "same length"),
        # A single sample: ``np.interp`` accepts it and returns a constant, so
        # this one is silent rather than loud -- every pixel linearizes to the
        # same code value and the picture is a flat colour that looks plausible.
        ([[0.5]] * 3, [0.0], "at least 2 samples"),
    ],
    ids=["two_channels", "length_mismatch", "single_sample"],
)
def test_a_malformed_lut_is_refused_at_load_time_not_at_render_time(
    tmp_path, rows, t_grid, expected_substring
) -> None:
    """形の違う ``.npz`` は読み込み時に 400 で断ること。

    ``load_inverse_lut`` は形を見ないので、ここで断らないと失敗するのは
    ``apply_inverse_lut`` -- つまり ``views/compensation.py`` の中であり、
    ``IndexError``/``ValueError`` はビューの 500 になる。「読み込んだLUTの形が
    違った」を「サーバが壊れた」として返すことになる。
    """
    luts.clear()
    path = _write_lut(tmp_path / "malformed.npz", rows=rows, t_grid=t_grid)

    client = TestClient(create_app())
    response = _load(client, path)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert str(path) in detail
    assert expected_substring in detail


def test_a_lut_holding_nan_is_refused(tmp_path) -> None:
    """NaN を含む LUT は 400。

    ``np.interp`` は NaN を伝播するだけなので、通してしまうとフレーム全体が
    NaN になり、有効領域外と同じマスク色で描かれる——「全部マスク」から
    「LUTにNaNが入っていた」へ辿る道が無い。
    """
    luts.clear()
    rows = [list(row) for row in _LUT_ROWS]
    rows[1][2] = float("nan")
    path = _write_lut(tmp_path / "nan.npz", rows=rows)

    client = TestClient(create_app())
    response = _load(client, path)
    assert response.status_code == 400
    assert "non-finite" in response.json()["detail"]


def test_lut_ids_do_not_cross_resolve_with_matrix_or_target_ids(tmp_path) -> None:
    """``LutStore`` は ``_IdStore`` を共有する3つ目の店子である。

    ``_items``/``_lock``/``_counter`` がクラス変数へ格上げされると3つの
    singleton が1つのid空間を共有し、``GET /api/lut/{ある行列のid}/curve`` が
    ``MatrixContext`` を解決して ``.lut`` 読み取りで素の500になる
    （``tests/test_web_target.py`` の同名テストと同じ論理）。
    """
    import cv2

    from yoshidacomp_web.store import store, targets

    store.clear()
    targets.clear()
    luts.clear()

    matrix_path = tmp_path / "m.npy"
    matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for channel in range(3):
        matrices[:, :, channel, channel] = 1.0
    np.save(matrix_path, matrices)

    target_path = tmp_path / "t.png"
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[..., 0] = 200
    assert cv2.imwrite(str(target_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    lut_path = _write_lut(tmp_path / "inverse_lut.npz")

    client = TestClient(create_app())
    matrix_id = client.post(
        "/api/session/matrix", json={"path": str(matrix_path)}
    ).json()["id"]
    target_id = client.post(
        "/api/target", json={"path": str(target_path)}
    ).json()["id"]
    lut_id = _load(client, lut_path).json()["id"]

    assert matrix_id.startswith("matrix")
    assert target_id.startswith("target")
    assert lut_id.startswith("lut")

    assert client.get(f"/api/lut/{matrix_id}/curve").status_code == 404
    assert client.get(f"/api/lut/{target_id}/curve").status_code == 404
    assert client.get(f"/api/matrix/{lut_id}/meta").status_code == 404
    assert client.get(f"/api/target/{lut_id}/meta").status_code == 404


def test_the_real_measured_luts_load_and_keep_their_sub_unit_endpoints() -> None:
    """出荷物ではなく**実測ファイル**で読み込み経路を通す。

    ``data/full_recalibration_20260730_iso400/inverse_lut/inverse_lut.npz`` の
    終端は 0.813 / 0.876 / 0.751 であり 1.0 ではない。この事実がこのタスクの
    プロットの縦軸を決めているので、ファイルが在る環境ではそれを確認する
    （このリポジトリの ``data/`` は git 管理外なので、無ければ skip）。

    その3つの値を名指しで固定する。当初は ``0 < end < 1`` しか見ておらず、
    docstring が挙げる数値のどれも固定していなかった。``data/`` 配下の実測LUT
    15本のうち **9本は終端がちょうど 1.0** であり（レビューが全走査して確認）、
    既定パスがそちら側のファイルを指すよう変わればこのテストは skip も fail も
    せずに静かに意味を失う——`plot.js` の縦軸の根拠として引用しているのは
    「1.0 未満で、しかも3チャンネルとも違う」という**この**ファイルの性質で
    あって、「1.0 未満のLUTが世の中にある」ではない。
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    measured = (
        repo_root
        / "data"
        / "full_recalibration_20260730_iso400"
        / "inverse_lut"
        / "inverse_lut.npz"
    )
    if not measured.is_file():
        pytest.skip(f"measured LUT not present in this checkout: {measured}")

    luts.clear()
    client = TestClient(create_app())
    response = _load(client, measured)
    assert response.status_code == 200, response.text
    payload = client.get(f"/api/lut/{response.json()['id']}/curve").json()

    assert len(payload["lut"]) == 3
    assert len(payload["t_grid"]) == len(payload["lut"][0])
    ends = [row[-1] for row in payload["lut"]]
    # The numbers this file is cited for, to 3 decimals -- the same ones
    # ``static/js/plot.js`` and this module's docstring quote. float32 samples
    # arriving through JSON, so an exact comparison would be about the codec
    # rather than about the file.
    assert [round(end, 3) for end in ends] == [0.813, 0.876, 0.751], ends
    # And the property those three have that the plot depends on, stated as a
    # property: below 1.0, and different from one another.
    assert all(0.0 < end < 1.0 for end in ends), ends
    assert len(set(round(end, 3) for end in ends)) == 3, ends
    assert payload["meta"]["source"] == "channel_ramp"
