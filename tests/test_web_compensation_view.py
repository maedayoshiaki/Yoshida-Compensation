"""The ``compensation`` view: a target image with the matrix applied to it.

このファイルの中心は
``test_view_matches_generate_projection_image_exactly`` である。ビューが
補償の数式を自前で持たず ``pipeline.generate_projection_image()`` を
そのまま通していることを、この1件（linearizer × output の4通り）が担保する。
期待値は手計算の定数ではなく、テスト内で実関数を呼んで作る。

判別力について（このリポジトリの事情）:

- 行列は**非対称**。計画B2では ``A`` が全て ``2·I`` だったため逆行列の転置が
  14テスト全部を通った。ここでは ``max|offdiag| / max|diag|`` を実測して
  アサートする。
- 行列は**画素ごとに変化**させる。全画素同一だと行列の並べ替えが見えない。
- ターゲット画像は**多色**かつ**縦横が違う**。単色だと画素の入れ替えが、
  正方形だと転置が見えない。
- ターゲットの縦横は有効領域の縦横とも違える。同じにすると
  ``embed_in_active_region`` のリサイズが恒等写像になり、その経路が
  検証されない。
"""

from __future__ import annotations

from typing import get_args

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp.compensation.pipeline import (  # noqa: E402
    LINEARIZERS,
    LinearizerName,
    TargetDecodeMode,
    generate_projection_image,
)
from yoshidacomp.config import get_config  # noqa: E402
from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store, targets  # noqa: E402

# The projector frame. Non-square, and neither side equals the other, so a
# transposed (H, W) anywhere lands on a different shape.
_FRAME_WIDTH, _FRAME_HEIGHT = 8, 6

# ``(x, y, w, h)``, deliberately off-centre and asymmetric: ``x != y`` and
# ``w != h``, so a transposed slice in the NaN mask hits a disjoint set of
# pixels. Rows 2-4, columns 1-5.
_REGION = (1, 2, 5, 3)

# The target image's own size: different from the frame *and* from the active
# region, in both dimensions, so ``embed_in_active_region``'s resize genuinely
# runs (7 -> 5 wide, 4 -> 3 tall, i.e. the INTER_AREA branch).
_TARGET_HEIGHT, _TARGET_WIDTH = 4, 7

# A deliberately non-symmetric affine. Scaled so that ``[t, 1] @ M`` lands
# inside [0, 1] for most of the target's range -- a fixture that clips
# everywhere would hide every difference this file is trying to see.
_A = np.array(
    [
        [0.60, 0.18, -0.12],
        [0.09, 0.54, 0.15],
        [-0.06, 0.12, 0.66],
    ],
    dtype=np.float32,
)
_B = np.array([0.08, 0.05, 0.10], dtype=np.float32)


def _matrices(height: int, width: int) -> np.ndarray:
    """A per-pixel-varying, non-symmetric ``(H, W, 4, 3)`` matrix.

    行列を画素ごとに変える（±約8%）。全画素同一だと、ビューが行列を
    上下反転・転置して渡しても同じ答えになってしまう。
    """
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    modulation = (1.0 + 0.004 * (rows + 2.0 * cols)).astype(np.float32)

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, :3, :] = _A * modulation[:, :, None, None]
    matrices[:, :, 3, :] = _B
    return matrices


def _target_image(height: int, width: int) -> np.ndarray:
    """A multi-coloured ``(H, W, 3)`` uint8 RGB image, deterministic.

    3チャンネルとも別の式にする。単色や灰色だと、チャンネルの入れ替えも
    画素の入れ替えも見えない。
    """
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (37 * cols + 11 * rows) % 256
    image[:, :, 1] = (17 * cols + 53 * rows + 90) % 256
    image[:, :, 2] = 255 - (29 * cols + 7 * rows) % 256
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


def _reference(
    target: np.ndarray,
    matrices: np.ndarray,
    *,
    linearizer: str = "none",
    output: str = "linear",
    target_decode: str = "srgb",
    target_gamma: float = 2.2,
    lut=None,
) -> np.ndarray:
    """What the CLI would compute, via the real pipeline function.

    期待値はここで作る。ビューと同じ引数を同じ関数へ渡し、返ってきた2つの
    配列のうち *output* が指すほうを返す。定数を書き写す余地を残さない。
    """
    linear, code = generate_projection_image(
        target,
        matrices,
        projector=get_config().projector,
        active_region=_REGION,
        linearizer=linearizer,  # type: ignore[arg-type]
        lut=lut,
        target_decode=target_decode,  # type: ignore[arg-type]
        target_gamma=target_gamma,
        dtype=np.float32,
    )
    return linear if output == "linear" else code


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
    return f"/api/matrix/{matrix_id}/view/compensation?{query}"


# --------------------------------------------------------------------------
# 1. The central claim: the view is the pipeline, not a second implementation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("linearizer", ["none", "gamma"])
@pytest.mark.parametrize("output", ["linear", "code"])
def test_view_matches_generate_projection_image_exactly(
    tmp_path, stand_in_projector_config, linearizer: str, output: str
) -> None:
    """The view's pixels must equal ``generate_projection_image()``'s, exactly.

    Both sides run the same float32 code over the same inputs, so this is an
    equality, not an ``allclose``: any rewrite of the compensation arithmetic
    inside the view -- even one that agrees to five decimals -- fails here.

    ビューの出力が ``generate_projection_image()`` の戻り値と**完全一致**する
    こと。両者は同じ float32 経路を通るので許容誤差を置かない。ビュー側に
    数式を書き直せば、小数5桁まで合っていてもここで落ちる。

    ``output`` を跨いで比較しているのがポイントで、戻り値2つの取り違え
    （linear と code の入れ替え）はこの比較でしか捕まらない。「linear と
    code が違う値になる」だけを見るテストは、入れ替えても両方違うままなので
    通ってしまう。
    """
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )

    response = client.get(
        _url(
            matrix_id,
            target=target_id,
            linearizer=linearizer,
            output=output,
        )
    )
    assert response.status_code == 200, response.text
    data = _hwc(response)

    expected = _reference(image, matrices, linearizer=linearizer, output=output)
    inside = _inside_mask()

    np.testing.assert_array_equal(data[inside], expected[inside])
    assert np.isnan(data[~inside]).all(), (
        "outside the active region the view must be NaN, not the pipeline's own "
        "zeros -- see test_outside_the_active_region_is_nan_not_black"
    )
    # The pipeline's own convention outside the region is 0.0 (mask_to_active_
    # region). Stating it here is what makes the NaN assertion above a real
    # difference rather than a restatement.
    assert (expected[~inside] == 0.0).all()

    # --- the fixture's own discriminating power, asserted rather than assumed.
    a_mat = matrices[0, 0, :3, :]
    asymmetry = float(
        np.max(np.abs(a_mat - np.diag(np.diag(a_mat))))
        / np.max(np.abs(np.diag(a_mat)))
    )
    assert asymmetry > 0.2, (
        f"A is too close to diagonal ({asymmetry:.3f}) for this file to tell a "
        "transposed matrix from the real one; real matrices sit near 0.25"
    )
    assert _TARGET_HEIGHT != _TARGET_WIDTH, "a square target hides a transpose"
    assert not np.array_equal(image[:, :, 0], image[:, :, 1]), (
        "the target's channels must differ or a channel swap is invisible"
    )
    values = expected[inside]
    assert float(values.std()) > 0.01, (
        "the expected result is nearly constant -- the fixture clips or "
        "saturates and would hide almost any error"
    )
    assert float(np.mean((values <= 0.0) | (values >= 1.0))) < 0.5, (
        "more than half the expected result is clipped; pick a smaller A"
    )


def test_a_transposed_target_would_produce_a_different_answer(
    tmp_path, stand_in_projector_config
) -> None:
    """The non-square target must actually matter.

    ``embed_in_active_region`` resizes whatever it is given into the active
    rectangle, so a view that handed the pipeline ``image.transpose(1, 0, 2)``
    would still return a full, plausible frame. This pins that such a frame is
    *different*, which is what makes
    ``test_view_matches_generate_projection_image_exactly`` able to catch it.

    ターゲットはリサイズされて有効領域に収まるため、転置して渡しても
    見た目には成立した結果が返る。転置版が別の値になることをここで固定する。
    これが上のテストに転置を検出する力を与えている。
    """
    _, _, _, matrices, image = _prepared(tmp_path, stand_in_projector_config)
    straight = _reference(image, matrices)
    transposed = _reference(np.ascontiguousarray(image.transpose(1, 0, 2)), matrices)
    inside = _inside_mask()
    assert not np.allclose(straight[inside], transposed[inside], atol=1e-4)


# --------------------------------------------------------------------------
# 2-3. The parameters have to do something.
# --------------------------------------------------------------------------


def test_linearizer_none_and_gamma_produce_different_pixels(
    tmp_path, stand_in_projector_config
) -> None:
    """``linearizer`` が効いていること。同じ値になるなら通っていない。"""
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)

    def _fetch(linearizer: str) -> np.ndarray:
        response = client.get(
            _url(matrix_id, target=target_id, linearizer=linearizer, output="code")
        )
        assert response.status_code == 200, response.text
        return _hwc(response)

    plain = _fetch("none")
    gamma = _fetch("gamma")
    inside = _inside_mask()
    assert np.isfinite(plain[inside]).all()
    assert np.isfinite(gamma[inside]).all()
    # Not "differ somewhere": the inverse gamma is a strictly increasing map
    # with a single fixed point at each end, so it must brighten essentially
    # every interior pixel. A view that ignored the parameter would be equal
    # everywhere.
    # 「どこかが違う」ではなく「内部の画素がほぼ全て明るくなる」ことを見る。
    differing = float(np.mean(gamma[inside] != plain[inside]))
    assert differing > 0.9, f"only {differing:.1%} of pixels changed with gamma"
    assert (gamma[inside] >= plain[inside]).all(), (
        "1/gamma brightens; a darker result means the exponent was applied "
        "the wrong way round"
    )


def test_output_selects_between_the_two_returned_images(
    tmp_path, stand_in_projector_config
) -> None:
    """``output`` が2つの戻り値を選び分けること。

    ``linearizer="gamma"`` で行う。``"none"`` では ``_linearize_none`` が
    受け取った配列をそのまま返すため linear と code は**同じ配列**であり、
    「違う値になること」を要求すると必ず落ちる。その同一性自体もここで
    固定しておく（後段の linearizer が入っても pass-through が保たれること）。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)

    def _fetch(linearizer: str, output: str) -> np.ndarray:
        response = client.get(
            _url(matrix_id, target=target_id, linearizer=linearizer, output=output)
        )
        assert response.status_code == 200, response.text
        return _hwc(response)

    inside = _inside_mask()
    linear = _fetch("gamma", "linear")
    code = _fetch("gamma", "code")
    assert not np.allclose(linear[inside], code[inside], atol=1e-6)

    # With no linearizer the two outputs are the same image by definition.
    np.testing.assert_array_equal(
        _fetch("none", "linear")[inside], _fetch("none", "code")[inside]
    )


def test_target_decode_changes_the_result_and_follows_the_pipeline(
    tmp_path, stand_in_projector_config
) -> None:
    """``target_decode`` はこのビューが持つ（Task 2 は生の uint8 を保持する）。

    ``srgb``（既定）と ``none`` で結果が変わり、どちらも実関数の答えと
    一致すること。既定が恒等でないことを示すのがこのテストの要点で、
    「復号方式を選べる」と称して常に同じ経路を通る実装をここで落とす。
    """
    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )
    inside = _inside_mask()

    results = {}
    for mode in ("srgb", "none"):
        response = client.get(
            _url(matrix_id, target=target_id, target_decode=mode)
        )
        assert response.status_code == 200, response.text
        results[mode] = _hwc(response)
        np.testing.assert_array_equal(
            results[mode][inside],
            _reference(image, matrices, target_decode=mode)[inside],
        )
    assert not np.allclose(results["srgb"][inside], results["none"][inside], atol=1e-6)

    # The default is srgb, not none: a PNG target is sRGB-encoded and the
    # compensation runs on linear light.
    default = _hwc(client.get(_url(matrix_id, target=target_id)))
    np.testing.assert_array_equal(default[inside], results["srgb"][inside])


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
        return _hwc(response)

    default_gamma = _fetch(2.2)
    other_gamma = _fetch(1.4)
    np.testing.assert_array_equal(
        default_gamma[inside],
        _reference(image, matrices, target_decode="gamma", target_gamma=2.2)[inside],
    )
    assert not np.allclose(default_gamma[inside], other_gamma[inside], atol=1e-6)


# --------------------------------------------------------------------------
# The mask. Outside the active region is NaN, not black.
# --------------------------------------------------------------------------


def test_outside_the_active_region_is_nan_not_black(
    tmp_path, stand_in_projector_config
) -> None:
    """有効領域の外は NaN。0 で埋めると「黒く投影した」ように描かれる。

    ``coef_tiles`` と ``surface`` が同じ規約でマスクしており、読み取り欄の
    マーカーもこれに合わせてある。パイプライン自身は
    ``mask_to_active_region`` で 0 を書くので、ビューが上書きしなければ
    ここは有限値のままになる -- それがこのテストが捕らえる欠陥である。

    ``(x, y, w, h)`` は非対称（``x != y``、``w != h``）なので、スライスの
    x/y や w/h を取り違えると別の画素集合に着地し、下の集合一致で落ちる。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    data = _hwc(response)

    finite = np.all(np.isfinite(data), axis=2)
    assert np.array_equal(finite, _inside_mask()), (
        "the NaN mask does not match (x, y, w, h) as configured -- check for a "
        "transposed x/y or w/h in the mask slice"
    )

    x, y, w, h = _REGION
    assert np.isfinite(data[y, x]).all()          # inside: the region's corner
    assert np.isnan(data[0, 0]).all()             # outside: the frame's corner
    # The point a transposed (x, y) reading would have kept instead.
    assert np.isnan(data[x, y]).all()


# --------------------------------------------------------------------------
# Refusals. Every one of these is a 400 with a readable reason, not a 500.
# --------------------------------------------------------------------------


def test_a_camera_space_matrix_is_refused(tmp_path, stand_in_projector_config) -> None:
    """カメラ座標の行列は ``ViewNotApplicable``（router 経由で 400）。

    ``surface`` と同じ理由で断る。カメラ座標では ``active_region`` が
    ``None`` になるため、``generate_projection_image`` に渡す領域が無く、
    ターゲットを投影フレームへ埋め込むこと自体が意味を持たない。
    """
    store.clear()
    targets.clear()
    # The matrix is the 8x6 fixture; the projector frame is not, so it
    # classifies as camera space.
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
    assert "compensation" in detail
    # A "not yet", not a "never": the P2C warp is plan B4's job.
    assert "B4" in detail or "計画" in detail

    # The refusal belongs to this view, not to camera-space matrices at large.
    assert client.get(f"/api/matrix/{matrix_id}/view/coef_tiles").status_code == 200


def test_an_unknown_target_id_is_refused(tmp_path, stand_in_projector_config) -> None:
    client, matrix_id, _, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target="target-does-not-exist"))
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "target-does-not-exist" in detail
    # 404 belongs to the matrix id in the path; an unknown *parameter* value is
    # this view declining the request it was handed.
    assert "target" in detail


def test_a_missing_target_is_refused(tmp_path, stand_in_projector_config) -> None:
    """``target`` を指定しない要求は断る。既定のターゲットは存在しない。

    空欄・空白のみ・パラメータ自体の省略はすべて同じ「未指定」であり、
    「id '  ' は存在しない」ではなく「target=<id> を渡せ」と言わねばならない。
    そこまで見ないと、``_identifier`` の ``.strip()`` を外しても
    *未知のid* の経路が 400 と `"target"` の両方を満たしてしまい、この
    テストは通ってしまう（レビュー Minor 1）。
    """
    client, matrix_id, _, _, _ = _prepared(tmp_path, stand_in_projector_config)
    for url in (
        f"/api/matrix/{matrix_id}/view/compensation",
        _url(matrix_id, target=""),
        _url(matrix_id, target="%20%20"),
    ):
        response = client.get(url)
        assert response.status_code == 400, (url, response.text)
        detail = response.json()["detail"]
        assert "target" in detail
        # The *missing* wording, not the *unknown-id* wording.
        assert "POST /api/target" in detail, (url, detail)
        assert "no target image with id" not in detail, (url, detail)


def test_a_matrix_loaded_under_another_projector_config_is_refused_not_500(
    tmp_path, stand_in_projector_config
) -> None:
    """読み込み後に config が変わった行列は 400 で断る（500 にしない）。

    ``MatrixContext.space`` は**読み込み時**の判定であり、``reload_config``
    は公開APIで、このリポジトリの ``config.toml`` には現在8つの姉妹版が
    作業ツリーに置かれている。フレームが食い違ったまま
    ``generate_projection_image`` へ進むと ``calc_compensation_image`` が
    numpy の形状 ``ValueError`` を投げ、router はそれを（ビューの
    ``ValueError`` ＝サーバのバグという規約どおりに）500 のまま通す。
    利用者から見れば「自分が変えた設定のせいでサーバが壊れた」と告げられる
    ことになる。

    これが ``_render`` の ``classify_space`` 再判定を唯一検証しているテスト
    である。再判定を削ると、この1件だけが落ちる。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    # Sanity: it really was projector space when it was loaded, and it renders.
    assert client.get(f"/api/matrix/{matrix_id}/meta").json()["space"] == "projector"
    assert client.get(_url(matrix_id, target=target_id)).status_code == 200

    # The projector frame changes under a still-valid matrix id.
    stand_in_projector_config(width=_FRAME_WIDTH + 2, height=_FRAME_HEIGHT + 1)

    # The stored classification still says "projector" -- that is the whole
    # point: only a re-check against the *current* config can see the problem.
    assert client.get(f"/api/matrix/{matrix_id}/meta").json()["space"] == "projector"

    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 400, response.text
    assert "compensation" in response.json()["detail"]


def test_inverse_lut_without_a_lut_is_refused(
    tmp_path, stand_in_projector_config
) -> None:
    """``linearizer="inverse_lut"`` は LUT を要求する。

    ``_linearize_inverse_lut`` は ``lut=None`` で ``ValueError`` を投げる。
    それは 500 になる（router は ``ViewNotApplicable`` だけを 400 に写す）
    ので、ビューが先に断らなければならない。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(
        _url(matrix_id, target=target_id, linearizer="inverse_lut")
    )
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "lut" in detail.lower()


def test_inverse_lut_with_an_unknown_lut_id_is_refused(
    tmp_path, stand_in_projector_config
) -> None:
    """未知の LUT id は 400。

    ``luts.clear()`` した直後なので ``"lut_that_was_never_loaded"`` はどの
    ストアにも無い。500（``_linearize_inverse_lut`` の ``ValueError``）では
    なく、次に何をすればよいか分かる 400 で返す。
    """
    from yoshidacomp_web.store import luts

    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    luts.clear()
    response = client.get(
        _url(
            matrix_id,
            target=target_id,
            linearizer="inverse_lut",
            lut="lut_that_was_never_loaded",
        )
    )
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "lut_that_was_never_loaded" in detail
    assert "/api/lut" in detail


@pytest.mark.parametrize("output", ["linear", "code"])
def test_inverse_lut_with_a_loaded_lut_matches_the_pipeline_exactly(
    tmp_path, stand_in_projector_config, output: str
) -> None:
    """計画B3 Task 4 の継ぎ目そのもの: 読み込んだ LUT が実際に使われること。

    ``_require_lut`` は Task 3 まで無条件に断っていた（``NoReturn``）ため、
    ``generate_projection_image(lut=...)`` へ渡る値は常に ``None`` だった。
    ここでは ``POST /api/lut`` で読み込んだ id を渡し、ビューの画素が
    **同じ ``InverseLut`` を渡した** ``generate_projection_image`` の戻り値と
    完全一致することを見る。

    判別力: ``linearizer="gamma"`` の結果とも違うことを併せて確認する。
    「LUT を無視して解析ガンマに落ちる」実装でも、それ単体では
    「``none`` と違う」検査を通ってしまうためである。3本のカーブは形が全部
    違い（``tests/test_web_lut.py`` の ``_LUT_ROWS`` と同じ理由）、終端も
    1.0 ではないので、チャンネルの取り違えも恒等LUTへの退化もここに出る。
    """
    from yoshidacomp.compensation.inverse_lut import InverseLut, save_inverse_lut
    from yoshidacomp_web.store import luts

    client, matrix_id, target_id, matrices, image = _prepared(
        tmp_path, stand_in_projector_config
    )
    luts.clear()

    t_grid = np.array([0.0, 0.2, 0.5, 0.9, 1.0], dtype=np.float32)
    rows = np.array(
        [
            [0.0, 0.11, 0.275, 0.495, 0.55],
            [0.0, 0.22, 0.311, 0.417, 0.44],
            [0.0, 0.028, 0.172, 0.557, 0.688],
        ],
        dtype=np.float32,
    )
    lut_path = tmp_path / "inverse_lut.npz"
    save_inverse_lut(lut_path, InverseLut(lut=rows, t_grid=t_grid, meta={"source": "gray"}))

    load = client.post("/api/lut", json={"path": str(lut_path)})
    assert load.status_code == 200, load.text
    lut_id = load.json()["id"]

    response = client.get(
        _url(
            matrix_id,
            target=target_id,
            linearizer="inverse_lut",
            lut=lut_id,
            output=output,
        )
    )
    assert response.status_code == 200, response.text
    data = _hwc(response)

    expected = _reference(
        image,
        matrices,
        linearizer="inverse_lut",
        output=output,
        lut=InverseLut(lut=rows, t_grid=t_grid, meta={"source": "gray"}),
    )
    inside = _inside_mask()
    np.testing.assert_array_equal(data[inside], expected[inside])
    assert np.isnan(data[~inside]).all()

    # Which of the two outputs the linearizer can even reach, asserted rather
    # than assumed.
    #
    # ``output="linear"`` is the compensation *before* linearization -- the
    # pipeline returns it unchanged whatever ``linearizer`` says -- so the LUT
    # provably cannot alter it, and the exact-match above is a pass-through
    # check there rather than a check of the LUT. ``output="code"`` is the one
    # the LUT actually shapes. Both directions are pinned here, in one place,
    # so neither claim rests on prose.
    #
    # どちらの出力に linearizer が効くのかを、仮定ではなく検証で固定する。
    # ``linear`` は線形化の**手前**であり LUT では変わりえない（上の完全一致は
    # そこでは素通しの検査である）。``code`` が LUT の形が出る側である。
    gamma_reference = _reference(image, matrices, linearizer="gamma", output=output)
    none_reference = _reference(image, matrices, linearizer="none", output=output)
    if output == "linear":
        np.testing.assert_array_equal(expected[inside], none_reference[inside])
    else:
        # The LUT has to be *this* LUT, not the analytic gamma standing in for
        # it, and not the identity.
        # 解析ガンマに落ちる実装とも、恒等に落ちる実装とも区別する。
        assert not np.allclose(expected[inside], gamma_reference[inside], atol=1e-4), (
            "the fixture LUT is too close to the analytic gamma to tell an "
            "implementation that ignores it from one that uses it"
        )
        assert not np.allclose(expected[inside], none_reference[inside], atol=1e-4), (
            "the fixture LUT is too close to the identity to tell an "
            "implementation that skips linearization from one that does it"
        )


@pytest.mark.parametrize("bad_gamma", ["0", "-2.2", "nan"])
def test_a_non_positive_target_gamma_is_refused(
    tmp_path, stand_in_projector_config, bad_gamma: str
) -> None:
    """``target_gamma <= 0`` は入力の誤りであってサーバのバグではない。

    ``decode_target`` は ``ValueError`` を投げ、router はそれを 500 のまま
    通す。数値欄に 0 を打つのは日常的な操作なので、ビューが 400 で断る。
    """
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


# --------------------------------------------------------------------------
# Shape, dtype, labels, display range, registration.
# --------------------------------------------------------------------------


def test_the_body_is_the_projector_frame_as_three_float32_channels(
    tmp_path, stand_in_projector_config
) -> None:
    """``(H, W, 3) float32``、ラベルは3つ。大きさは**投影フレーム**であって
    ターゲット画像の大きさではない（ターゲットは有効領域へリサイズされる）。
    """
    import base64
    import json

    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target=target_id))
    assert response.status_code == 200, response.text
    assert response.headers["X-Kind"] == "rgb"
    assert response.headers["X-Channels"] == "3"
    assert int(response.headers["X-Height"]) == _FRAME_HEIGHT
    assert int(response.headers["X-Width"]) == _FRAME_WIDTH
    # Not the target's own size, and not the active region's.
    assert (_FRAME_HEIGHT, _FRAME_WIDTH) != (_TARGET_HEIGHT, _TARGET_WIDTH)
    assert (_FRAME_HEIGHT, _FRAME_WIDTH) != (_REGION[3], _REGION[2])

    labels = json.loads(
        base64.b64decode(response.headers["X-Labels-B64"]).decode("utf-8")
    )
    assert labels == ["R", "G", "B"]

    data = _hwc(response)
    assert data.shape == (_FRAME_HEIGHT, _FRAME_WIDTH, 3)
    assert data.dtype == np.float32


def test_the_view_result_itself_is_hwc_float32(
    tmp_path, stand_in_projector_config
) -> None:
    """``ViewResult`` を直接見る。router を通すと dtype はワイヤ形式に
    寄せられるため、``(H, W, 3) float32`` という契約はここでしか見えない。
    """
    from yoshidacomp_web.matrix import load_matrix
    from yoshidacomp_web.views import REGISTRY

    _, _, target_id, matrices, image = _prepared(tmp_path, stand_in_projector_config)
    context = load_matrix(tmp_path / "m.npy", get_config().projector)
    result = REGISTRY["compensation"].render(context, {"target": target_id})

    assert result.data.shape == (_FRAME_HEIGHT, _FRAME_WIDTH, 3)
    assert result.data.dtype == np.float32
    assert result.labels == ["R", "G", "B"]
    assert result.vmax > result.vmin

    # A direct call passes no parameter but ``target``, so this is the only
    # place the *in-function* fallbacks are visible: over HTTP the router
    # always fills the declared default, which would hide a drift between the
    # two copies of each default (review Minor 3). Asserting the answer equals
    # the declared defaults' answer pins them equal.
    # HTTP経由では router が宣言側の既定を必ず埋めるため、関数内フォールバック
    # がずれても表に出ない。直接呼び出しだけがそれを見られる場所である。
    inside = _inside_mask()
    np.testing.assert_array_equal(
        result.data[inside],
        _reference(image, matrices, linearizer="none", output="linear",
                   target_decode="srgb", target_gamma=2.2)[inside],
    )


def test_an_unrecognised_output_is_refused_on_a_direct_render_call(
    tmp_path, stand_in_projector_config
) -> None:
    """``output`` の綴り間違いは黙って ``code`` に落ちてはならない。

    HTTP 経由では router が ``choices`` で弾くため到達しないが、
    ``render()`` を直接呼ぶ経路（他のビューや後続タスクからの再利用）では
    到達する。「もっともらしい別の画像」を返すのが最悪の失敗なので、
    ここは断る。
    """
    from yoshidacomp_web.matrix import load_matrix
    from yoshidacomp_web.views import REGISTRY, ViewNotApplicable

    _, _, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    context = load_matrix(tmp_path / "m.npy", get_config().projector)
    with pytest.raises(ViewNotApplicable, match="output"):
        REGISTRY["compensation"].render(
            context, {"target": target_id, "output": "linaer"}
        )


def test_the_display_range_follows_surface(tmp_path, stand_in_projector_config) -> None:
    """``vmin`` は 0 に固定、``vmax`` はデータの99パーセンタイル。

    3チャンネルが1つのレンジを共有するため、床を浮かせると R/G/B から
    同じ定数を引くことになり、露出ではなく彩度が変わる。``surface`` と
    同じ折衷を踏襲する。
    """
    client, matrix_id, target_id, _, _ = _prepared(tmp_path, stand_in_projector_config)
    response = client.get(_url(matrix_id, target=target_id))
    data = _hwc(response)

    assert float(response.headers["X-Vmin"]) == 0.0
    expected_vmax = float(np.nanpercentile(data, 99))
    assert float(response.headers["X-Vmax"]) == pytest.approx(expected_vmax)
    # The masked pixels must not be in the percentile: if they were counted as
    # zeros, 74% of this frame is outside the region and vmax would collapse.
    assert float(response.headers["X-Vmax"]) > float(np.percentile(data[_inside_mask()], 50))


def test_the_view_is_listed_with_its_params() -> None:
    """``/api/views`` の宣言。選択肢はパイプライン側の定義と一致させる。

    ``linearizer`` と ``target_decode`` の選択肢は ``pipeline`` の
    ``LINEARIZERS`` / ``TargetDecodeMode`` と突き合わせる。ここを定数で
    書き写すと、パイプラインに方式が増えたときGUIだけが古いまま静かに
    ずれる。
    """
    store.clear()
    client = TestClient(create_app())
    views = client.get("/api/views").json()["views"]
    spec = next(v for v in views if v["key"] == "compensation")

    assert spec["kind"] == "rgb"
    assert spec["channels"] == 3
    # The picture is the target with the matrix applied, not the matrix, so the
    # readout's second panel says something the 4x3 table cannot.
    assert spec["values_are_matrix"] is False

    params = {p["name"]: p for p in spec["params"]}
    assert set(params) == {
        "target",
        "linearizer",
        "lut",
        "output",
        "target_decode",
        "target_gamma",
    }

    # Runtime ids: free text, no choices. A ``select`` with an empty ``choices``
    # renders an empty dropdown in ``static/js/main.js::buildServerParams`` and
    # could never be given a value at all.
    for name in ("target", "lut"):
        assert params[name]["type"] == "text", name
        assert params[name]["choices"] == [], name
        assert params[name]["default"] == "", name

    assert params["linearizer"]["default"] == "none"
    assert params["linearizer"]["choices"] == list(LINEARIZERS)
    assert set(params["linearizer"]["choices"]) == set(get_args(LinearizerName))

    assert params["output"]["choices"] == ["linear", "code"]
    assert params["output"]["default"] == "linear"

    assert set(params["target_decode"]["choices"]) == set(get_args(TargetDecodeMode))
    assert params["target_decode"]["default"] == "srgb"

    assert params["target_gamma"]["type"] == "number"
    assert params["target_gamma"]["default"] == 2.2

    # Every select's default must be one of its own choices, or the frontend's
    # ``el.value = p.default`` silently selects nothing.
    for p in spec["params"]:
        if p["type"] == "select":
            assert p["default"] in p["choices"], p["name"]
