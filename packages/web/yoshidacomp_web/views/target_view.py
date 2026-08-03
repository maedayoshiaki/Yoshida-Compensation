"""What every view that renders a target image through the matrix shares.

``compensation`` と ``gamut`` は同じ問いの前半を共有する: どのターゲットを、
どう復号して、どの座標空間の行列に通し、どの有効領域の外を「答え無し」に
するか。違うのは最後の一手だけで、``compensation`` は補償値をそのまま描き、
``gamut`` はそれを4クラスに分ける。

ここに集めるのは、二重に書けば静かにずれるものに限る -- 既定値、パラメータ
宣言、断り文、座標空間の判定（``require_projector_space``）、有効領域のマスク。
**数式は1つも無い**（補償の計算は ``pipeline.generate_projection_image()`` が
持つ）。

なぜ写しではなく共有なのか: このリポジトリは同じ値を2箇所に書いた結果ずれる
欠陥を繰り返し記録している（``views/compensation.py`` の ``_DEFAULTS``、
``store.py`` の ``_IdStore``）。断り文はとくにそうで、テストが本文の一部を
文字列で見ているため、片方だけ直すと「もう片方は古い案内を返し続けるが
テストは緑」になる。ビュー名だけを引数にした1つの雛形にしておけば、その形の
ずれは起こりようがない。

ビュー名を引数に取るのは、断り文が必ず「どのビューが断ったか」を名乗るため
である。``routers/view.py`` はこの本文をそのまま 400 の ``detail`` にし、
``static/js/api.js::fetchView`` がそれをステータス行へ出す。利用者が見る唯一の
手掛かりなので、「compensation」と書いてある文が ``gamut`` から返ってはならない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import get_args

import numpy as np

from yoshidacomp.compensation.pipeline import TargetDecodeMode
from yoshidacomp.config import ProjectorConfig

from ..matrix import MatrixContext, classify_space
from ..store import targets
from .base import ParamSpec, ViewNotApplicable

# The target-related defaults, in one place for every view that has them.
#
# Each of these used to be written twice inside ``compensation`` -- once in the
# ``ParamSpec`` the GUI reads and once as the fallback in ``params.get(name,
# ...)`` -- and only the declared copy was pinned by a test. Over HTTP the
# router always fills the declared default, so the fallback is unreachable
# there and the two could drift apart silently; a direct ``render()`` call,
# which both views support, would then get the drifted value. One dict, read by
# the declaration and by the reader, removes the question -- and now removes it
# across views as well as within one.
#
# 既定値は1箇所にまとめる。宣言側とフォールバック側に二重に書くと、router が
# 常に宣言側を埋めるためフォールバックは HTTP からは到達不能になり、静かに
# ずれても誰も気付かない。``render()`` を直接呼ぶ経路だけがそのずれた値を
# 受け取る。
TARGET_DEFAULTS: dict[str, object] = {
    "target": "",
    "target_decode": "srgb",
    "target_gamma": 2.2,
}


def _camera_space_refusal(view: str) -> str:
    """Why *view* will not answer for a camera-space matrix.

    Correctness first, unlike ``views/surface.py`` where cost came first:
    outside projector space ``active_region`` is ``None``, and it is not
    optional for these views -- ``generate_projection_image`` needs it to place
    the target inside the frame at all. There is no meaningful answer to
    substitute, because "the target image resized into the projector's active
    rectangle" is not a thing a camera-space matrix has. The cost argument is
    ``surface``'s and applies unchanged: the work is set by the pixel count,
    and ``data/color_mixing_matrices.npy`` (4660x6984 = 32.5M pixels) is 15.7x
    the 1080p projector frame.

    カメラ座標では ``active_region`` が ``None`` になり、これらのビューに
    とってそれは省略可能な情報ではない（``generate_projection_image`` は
    ターゲットをフレームのどこへ置くかをそれで決める）。「投影領域へリサイズ
    したターゲット」という概念自体がカメラ座標の行列には無いので、代わりに
    描けるものが無い。コスト面の理由は ``surface`` と同じで、画素数だけで
    決まる。
    """
    return (
        f'view "{view}" needs a projector-space matrix; this one is in camera '
        "space, where there is no active region to place the target image in. "
        # **B5, not B4.** B4 is the plan this view shipped in, and it defers the
        # warp to B5 (a background job, `POST /api/matrix/{id}/warp`). Naming B4
        # here would point every user at a plan that has already shipped without
        # the feature -- and this sentence is the only pointer inside the
        # product.
        # ここは B5 である（B4 はこのビューが載った計画そのもので、ワープは
        # B5 へ送られている）。製品内で唯一の案内なので、済んだ計画を指しては
        # ならない。
        "Warping a camera-space matrix into projector space is planned for "
        "B5 -- until then, load a matrix whose frame matches the configured "
        "projector. "
        f"（{view} はプロジェクタ座標の行列を必要とします。カメラ座標では"
        "ターゲットを収める有効領域が存在しません。カメラ座標からのワープは"
        "計画B5の予定です。）"
    )


def active_region_mask(
    height: int, width: int, region: tuple[int, int, int, int]
) -> np.ndarray:
    """A ``(H, W)`` boolean: True inside ``(x, y, w, h)``.

    有効領域の内側を True にした真偽値配列。x/y と w/h の取り違えはここ1箇所
    だけで起こりうる（各ビューで書くと、ビューの数だけ起こりうる）。
    """
    x, y, w, h = region
    mask = np.zeros((height, width), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


# The three parameter declarations, as builders rather than a ready-made list:
# ``compensation`` interleaves its own ``linearizer``/``lut``/``output``
# between them, and the order of ``params`` is the order the GUI draws the
# controls in.
# 3つの宣言は「まとめたリスト」ではなく個別の関数にする。``compensation`` は
# この間に自分のパラメータを挟むうえ、``params`` の並びがそのままGUIの
# コントロールの並びになるためである。


def target_spec(defaults: dict[str, object]) -> ParamSpec:
    """The ``target`` control: free text, because ids only exist at runtime.

    ``ParamSpec.choices`` は静的な ``list[str]`` で ``ViewSpec`` は frozen
    なので、有効なidの一覧を ``select`` で表すことはできない。空の
    ``choices`` を持つ ``select`` は ``static/js/main.js::buildServerParams``
    で選択肢ゼロのドロップダウンになり、値を入れる手段が無い。詳細は
    ``views/base.py::ParamSpec.type``。
    """
    return ParamSpec(
        name="target", label="ターゲットID", type="text", default=defaults["target"]
    )


def target_decode_spec(defaults: dict[str, object]) -> ParamSpec:
    """The ``target_decode`` control. Choices come from the pipeline's own type."""
    return ParamSpec(
        name="target_decode",
        label="ターゲット復号",
        type="select",
        default=defaults["target_decode"],
        # Taken from the pipeline's own ``Literal`` rather than written out
        # again: a mode added there appears here without a second edit, and
        # cannot be offered here without existing there.
        choices=list(get_args(TargetDecodeMode)),
    )


def target_gamma_spec(defaults: dict[str, object]) -> ParamSpec:
    return ParamSpec(
        name="target_gamma",
        label="ターゲットγ",
        type="number",
        default=defaults["target_gamma"],
    )


# ``eq=False`` so that no ``__hash__`` is synthesised and ``object.__hash__``
# is inherited instead. ``frozen=True`` with ``eq=True`` synthesises a
# ``__hash__`` that hashes every field, and one of the fields is a ``dict`` --
# so the synthesised one raises ``TypeError: unhashable type: 'dict'`` for
# every instance this module creates. A method that cannot be called on any
# instance is not a guarantee, it is a trap left for the first caller who puts
# a ``ViewParams`` in a set. Identity equality is also the right semantics
# here: two ``ViewParams`` for the same view are the same reader, and nothing
# compares them.
# ``frozen=True`` かつ ``eq=True`` だと全フィールドを見る ``__hash__`` が
# 生成されるが、フィールドに ``dict`` があるため生成された ``__hash__`` は
# どのインスタンスでも ``TypeError`` になる。呼べないメソッドは保証ではなく罠。
@dataclass(frozen=True, eq=False)
class ViewParams:
    """Reads one view's parameters, and refuses in that view's name.

    Attributes:
        view: The view key, used in every refusal message so the user is told
            which view declined and why.
        defaults: That view's declared defaults -- the same dict its
            ``ParamSpec``s are built from. Read here so a direct ``render()``
            call and an HTTP request cannot disagree.
    """

    view: str
    defaults: dict[str, object]

    def require_projector_space(
        self, context: MatrixContext, projector: ProjectorConfig
    ) -> tuple[int, int, tuple[int, int, int, int]]:
        """Return ``(height, width, active_region)``, or refuse.

        Two clauses, both load-bearing, and deliberately not three.

        ``region is None`` *is* "the matrix is in camera space": ``matrix.py``
        sets ``active_region`` from ``resolve_active_region`` only when the
        space is ``"projector"``, and to ``None`` otherwise, so testing
        ``context.space == "camera"`` as well says the same thing a second
        time. This condition used to carry that third clause; it was removed
        after a review measured that reducing the whole condition to ``region
        is None`` left every test green. ``region is None`` is the clause kept
        because it is also what narrows ``region`` to a tuple for the callers.

        The ``classify_space`` call is the one that adds behaviour, and it is a
        *different* question: ``context.space`` was decided when the matrix was
        loaded, against the config in force **then**. Re-deciding it against
        the config in force **now** is what keeps a mismatched frame from
        reaching ``calc_compensation_image`` as a numpy shape ``ValueError`` --
        a 500 blaming the server for a configuration the user changed under a
        matrix id that is still perfectly valid. ``reload_config`` is a public
        API and this repository's ``config.toml`` has eight sibling variants in
        the working tree right now. Pinned by
        ``test_a_matrix_loaded_under_another_projector_config_is_refused_not_500``
        in *both* views' test files; dropping the ``classify_space`` clause
        fails one in each.

        This lives here rather than in each view because the two copies were
        verbatim, comment included, and this module exists for exactly that.

        2条件であって3条件ではない。``region is None`` は「カメラ座標である」
        ことそのもの（``matrix.py`` がプロジェクタ座標のときだけ領域を入れる）
        なので、``context.space == "camera"`` を併記しても同じことを2度言う
        だけである。``classify_space`` の方は別の問いで、読み込み時ではなく
        **現在の** config との一致を見る。食い違ったまま進むと numpy の形状
        ``ValueError``（＝500）になるが、それはサーバのバグではない。

        Raises:
            ViewNotApplicable: the matrix is in camera space, or its frame no
                longer matches the configured projector.
        """
        height, width = int(context.array.shape[0]), int(context.array.shape[1])
        region = context.active_region
        if region is None or classify_space(height, width, projector) == "camera":
            raise ViewNotApplicable(_camera_space_refusal(self.view))
        return height, width, region

    def value(self, params: dict, name: str) -> object:
        """Read *name* from *params*, falling back to its declared default."""
        value = params.get(name)
        return self.defaults[name] if value is None else value

    def identifier(self, params: dict, name: str) -> str:
        """Read a runtime id parameter as a stripped string.

        ``""`` means "not given". The router supplies the declared default for
        a parameter absent from the query string, so an omitted ``target`` and
        an empty box arrive here identically -- and both mean the same thing.
        ``.strip()`` for the same reason: a box the user cleared with the space
        bar is an empty box, and telling them "no target with id '  '" instead
        of "pass target=<id>" answers a question they did not ask.
        未指定と空欄は同じ意味なので同じ ``""`` に落とす。空白だけの入力も
        同様で、「id '  ' は存在しない」ではなく「target=<id> を渡せ」と
        言うべきである。
        """
        return str(self.value(params, name)).strip()

    def require_target(self, params: dict) -> np.ndarray:
        """Return the selected target image, or refuse.

        Raises:
            ViewNotApplicable: ``target`` was not given, or names a target that
                is not loaded. Both are the client asking for something this
                server does not have, not a bug -- ``routers/view.py`` turns
                them into a 400 carrying this message, which is where the user
                is told to POST the image first.
        """
        target_id = self.identifier(params, "target")
        if not target_id:
            raise ViewNotApplicable(
                f'view "{self.view}" needs a target image: pass target=<id>, '
                "the id returned by POST /api/target. "
                f"（{self.view} は target=<id> を必要とします。先に "
                "POST /api/target で画像を読み込んでください。）"
            )
        try:
            return targets.get(target_id).image
        except KeyError:
            raise ViewNotApplicable(
                f'view "{self.view}": no target image with id {target_id!r} is '
                "loaded. POST /api/target to load one and use the id it "
                f"returns. （target {target_id!r} は読み込まれていません。）"
            ) from None

    def target_decode(self, params: dict) -> tuple[str, float]:
        """Read the target's transfer function and its exponent, or refuse.

        The view owns the decode mode, and deliberately so. A target ``.png``
        is stored as raw uint8 by ``target.py`` because ``decode_target`` needs
        a ``mode`` and a ``gamma`` that are a property of *this request*, not
        of the file: the same image is a different compensation input
        depending on how it was authored. The names, the choices and the
        defaults are the CLI's (``rebuild_compensation_with_map.py``'s
        ``--target-decode`` / ``--target-gamma``, defaulting to ``srgb`` /
        ``2.2``), so a GUI reading and a command line are comparable without a
        translation table.

        復号方式はビューが持つ。``target.py`` が uint8 の生画素だけを保持して
        いるのは、``mode``/``gamma`` がファイルの性質ではなく要求ごとの選択
        だからである。名前・選択肢・既定値は CLI に揃えてあるので、GUIの
        見た目とCLIの出力を突き合わせられる。

        Raises:
            ViewNotApplicable: ``target_decode="gamma"`` with a non-positive or
                non-finite exponent. ``decode_target`` raises ``ValueError``
                for those, which would be a 500; typing 0 into a number box is
                a routine mistake, not a server bug.
        """
        mode = str(self.value(params, "target_decode"))
        gamma = float(self.value(params, "target_gamma"))  # type: ignore[arg-type]
        if mode == "gamma" and not (math.isfinite(gamma) and gamma > 0.0):
            raise ViewNotApplicable(
                f'view "{self.view}": target_gamma must be a finite positive '
                f"number, got {gamma!r}. "
                "（target_gamma は正の有限値である必要があります。）"
            )
        return mode, gamma
