// The wire contract: the two things a view payload declares about itself, and
// the guards that refuse anything outside them.
//
// `X-Layout` says how the floats are ordered; `X-Kind` says what they mean.
// Both used to be compared against bare string literals scattered across
// api.js, glview.js and main.js -- four sites for the layout, five for the kind
// -- with no validation at any of them. Every one of those comparisons had the
// same shape: `=== "planar"` or `=== "rgb"`, with the *else* branch being the
// other case rather than an error. So an unrecognised value did not fail; it
// silently selected the other behaviour at every site at once and drew a
// viewer that looks like it works.
//
// Collecting them here is not tidiness. It is so that "unknown" has exactly one
// meaning -- refused -- instead of nine independent opportunities to be
// interpreted as the default.
//
// ワイヤ上でペイロードが自分について宣言する2つのこと（`X-Layout` = 並び、
// `X-Kind` = 中身の意味）と、それ以外を断るガード。以前は api.js・glview.js・
// main.js に素の文字列比較が散らばっており（layout 4か所、kind 5か所）、
// どこにも検査が無かった。比較はいずれも `=== "planar"` / `=== "rgb"` の形で、
// else 側は「もう一方」であって「誤り」ではない。したがって未知の値は失敗せず、
// 9か所すべてで黙ってもう一方の挙動を選び、動いているように見えるビューアを
// 描いてしまう。ここに集めるのは整理のためではなく、「未知」の意味を
// 「断る」の1つに固定し、既定値として解釈される機会を9個から0個にするため。

// ------------------------------------------------------------------- layout

// "planar" = (C, H, W); "interleaved" = (H, W, C).
export const LAYOUT_PLANAR = "planar";
export const LAYOUT_INTERLEAVED = "interleaved";
export const LAYOUTS = Object.freeze([LAYOUT_PLANAR, LAYOUT_INTERLEAVED]);

// An absent `X-Layout` is not an error -- it is what a pre-planar endpoint
// looks like, and api.js deliberately reads it as interleaved so such an
// endpoint still renders. A *present* value outside the set is a different
// thing entirely: something produced it on purpose and got it wrong.
//
// The failure this refuses is silent by construction. "Planar" with a capital
// P, or a header a proxy rewrote, fails every `=== "planar"` test and routes
// all four consumers to the interleaved walk at once. Nothing throws. Every
// channel is attributed to the wrong tile, the hover readout agrees with the
// wrong tiles, and the per-tile ranges agree with both -- three independent
// readings that corroborate each other because they are all wrong the same way.
//
// ヘッダが無いのは誤りではない（planar 化前のエンドポイントの見え方であり、
// api.js は意図的にインタリーブとして読む）。しかし値が有って集合の外にある
// のは別物で、誰かが意図して作って間違えたということである。"Planar" のような
// 大文字違いは `=== "planar"` を全て外し、4つの読み手が一斉にインタリーブ側へ
// 落ちる。例外は出ず、タイル・読み取り値・タイル別レンジの3者が「同じ間違い方」
// で一致するため、互いに裏付け合ってしまう。
export function requireKnownLayout(layout, where) {
  if (!LAYOUTS.includes(layout)) {
    throw new Error(
      `${where} の X-Layout="${layout}" はこのビューアの知らない配置です` +
      `（扱えるのは "${LAYOUTS.join('" / "')}" のみ）。`
    );
  }
}

// --------------------------------------------------------------------- kind

// Everything the frontend does differently per kind, in one row each.
//
// Five sites used to decide this independently -- the supported-kind list, the
// tile layout, the range calculation, the colormap control and the texture
// upload -- and each defaulted to scalar for anything it did not recognise.
// Adding "categorical" to a list was therefore one token that silently enabled
// four wrong behaviours at once: label-derived axes, per-channel +/- ranges, an
// active diverging colormap, and an R32F upload of class-index data. Nothing
// fails; the viewer paints plausible pixels for a view it does not understand.
// The next plan adds `compensation` and `gamut`.
//
// With the traits in a table, an unlisted kind has no row to read and
// `traitsFor` throws. Adding a kind means writing down what it does at every
// one of these decisions, which is the point.
//
// kind ごとに違う挙動を1行ずつ並べた表。以前は「対応kind一覧」「タイル配置」
// 「レンジ計算」「カラーマップの有効/無効」「テクスチャのアップロード」の
// 5か所が独立に判断し、いずれも知らない kind を scalar として扱っていた。
// 一覧に "categorical" を足すだけで、ラベル由来の軸・チャンネル別±レンジ・
// 発散カラーマップ・クラス番号のR32Fアップロードという4つの誤った挙動が
// 同時に有効になり、しかも何も失敗しない。表にしておけば、未登録の kind は
// 読む行が無く `traitsFor` が投げる。kind を足すとは、これらの判断すべてに
// ついて挙動を書き下すことである、という形にする。
export const KIND_TRAITS = Object.freeze({
  scalar: Object.freeze({
    // One tile per channel: 12 coefficients are 12 unrelated numbers.
    singleTile: false,
    // Each channel gets its own display range.
    sharedRange: false,
    // A number needs a colormap to become a colour.
    colormap: true,
    // One plane per texture, uploaded as a single float channel.
    packed: false,
    texture: Object.freeze({ internal: "R32F", format: "RED" }),
    // shaders.js: 0 selects the colormap branch.
    uMode: 0,
  }),
  rgb: Object.freeze({
    // Three channels that are one picture, not three pictures.
    singleTile: true,
    // ...so they must share one range, or R/G/B get an independent white
    // balance the data does not have.
    sharedRange: true,
    // Already a colour; the shader's rgb branch has no colormap in it.
    colormap: false,
    // Three planes packed interleaved into one texture.
    packed: true,
    texture: Object.freeze({ internal: "RGB32F", format: "RGB" }),
    // shaders.js: 1 selects the straight-through rgb branch.
    uMode: 1,
  }),
});

export const SUPPORTED_KINDS = Object.freeze(Object.keys(KIND_TRAITS));

// Falling through for an unimplemented kind would paint `categorical` data
// with the diverging scalar colormap: plausible pixels for a view this
// frontend does not understand, which is the same class of silent misreading
// the NaN mask exists to prevent.
// 未対応の kind をそのまま描くと、理解していないデータを「それらしい」色で
// 表示してしまう。黙って通さず、はっきり失敗させる。
export function requireSupportedKind(kind, where) {
  if (!SUPPORTED_KINDS.includes(kind)) {
    throw new Error(
      `${where} の kind="${kind}" はこのビューアでは未対応です` +
      `（対応しているのは kind="${SUPPORTED_KINDS.join('" / kind="')}" のみ）。`
    );
  }
}

// The only way to read the table: unknown kinds cannot reach a default row.
// 表を読む唯一の経路。未知の kind が既定行へ落ちることはない。
export function traitsFor(kind, where) {
  requireSupportedKind(kind, where);
  return KIND_TRAITS[kind];
}
