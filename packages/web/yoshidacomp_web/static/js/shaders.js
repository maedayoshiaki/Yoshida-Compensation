// GLSL ES 3.00 shaders for the float-texture tile view.
//
// The vertex shader is unchanged from the reference implementation: it maps a
// unit quad through (scale, translate) in device pixels, so pan/zoom never
// re-rasterises anything on the CPU.
//
// The fragment shader has three modes, selected by u_mode:
//
//   0 (scalar) reads one R32F channel, paints NaN with the mask colour (values
//     outside the active projection region are NaN on purpose -- the fit is
//     physically meaningless there), then colour-maps, applies exposure/gamma,
//     the sRGB OETF and a dither so an 8-bit canvas does not band.
//   1 (rgb) reads all three components of an RGB32F texture, masks the pixel if
//     any of them is NaN, and takes the same tone chain with no colormap: the
//     input is already a colour.
//   2 (categorical) reads one R32F channel holding a *class index*, masks NaN
//     the same way, and looks the colour up in a discrete palette. No colormap
//     and no tone chain at all -- see CLASS_COLORS below.
//
// 頂点シェーダは参照実装のまま。フラグメントシェーダは u_mode で3つのモードを
// 持つ。0 は R32F の1チャンネルを読み、NaN をマスク色にしてからカラーマップ・
// 露出・γ・sRGB OETF・ディザを掛ける。1 は RGB32F の3成分を読み、いずれかが
// NaN ならマスクし、カラーマップを通さず同じトーンチェーンを掛ける。2 は
// R32F のクラス番号を読み、離散パレットを引く（カラーマップもトーンチェーンも
// 通さない）。

// The discrete palette: class index -> colour, as the framebuffer wants it.
//
// **These are display-referred sRGB values and the shader writes them out
// untouched** -- no exposure, no gamma, no OETF, no dither. That is not a
// shortcut, it is what makes the legend trustworthy: the swatch beside
// "1超クリップ" in the sidebar is `classColorCss(2)`, and it is the same colour
// as the pixels only because nothing transforms them on the way to the
// framebuffer. Put the tone chain back and the swatch is a claim about the
// picture rather than a copy of it. See KIND_TRAITS.categorical.toneChain in
// contract.js for the other half of that argument.
//
// Four colours that stay four colours in large flat regions. Measured over 12
// real matrix x photo pairs, 46.5-85.0% of pixels are out of gamut, so this
// view is normally *mostly* not class 0: a palette that only reads when a few
// pixels are flagged (a neutral field with one accent) would be the wrong
// choice here. These are Okabe-Ito qualitative colours. Separations below are
// measured on the bytes ANGLE/D3D11 actually wrote, read back with
// gl.readPixels, not computed from these literals:
//
//   closest pair (class 2 vs 3, orange vs vermillion)  CIEDE2000 22.2
//   next closest (class 0 vs 1)                        CIEDE2000 38.4
//   nearest class to the mask colour (class 1)         CIEDE2000 44.2
//
// NAN_COLOR (magenta, 0.85/0.1/0.75) is deliberately outside the family, so
// "no answer" cannot be read as a class. Under simulated dichromacy (Vienot
// 1999) the minimum pairwise CIE76 separation falls from 33.4 to 26.8
// (protanopia) and 19.6 (deuteranopia), which is still legible; tritanopia is
// the weak case at 10.2 and is not designed for here.
//
// 距離は実測値（ANGLE/D3D11 が書いたバイトを readPixels で読み戻したもの）で
// あり、この定数から計算したものではない。最接近対は 22.2、マスク色は最も近い
// クラスからも 44.2 離れている。P/D型色覚では最小 26.8 / 19.6、T型は 10.2。
//
// The index *is* the class index, so the order is a wire-facing decision and
// not a palette-designer's: views/gamut.py's LABELS are
// ["範囲内", "負クリップ", "1超クリップ", "両方"], and green / blue / orange /
// vermillion line up with them as in-gamut, dark clip, bright clip, both.
//
// 離散パレット。**表示色そのもの**を書き出し、露出もγもOETFもディザも掛け
// ない。凡例のスウォッチ（classColorCss）が画素と同じ色であることの根拠が
// これである。実データでは46.5〜85.0%が色域外＝大半がクラス0以外なので、
// 「少数の画素だけが目立つ」型のパレットは選べない。Okabe-Ito の定性色で、
// 最接近対（橙と朱）でも CIEDE2000 で 20.6 離れている。マスク色（マゼンタ）
// は4色すべてから 32 以上離してある。番号はクラス番号そのものなので並びは
// 表示上の都合ではない。
export const CLASS_COLORS = Object.freeze([
  Object.freeze([0.0000, 0.6196, 0.4510]),   // #009E73 bluish green
  Object.freeze([0.0000, 0.4471, 0.6980]),   // #0072B2 blue
  Object.freeze([0.9020, 0.6235, 0.0000]),   // #E69F00 orange
  Object.freeze([0.8353, 0.3686, 0.0000]),   // #D55E00 vermillion
]);

// One palette entry as a CSS colour, quantised the way the framebuffer
// quantises it -- the same relationship nanColorCss() has to NAN_COLOR, and
// for the same reason: the legend must be *derived* from the numbers the
// shader draws with, never a second copy of them.
// パレット1色をCSS色にする。nanColorCss と同じ関係（凡例は写しではなく導出）。
export function classColorCss(index) {
  const c = CLASS_COLORS[index];
  if (!c) return null;
  const to8 = (v) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
  return `rgb(${to8(c[0])}, ${to8(c[1])}, ${to8(c[2])})`;
}

// The same numbers, as the body of the shader's classColor().
//
// Generated rather than typed out below, so the GLSL cannot disagree with
// CLASS_COLORS -- the disagreement being invisible by construction, since a
// palette that is wrong in the shader and right in the legend still draws four
// distinct colours and still labels four rows.
//
// An if-chain rather than an indexed array: ESSL 3.00 permits dynamic indexing
// of a const array, but this is the one path in this file with no fallback if
// a driver disagrees, and the chain compiles to the same thing everywhere.
// GLSL側は CLASS_COLORS から生成する（写しを作らない）。配列の動的添字では
// なく if の連鎖にしてあるのはドライバ差を持ち込まないため。
const CLASS_PALETTE_GLSL = CLASS_COLORS
  .map((c, i) => `    if (i == ${i}) return vec3(${c.map((v) => v.toFixed(4)).join(", ")});`)
  .join("\n");

export const VERT_SRC = `#version 300 es
in vec2 a_uv;
uniform vec2 u_imageSize;   // (W,H) in image pixels
uniform float u_scale;      // device px per image px
uniform vec2 u_translate;   // device px offset of image top-left
uniform vec2 u_resolution;  // canvas size in device px
out vec2 v_uv;
void main() {
  v_uv = a_uv;
  vec2 screen = a_uv * u_imageSize * u_scale + u_translate;
  vec2 clip = screen / u_resolution * 2.0 - 1.0;
  clip.y = -clip.y;               // screen y-down -> clip y-up
  gl_Position = vec4(clip, 0.0, 1.0);
}`;

export const FRAG_SRC = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform float u_vmin;
uniform float u_vmax;
uniform int   u_mode;       // 0 = scalar (R32F + colormap), 1 = rgb (RGB32F),
                            // 2 = categorical (R32F class index + palette)
uniform int   u_colormap;   // 0 = diverging, 1 = grayscale (scalar mode only)
uniform float u_exposure;   // EV
uniform float u_gamma;
uniform float u_dither;
uniform vec3  u_nanColor;

// Blue -> white -> red. Zero sits at white when the range is symmetric.
// 青→白→赤。レンジが対称なら0が白に来る。
vec3 diverging(float t) {          // t in [-1, 1]
    float a = clamp(abs(t), 0.0, 1.0);
    vec3 cold = vec3(0.23, 0.30, 0.75);
    vec3 warm = vec3(0.71, 0.02, 0.15);
    vec3 tip  = (t < 0.0) ? cold : warm;
    return mix(vec3(1.0), tip, a);
}

// Standard sRGB OETF. Applied once, immediately before quantisation.
// 標準sRGB OETF。量子化の直前に一度だけ掛ける。
vec3 srgbEncode(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// NaN test on the IEEE-754 bit pattern: exponent all ones, mantissa non-zero.
//
// Neither isnan(v) nor (v != v) survives here. Measured on this machine
// (Chrome 150 / ANGLE D3D11 / RTX 3060 Ti) both were folded away, and the
// masked region rendered as fully saturated blue -- a *plausible* strong
// negative coefficient, which is precisely the misreading the NaN mask exists
// to prevent. Integer ops are not subject to float fast-math, so this holds.
// isnan() も (v != v) も実測で最適化により消え、有効領域外が「濃い負の値」に
// 見えてしまった。整数演算なら fast-math の影響を受けないため確実に判定できる。
// highp is not optional here. ESSL 3.00 gives the fragment language a default
// of mediump for int (and uint follows int), and only float is declared highp
// above. Desktop ANGLE happens to implement mediump int as 32-bit, but where it
// is 16-bit the 0x7F800000u mask truncates and every NaN reads as finite --
// reintroducing the exact plausible-wrong-pixel bug this function exists to fix.
// highp は必須。ESSL 3.00 の int/uint 既定は mediump で、16bit 実装では
// マスクが切り捨てられ NaN 判定が壊れる。
bool isNaNBits(float x) {
    highp uint b = floatBitsToUint(x);
    return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

// Declared here, defined *below* main(). The order is deliberate and load
// bearing: tests/test_web_golden.py pins the scalar tone chain by the position
// of the *first* occurrence of each of its lines in this source, and the rgb
// branch repeats those same lines verbatim. Defining rgbColor() above main()
// would move every first occurrence into it, and the scalar path's ordering
// would silently stop being pinned by anything.
// 前方宣言。定義を main() より後ろに置くのは意図的で、ゴールデンテストが
// 「最初に現れた位置」でスカラー経路の順序を固定しているため。rgb経路は同じ
// 行を繰り返すので、上に置くと固定対象がそちらへ移ってしまう。
//
// categoricalColor() is declared here and defined *after* rgbColor(), for the
// mirror image of the same reason: that test reads "the rgb branch" as
// everything from rgbColor()'s definition to the end of the file, and pins its
// tone chain by first occurrence inside that region too. A categorical
// definition wedged between main() and rgbColor() would fall into neither
// region, which is a quieter way of going unpinned than being pinned to the
// wrong lines.
// categoricalColor() を rgbColor() の後ろに定義するのも同じ理由の裏返しで、
// あいだに挟むとどちらの検査範囲にも入らない。
vec3 rgbColor();
vec3 categoricalColor();

void main() {
    if (u_mode == 1) { fragColor = vec4(rgbColor(), 1.0); return; }
    if (u_mode == 2) { fragColor = vec4(categoricalColor(), 1.0); return; }
    float v = texture(u_tex, v_uv).r;
    if (isnan(v) || isNaNBits(v)) { fragColor = vec4(u_nanColor, 1.0); return; }

    float span = max(u_vmax - u_vmin, 1e-9);
    float t = (u_colormap == 0)
        ? clamp((2.0 * (v - u_vmin) / span) - 1.0, -1.0, 1.0)   // [-1, 1]
        : clamp((v - u_vmin) / span, 0.0, 1.0);                 // [0, 1]

    vec3 rgb = (u_colormap == 0) ? diverging(t) : vec3(t);
    rgb *= exp2(u_exposure);
    rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)));
    rgb = srgbEncode(rgb);

    // Dither before the driver quantises to 8 bit, so gradients do not band.
    // ドライバが8bitへ量子化する前にディザを入れ、階調のバンディングを防ぐ。
    rgb += (hash12(gl_FragCoord.xy) - 0.5) * (u_dither / 255.0);
    fragColor = vec4(rgb, 1.0);
}

// rgb mode: three channels that are one colour.
//
// There is no colormap here, and putting one back would be a mistake rather
// than a style choice: the scalar map exists to give a *number* a colour, and
// this input is already a colour. The range maps straight onto linear RGB and
// then takes the same exposure -> gamma -> OETF -> dither chain as above.
//
// Any single NaN component masks the whole pixel. The producing view writes
// NaN to all three at once (a pixel with no invertible A has no answer in any
// channel), so a partial NaN is not a state the backend can produce; and
// painting two real components alongside one invented one would be exactly the
// plausible-but-wrong pixel the mask exists to prevent -- worse here than in
// scalar mode, because a wrong hue looks like data.
//
// isNaNBits, not isnan(), for the reason spelled out on isNaNBits itself: on
// ANGLE/D3D11 the float test folds away. isnan() is kept alongside it only as
// the same belt-and-braces the scalar path uses.
//
// rgbモードは「1つの色である3チャンネル」。カラーマップは通さない（数値に色を
// 与えるための仕組みであり、入力が既に色ならば不要どころか有害）。1成分でも
// NaN なら画素全体をマスクする。元ビューは3成分同時に NaN を書くので部分的な
// NaN は発生せず、2成分だけ本物の色を描くのはもっともらしい誤りになる。
vec3 rgbColor() {
    vec3 v = texture(u_tex, v_uv).rgb;
    if (any(isnan(v)) || isNaNBits(v.r) || isNaNBits(v.g) || isNaNBits(v.b)) {
        return u_nanColor;
    }

    float span = max(u_vmax - u_vmin, 1e-9);
    vec3 rgb = clamp((v - vec3(u_vmin)) / span, 0.0, 1.0);
    rgb *= exp2(u_exposure);
    rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)));
    rgb = srgbEncode(rgb);
    rgb += (hash12(gl_FragCoord.xy) - 0.5) * (u_dither / 255.0);
    return rgb;
}

// The palette, generated from CLASS_COLORS in this file's JavaScript.
//
// Out-of-range indices return the mask colour rather than the nearest class.
// The producing view cannot emit one -- routers/view.py pins
// vmax == len(labels) - 1 and the frontend refuses more classes than the
// palette has -- so this branch is for the case where that stops being true,
// and the answer it gives then is "no answer" rather than a class the pixel is
// not. Clamping instead would report every unknown index as the last class,
// which is a picture.
// 範囲外の番号は最寄りのクラスではなくマスク色にする。クランプすると未知の
// 番号がすべて最後のクラスとして描かれ、それは「絵」になってしまう。
vec3 classColor(int i) {
${CLASS_PALETTE_GLSL}
    return u_nanColor;
}

// categorical mode: one channel holding a class index.
//
// No colormap and no tone chain, and neither is an omission. A colormap turns
// a number into a colour by placing it on a continuum, and class 1 is not
// between class 0 and class 2 -- it is not on a line at all. The tone chain is
// worse than useless here: exposure and gamma would move the palette away from
// the colours the sidebar legend is drawing, so the legend would stop
// explaining the picture the moment anyone touched a slider. u_exposure,
// u_gamma and u_dither are therefore not read anywhere below, which is the
// half of that decision a driver cannot opt out of; main.js disables the
// controls, which is the half the user can see. See
// KIND_TRAITS.categorical.toneChain in contract.js.
//
// isNaNBits, not isnan(), for the reason spelled out on isNaNBits itself. It
// matters more here than anywhere else in this file: outside the active
// projection region the pixel has no class, and 0.0 -- what a folded-away NaN
// test leaves in play once the value is rounded -- is a perfectly good class 0,
// "範囲内". The failure would be a frame reporting that the compensation is
// representable exactly where there is no compensation at all.
//
// The rounding is nearest, not truncation: the value arrives as an exact
// float32 whole number through an unfiltered R32F texture, so any drift is a
// bug rather than a quantity to floor, and floor() would turn a hypothetical
// 0.9999997 into class 0 instead of class 1.
//
// クラス番号を持つ1チャンネル。カラーマップもトーンチェーンも通さない。
// クラスは連続量ではなく、露出やγを掛けるとパレットが凡例から離れてしまう
// （スライダを触った瞬間に凡例が絵を説明しなくなる）。u_exposure /
// u_gamma / u_dither は以下のどこからも読まない。NaN 判定にビットパターンを
// 使う理由はここで最も重い——有効領域外が黙って「範囲内」になる。
vec3 categoricalColor() {
    float v = texture(u_tex, v_uv).r;
    if (isnan(v) || isNaNBits(v)) return u_nanColor;
    return classColor(int(floor(v + 0.5)));
}`;
