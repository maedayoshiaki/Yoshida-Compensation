// GLSL ES 3.00 shaders for the scalar float-texture tile view.
//
// The vertex shader is unchanged from the reference implementation: it maps a
// unit quad through (scale, translate) in device pixels, so pan/zoom never
// re-rasterises anything on the CPU.
//
// The fragment shader reads one R32F channel, paints NaN with the mask colour
// (values outside the active projection region are NaN on purpose -- the fit is
// physically meaningless there), then colour-maps, applies exposure/gamma, the
// sRGB OETF and a dither so an 8-bit canvas does not band.
//
// 頂点シェーダは参照実装のまま。フラグメントシェーダは R32F の1チャンネルを読み、
// NaN をマスク色にしてからカラーマップ・露出・γ・sRGB OETF・ディザを掛ける。

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
uniform int   u_colormap;   // 0 = diverging, 1 = grayscale
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
bool isNaNBits(float x) {
    uint b = floatBitsToUint(x);
    return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

void main() {
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
}`;
