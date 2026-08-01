// WebGL2 float-texture image view. Ported from the hyperspectral-color
// reference implementation; see the report for the exact diff.
//
// Why this shape, restated because it is easy to undo by accident:
//  - HiDPI blur : the canvas backing store is sized in *device* pixels
//    (clientSize * devicePixelRatio) and the image is a GPU quad, so it renders
//    at native resolution instead of an upscaled fixed-dpi raster. Removing the
//    devicePixelRatio term in resize() is what makes 150% Windows scaling look
//    soft -- that single line is the reason this project is not PyQt6.
//  - laggy zoom : pan/zoom are just uniform (scale/translate) changes on one
//    textured quad; nothing re-rasterizes on the CPU.
//  - banding    : the texture holds float32 (R32F, or RGB32F for an rgb view);
//    colour mapping, sRGB encoding and dithering happen in the shader from
//    full precision.
//
// Differences from the reference: no ROI machinery, channel extraction from a
// planar (C, H, W) -- or, still supported, interleaved (H, W, C) -- array, and
// u_vmin/u_vmax/u_colormap in place of u_bandLo/u_bandHi/u_bright. This
// shader's u_mode is its own thing (scalar vs rgb), unrelated to the
// reference's u_mode.
//
// この設計が「なぜこの形か」は、うっかり元に戻しやすいので日本語でも残す。
//  - HiDPIのぼやけ: キャンバスのバッキングストアは *デバイス* ピクセル
//    （clientSize × devicePixelRatio）で確保し、画像はGPUのクアッドとして
//    描く。resize() の devicePixelRatio を外すと、Windowsの150%スケーリングで
//    画像が甘くなる。この1行こそ本プロジェクトがPyQt6でない理由である。
//  - ズームの重さ: パン/ズームは1枚のテクスチャ付きクアッドに対する uniform
//    （scale/translate）の変更にすぎず、CPU側で再ラスタライズしない。
//  - バンディング: テクスチャはfloat32（R32F）。カラーマップ・sRGB符号化・
//    ディザはすべてシェーダ内でフル精度から行う。

import { VERT_SRC, FRAG_SRC } from "./shaders.js";

// The single source of truth for the NaN mask colour (linear 0..1 RGB, as the
// shader's u_nanColor wants it). It had three copies -- this default, main.js's
// `display`, and style.css's `--nan` -- and they had already drifted: the
// legend swatch was one 8-bit LSB away from the pixels it was explaining.
// main.js pushes nanColorCss() into --nan at startup so the swatch is derived
// from these exact numbers instead of restating them.
// NaNマスク色の唯一の情報源（シェーダのu_nanColorに合わせた線形0..1のRGB）。
// 以前は3か所に写しがあり、実際に1LSBずれていた。CSSの --nan は起動時に
// main.js が nanColorCss() から注入する。
export const NAN_COLOR = Object.freeze([0.85, 0.1, 0.75]);

// Same triple as a CSS colour, quantised the way the framebuffer quantises it.
// A driver may still differ by one LSB on a component that lands exactly on
// .5 (0.1 * 255 = 25.5), which is the driver's tie-breaking, not a second
// source of truth.
// NAN_COLOR を8bitへ量子化してCSS色にする。ちょうど.5になる成分はドライバの
// 丸め方次第で1LSBずれ得るが、それは情報源が二つあることを意味しない。
export function nanColorCss(color = NAN_COLOR) {
  const to8 = (v) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
  return `rgb(${to8(color[0])}, ${to8(color[1])}, ${to8(color[2])})`;
}

// Planar (C, H, W) -> interleaved (H, W, 3), for the rgb texture.
//
// This copy is unavoidable and it is the only one on this path: WebGL textures
// are interleaved, the wire format is planar, so somebody has to transpose.
// Three channels make it a quarter of the twelve-channel de-interleave that
// moving the wire format to planar removed, and it happens once per upload --
// it must never migrate into draw(), which the display controls call on every
// slider tick. An already-interleaved body with exactly three channels is the
// texture's own order and is uploaded with no copy at all.
//
// planar (C, H, W) を (H, W, 3) へ詰め替える。WebGLのテクスチャはインタリーブ
// しか受け付けないので、この1回だけは避けられない。3チャンネルなので12chの
// デインターリーブの1/4で済む。setImage の中だけで行い、draw() へ持ち込まない
// こと（表示コントロールは毎回 draw() を呼ぶ）。既にインタリーブで3チャンネル
// なら並びはそのままなので複製しない。
function packRgb(data, width, height, channels, layout) {
  const n = width * height;
  if (layout !== "planar") {
    if (channels === 3) return data;
    const out = new Float32Array(n * 3);
    for (let i = 0, k = 0; i < n; i++, k += channels) {
      out[3 * i] = data[k];
      out[3 * i + 1] = data[k + 1];
      out[3 * i + 2] = data[k + 2];
    }
    return out;
  }
  const out = new Float32Array(n * 3);
  for (let c = 0; c < 3; c++) {
    const base = c * n;
    for (let i = 0, o = c; i < n; i++, o += 3) out[o] = data[base + i];
  }
  return out;
}

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("shader compile: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

export class GLView {
  constructor(canvas) {
    // preserveDrawingBuffer lets us read the canvas back for PNG export.
    const gl = canvas.getContext("webgl2", {
      antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error("WebGL2 not available (RDP/VM/old GPU may fall back to software).");
    this.canvas = canvas;
    this.gl = gl;

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT_SRC));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG_SRC));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("program link: " + gl.getProgramInfoLog(prog));
    }
    this.prog = prog;
    this.u = {};
    for (const n of ["u_imageSize", "u_scale", "u_translate", "u_resolution", "u_tex",
      "u_vmin", "u_vmax", "u_mode", "u_colormap", "u_exposure", "u_gamma", "u_dither",
      "u_nanColor"]) {
      this.u[n] = gl.getUniformLocation(prog, n);
    }

    // Unit quad in image UV [0,1]^2 as a triangle strip.
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "a_uv");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.vao = vao;

    this.tex = gl.createTexture();
    this.image = null;                 // { width, height, channel }
    this.view = { scale: 1, tx: 0, ty: 0 };
    this.params = {
      exposure: 0, gamma: 1, vmin: -1, vmax: 1, colormap: 0,
      dither: 1, nanColor: NAN_COLOR,
    };

    // Interaction callbacks (set by the app).
    this.onPixel = null;      // (col, row)
    this.onHover = null;      // (col, row) | null
    this.onViewChange = null; // ()

    this._installInput();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas);
    this._watchDpr();
    this.resize();
  }

  // A devicePixelRatio change does not resize the element, so ResizeObserver
  // stays silent and the backing store would keep its old device-pixel size --
  // exactly the fixed-DPI raster this whole design exists to avoid. Dragging
  // the window to the 150%-scaled projector display is enough to hit it, so
  // watch the resolution media query and re-arm it at the new ratio.
  // DPR が変わっても要素サイズは変わらず ResizeObserver は鳴らない。
  // 解像度メディアクエリを監視し、変化のたびに張り直す。
  _watchDpr() {
    this._onDpr = () => { this._armDprQuery(); this.resize(); };
    this._armDprQuery();
    // Backstop: a window resize is the other event that accompanies a monitor
    // or scaling change, and it costs one no-op draw when the ratio is stable.
    // 画面移動・スケーリング変更に伴う resize も拾う保険。
    window.addEventListener("resize", this._onDpr);
  }

  _armDprQuery() {
    const dpr = window.devicePixelRatio || 1;
    if (this._dprMq && this._dprArmedAt === dpr) return;
    if (this._dprMq) this._dprMq.removeEventListener("change", this._onDpr);
    this._dprArmedAt = dpr;
    this._dprMq = window.matchMedia(`(resolution: ${dpr}dppx)`);
    this._dprMq.addEventListener("change", this._onDpr);
  }

  // img: { data, width, height, channels, channel, layout, mode }
  //
  // `data` is the whole multi-channel array as fetched, in either layout. The
  // caller keeps it alive on the CPU so pixel probing never needs a server
  // round-trip.
  //
  // `mode` decides what comes out of it. "scalar" (the default) slices one
  // plane and uploads it as R32F, one tile per channel. "rgb" takes three
  // channels as a single colour and uploads them as one RGB32F texture; the
  // shader is told which via u_mode. RGB32F is texture-only in WebGL2 (not
  // colour-renderable), which is all this needs -- and it is never filtered,
  // see the NEAREST note below.
  //
  // data は取得した多チャンネル配列そのもの（layout はどちらでもよい）。
  // mode が "scalar" なら1面を切り出して R32F、"rgb" なら3チャンネルを1枚の
  // RGB32F に詰めて載せる。どちらかはシェーダへ u_mode で伝える。
  setImage(img) {
    const gl = this.gl;
    const width = img.width, height = img.height;
    const channels = img.channels ?? 1;
    const channel = img.channel ?? 0;
    const mode = img.mode === "rgb" ? "rgb" : "scalar";
    const previous = this.image;

    let plane;
    if (mode === "rgb") {
      // One transpose, here, never in draw(). See packRgb.
      plane = packRgb(img.data, width, height, channels, img.layout);
    } else if (channels === 1) {
      plane = img.data;
    } else if (img.layout === "planar") {
      // Planar (C, H, W): the channel is already contiguous. subarray shares
      // the buffer, so this costs nothing.
      // planar (C, H, W) ならチャンネルは既に連続しており、subarray は
      // バッファを共有するのでコストがかからない。
      const n = width * height;
      plane = img.data.subarray(channel * n, (channel + 1) * n);
    } else {
      // Interleaved (H, W, C): one strided walk per channel. This is the
      // fallback for a response that omits X-Layout -- api.js defaults to
      // "interleaved" rather than assuming the newer format, so an endpoint
      // that has not been converted still renders instead of drawing garbage.
      // Note the `channels === 1` case never reaches either branch: a
      // single-channel body is already the plane and is uploaded without a copy.
      // インタリーブ (H, W, C) はチャンネルごとのストライド走査になる。
      // X-Layout を返さない応答のための経路であり、api.js は既定を
      // "interleaved" にして新形式を勝手に仮定しない。なお `channels === 1`
      // はどちらの分岐にも入らず、複製なしでそのまま載る。
      const n = width * height;
      plane = new Float32Array(n);
      for (let i = 0, k = channel; i < n; i++, k += channels) plane[i] = img.data[k];
    }

    this.image = { width, height, channel, mode };
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    // NEAREST, in both modes. LINEAR would interpolate across the mask
    // boundary and smear a NaN into its finite neighbours -- the mask exists
    // precisely so those pixels are never given a plausible value.
    // 両モードとも NEAREST。LINEAR にすると NaN が隣接画素へにじみ、
    // マスク境界が汚染される。
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (mode === "rgb") {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, width, height, 0, gl.RGB, gl.FLOAT, plane);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, plane);
    }

    // Only reframe when there is nothing sensible to keep. Re-fetching the
    // same matrix with a different server-side parameter replaces the pixels
    // at identical dimensions, and throwing the user's pan and zoom away for
    // that is a data loss they did not ask for. Different dimensions (or the
    // first image) have no meaningful view to preserve, so fit then.
    // 同じ寸法で画素だけ差し替わる場合はパン/ズームを保つ。寸法が変わった
    // ときと初回だけ fit() する。
    if (!previous || previous.width !== width || previous.height !== height) {
      this.fit();
    } else {
      this.draw();
    }
  }

  setParams(p) { Object.assign(this.params, p); this.draw(); }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const old = this.dpr;
    this.dpr = dpr;

    // view.scale/tx/ty are device pixels measured from the canvas origin, so a
    // ratio change invalidates them: at 1.0 -> 1.5 the backing store grows by
    // half while the transform does not, leaving the image two-thirds size in
    // the top-left corner until someone hits 表示リセット.
    //
    // Rescaling by newDpr/oldDpr is a similarity about that same origin, and the
    // element's CSS size has not changed, so what the user sees is unmoved and
    // unresized -- their pan and zoom survive being dragged to another monitor.
    // fit() would also clear the artefact, but by discarding the view, and it
    // fires onViewChange from all 12 tiles at once; this needs no propagation
    // because every tile derives the same ratio from the same global.
    // DPR比で相似変換すれば見た目は不変でパン/ズームも保たれる。fit() だと
    // 視点を捨ててしまい、12枚が同時に onViewChange を撃つ。
    if (old && old !== dpr) {
      const k = dpr / old;
      this.view.scale *= k;
      this.view.tx *= k;
      this.view.ty *= k;
    }

    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    this.draw();
  }

  fit() {
    if (!this.image) return;
    const W = this.canvas.width, H = this.canvas.height;
    const s = Math.min(W / this.image.width, H / this.image.height);
    this.view.scale = s;
    this.view.tx = (W - this.image.width * s) / 2;
    this.view.ty = (H - this.image.height * s) / 2;
    this.draw();
    if (this.onViewChange) this.onViewChange();
  }

  // Map a CSS-pixel event position to integer image (col, row), or null.
  screenToImage(cssX, cssY) {
    if (!this.image) return null;
    const dpr = this.dpr;
    const dx = cssX * dpr, dy = cssY * dpr;
    const col = Math.floor((dx - this.view.tx) / this.view.scale);
    const row = Math.floor((dy - this.view.ty) / this.view.scale);
    if (col < 0 || row < 0 || col >= this.image.width || row >= this.image.height) return null;
    return { col, row };
  }

  // Map image (col, row) to CSS pixels (for positioning DOM overlays).
  imageToScreen(col, row) {
    const dpr = this.dpr;
    return {
      x: (col * this.view.scale + this.view.tx) / dpr,
      y: (row * this.view.scale + this.view.ty) / dpr,
    };
  }

  draw() {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.image) return;
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this.u.u_tex, 0);
    gl.uniform2f(this.u.u_imageSize, this.image.width, this.image.height);
    gl.uniform1f(this.u.u_scale, this.view.scale);
    gl.uniform2f(this.u.u_translate, this.view.tx, this.view.ty);
    gl.uniform2f(this.u.u_resolution, this.canvas.width, this.canvas.height);
    const p = this.params;
    gl.uniform1f(this.u.u_vmin, p.vmin);
    gl.uniform1f(this.u.u_vmax, p.vmax);
    // Not a display parameter: the mode follows the texture that was uploaded,
    // so it comes from this.image rather than from params. Putting it in
    // params would let a control set it and read an R32F texture as RGB.
    // モードはアップロード済みテクスチャに従うので params ではなく image から
    // 取る。params に置くとコントロールから切り替えられてしまう。
    gl.uniform1i(this.u.u_mode, this.image.mode === "rgb" ? 1 : 0);
    gl.uniform1i(this.u.u_colormap, p.colormap | 0);
    gl.uniform1f(this.u.u_exposure, p.exposure);
    gl.uniform1f(this.u.u_gamma, p.gamma);
    gl.uniform1f(this.u.u_dither, p.dither);
    gl.uniform3fv(this.u.u_nanColor, p.nanColor);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // Block until the driver has consumed everything queued. Only used for
  // honest timing measurements -- never on the interactive path.
  finish() { this.gl.finish(); }

  // Release the WebGL context. Browsers cap the number of live contexts
  // (~16 in Chrome) and 12 tiles sit close to that ceiling, so rebuilding the
  // grid must hand contexts back rather than orphan them.
  // ブラウザのコンテキスト数上限（Chromeで約16）に対し12枚は余裕がない。
  // グリッドを組み直すときは必ず解放する。
  dispose() {
    this._ro.disconnect();
    if (this._dprMq) this._dprMq.removeEventListener("change", this._onDpr);
    window.removeEventListener("resize", this._onDpr);
    const ext = this.gl.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
    this.image = null;
  }

  _installInput() {
    const c = this.canvas;
    let dragging = false, moved = false;
    let lastX = 0, lastY = 0;

    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      moved = false; lastX = e.clientX; lastY = e.clientY;
      dragging = true;
    });

    c.addEventListener("pointermove", (e) => {
      const lx = this._localX(e), ly = this._localY(e);
      if (dragging) {
        const dpr = this.dpr;
        this.view.tx += (e.clientX - lastX) * dpr;
        this.view.ty += (e.clientY - lastY) * dpr;
        lastX = e.clientX; lastY = e.clientY; moved = true;
        this.draw();
        if (this.onViewChange) this.onViewChange();
      } else if (this.onHover) {
        this.onHover(this.screenToImage(lx, ly));
      }
    });

    c.addEventListener("pointerup", (e) => {
      const lx = this._localX(e), ly = this._localY(e);
      if (dragging && !moved && this.onPixel) {
        const hit = this.screenToImage(lx, ly);
        if (hit) this.onPixel(hit.col, hit.row);
      }
      dragging = false;
    });

    c.addEventListener("pointerleave", () => {
      if (!dragging && this.onHover) this.onHover(null);
    });

    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const dpr = this.dpr;
      const cx = this._localX(e) * dpr, cy = this._localY(e) * dpr;
      const imgX = (cx - this.view.tx) / this.view.scale;
      const imgY = (cy - this.view.ty) / this.view.scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.view.scale *= factor;
      this.view.tx = cx - imgX * this.view.scale;
      this.view.ty = cy - imgY * this.view.scale;
      this.draw();
      if (this.onViewChange) this.onViewChange();
    }, { passive: false });
  }

  _localX(e) { return e.clientX - this.canvas.getBoundingClientRect().left; }
  _localY(e) { return e.clientY - this.canvas.getBoundingClientRect().top; }
}
