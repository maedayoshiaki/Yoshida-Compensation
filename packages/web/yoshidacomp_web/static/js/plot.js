// The measured inverse LUT, drawn, and the analytic gamma it replaces.
//
// One export, `drawLutCurve`, and no state of its own: it is handed a canvas
// and the body of `GET /api/lut/{id}/curve` and draws that, so the same
// function redraws on a resize, on a DPR change and on the overlay toggle
// without anything having to be remembered between calls.
//
// 逆LUTの実測カーブと、それが置き換えている解析ガンマを描く。関数は1つだけで
// 状態を持たない。canvas と `GET /api/lut/{id}/curve` の本体を渡されて描くので、
// リサイズ・DPR変更・重ね描きの切替はすべて同じ呼び出しで済む。
//
// ---------------------------------------------------------------------------
// **The vertical axis comes from the data, and must not be pinned to 1.0.**
//
// A measured inverse LUT does not end at 1.0. On
// data/full_recalibration_20260730_iso400/inverse_lut/inverse_lut.npz the three
// channels stop at 0.813 / 0.876 / 0.751, and that gap -- "the projector never
// gets there" -- is one of the few things this plot exists to show. An axis
// fixed at [0, 1] draws exactly the same picture for a LUT that reaches 1.0 and
// for one that stops at 0.69: three curves, in the right colours, in the right
// shapes, ending somewhere below the top. Nothing on screen would say which.
//
// So `vMax` is the largest value in the LUT itself. The three channels share
// it, deliberately: giving each its own axis would draw a channel that reaches
// 0.75 and one that reaches 0.88 at the same height, which is precisely the
// difference the plot is being read for.
//
// 縦軸はデータから決める。実測LUTの終端は 1.0 ではなく（上記ファイルで
// 0.813 / 0.876 / 0.751）、その差こそがこの図を見る理由の1つである。[0, 1]
// に固定すると、1.0 まで届くLUTと 0.69 で止まるLUTが同じ絵になり、画面上に
// 区別が無くなる。3チャンネルは1つの軸を共有する（別々にすると 0.75 の
// チャンネルと 0.88 のチャンネルが同じ高さに描かれ、読み取りたい差が消える）。
// ---------------------------------------------------------------------------
//
// The frame and the gridlines are rectangles (`strokeRect` / `fillRect`), not
// stroked paths. That is a testability decision as much as a drawing one: it
// makes every `beginPath … stroke` in this module exactly one data curve, so
// tests/js/frontend_harness.mjs can count the curves and read back each one's
// coordinates without having to tell a polyline from a grid line first.
// 枠と目盛線は矩形で描き、パスのストロークは使わない。これにより
// `beginPath … stroke` の1回が必ずデータのカーブ1本に対応し、ハーネスは
// 目盛線と区別する前処理なしに本数と座標を読める。

// One colour per channel, plus the overlay's. Chosen to stay legible on both
// the light and the dark palette in style.css (the canvas sits on --panel-2 in
// both), which is why they are literals here rather than read from the CSS
// variables: --ink and --muted flip between themes, these must not.
// チャンネルごとの色と重ね描きの色。style.css の明暗どちらの配色でも読める
// 値にしてある（CSS変数から引かないのは、--ink などが配色で反転するため）。
const CHANNEL_COLORS = ["#e5484d", "#2f9e5f", "#3e63dd"];
const GAMMA_COLOR = "#8b95a5";
const AXIS_COLOR = "#8b95a5";

// Margins around the frame, in CSS pixels: room for the value labels on the
// left and the t labels underneath.
const PAD_L = 34, PAD_R = 8, PAD_T = 8, PAD_B = 16;

// The data area is inset from the frame so a curve sitting *at* vMax is drawn
// whole rather than half-clipped by the frame's own line. Without it the
// tallest channel -- which by construction touches the top -- looks cut off,
// i.e. looks exactly like the clipping this plot exists to rule out.
// データ領域は枠から少し内側に取る。vMax にある曲線が枠線で半分切れて見えると、
// 「上で切り詰められている」という、この図が否定したい見え方そのものになる。
const INSET = 3;

function extent(rows) {
  let lo = Infinity, hi = -Infinity;
  for (const row of rows) {
    for (const v of row) {
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return { lo, hi };
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100 || a < 0.001) return v.toExponential(1);
  return v.toFixed(3);
}

// Draw one polyline. The only place a path is stroked in this module.
function polyline(ctx, xs, ys, color, dashed) {
  ctx.save();
  ctx.lineWidth = dashed ? 1.4 : 1.8;
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.strokeStyle = color;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    if (i === 0) ctx.moveTo(xs[i], ys[i]);
    else ctx.lineTo(xs[i], ys[i]);
  }
  ctx.stroke();
  ctx.restore();
}

/**
 * Draw a measured inverse LUT, and optionally the analytic gamma beside it.
 *
 * @param {HTMLCanvasElement} canvas  sized by CSS; its backing store is set here
 * @param {{t_grid: number[], lut: number[][]}} curve  the /curve response body
 * @param {{gamma?: number|null, dpr?: number}} [options]
 *        `gamma` draws `t ** (1 / gamma)` as a fourth, dashed line -- the
 *        analytic linearizer this measured one replaces (`LINEARIZERS["gamma"]`
 *        applies exactly that exponent). Omitted, null or non-positive draws no
 *        overlay. `dpr` overrides `window.devicePixelRatio`, for tests.
 * @returns {{plot: {x: number, y: number, w: number, h: number},
 *            tMin: number, tMax: number, vMin: number, vMax: number,
 *            gamma: number|null, curves: number}}
 *          the mapping that was drawn: `plot` is the data rectangle in CSS
 *          pixels, and a value `v` at `t` was drawn at
 *          `x = plot.x + plot.w * (t - tMin) / (tMax - tMin)`,
 *          `y = plot.y + plot.h * (1 - (v - vMin) / (vMax - vMin))`.
 */
export function drawLutCurve(canvas, curve, options = {}) {
  const rows = curve && curve.lut ? curve.lut : [];
  const grid = curve && curve.t_grid ? curve.t_grid : [];
  if (!rows.length || !grid.length) {
    throw new Error("drawLutCurve: カーブに t_grid か lut がありません。");
  }

  // The backing store is CSS size x DPR, and the context is scaled by DPR so
  // everything below is written in CSS pixels. Without this the whole plot is
  // drawn into a quarter of the canvas on a 200% display and then stretched --
  // the same reason glview.js's resize() carries a devicePixelRatio term.
  // バッキングストアは CSSサイズ×DPR で確保し、コンテキストを DPR で拡大する
  // ので以下は CSS ピクセルで書ける。これが無いと 200% 表示では図が canvas の
  // 1/4 に描かれて引き伸ばされる（glview.js の resize() と同じ理由）。
  const dpr = Number(options.dpr ?? (typeof window !== "undefined" ? window.devicePixelRatio : 1)) || 1;
  const cssW = Math.max(1, canvas.clientWidth || 1);
  const cssH = Math.max(1, canvas.clientHeight || 1);
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const frame = {
    x: PAD_L, y: PAD_T,
    w: Math.max(1, cssW - PAD_L - PAD_R),
    h: Math.max(1, cssH - PAD_T - PAD_B),
  };
  const plot = {
    x: frame.x, y: frame.y + INSET,
    w: frame.w, h: Math.max(1, frame.h - 2 * INSET),
  };

  const tMin = Number(grid[0]);
  const tMaxRaw = Number(grid[grid.length - 1]);
  const tMax = tMaxRaw > tMin ? tMaxRaw : tMin + 1;

  // Read off the LUT and nothing else. The overlay is deliberately *not* in
  // this extent: `t ** (1 / gamma)` reaches 1.0 by construction, so including
  // it would put the axis back at [0, 1] for every LUT and undo the whole
  // point above. Where the analytic curve runs past the measured range it
  // leaves the top of the plot, which is the honest picture of that difference.
  // 範囲はLUTだけから取る。重ね描きは必ず 1.0 に達するので、これを含めると
  // どのLUTでも軸が [0, 1] に戻り、上に書いた理由が丸ごと無効になる。解析
  // カーブが実測の範囲を超える所では図の上へ抜けるが、それがその差の正直な絵
  // である。
  const { lo, hi } = extent(rows);
  // `Math.min(0, lo)` and not `0`: the axis always includes 0, because 0 is
  // "no drive" and a curve's distance from it is what is being read, but a LUT
  // holding negative values must not have them drawn off the bottom of the
  // frame with nothing to say so. `lut.py::_validate` refuses a non-finite LUT
  // and a mis-shaped one; it does not require [0, 1], so a negative sample
  // loads.
  //
  // No test holds this to anything, and that is recorded rather than fixed:
  // every fixture LUT starts at exactly 0, so the harness's
  // `closeEnough(plot.vMin, Math.min(0, fixtureMin))` computes 0 either way and
  // `const vMin = 0;` passes it (measured, task-4 final review Minor 2). The
  // *upper* bound is a real guard -- `lut_plot_axis_pinned_to_one` fails its
  // named check. Discriminating the lower one needs a fixture LUT with a
  // negative sample, which is a physically meaningless file this loader
  // nonetheless accepts.
  // `0` ではなく `Math.min(0, lo)` にしてある理由と、それを検証しているものが
  // 無いことの両方をここに書いておく。フィクスチャの最小値が 0 ちょうどなので
  // 下端のチェックは何も述べていない（上端は本物のガード）。
  const vMin = Math.min(0, Number.isFinite(lo) ? lo : 0);
  const vMaxRaw = Number.isFinite(hi) ? hi : 1;
  const vMax = vMaxRaw > vMin ? vMaxRaw : vMin + 1;

  const mapX = (t) => plot.x + (plot.w * (t - tMin)) / (tMax - tMin);
  const mapY = (v) => plot.y + plot.h * (1 - (v - vMin) / (vMax - vMin));

  // --- frame, gridlines and labels: rectangles and text only (see the header)
  ctx.strokeStyle = AXIS_COLOR;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.55;
  ctx.strokeRect(frame.x + 0.5, frame.y + 0.5, frame.w - 1, frame.h - 1);
  ctx.fillStyle = AXIS_COLOR;
  ctx.globalAlpha = 0.25;
  for (const frac of [0.25, 0.5, 0.75]) {
    ctx.fillRect(Math.round(frame.x + frame.w * frac), frame.y, 1, frame.h);
    ctx.fillRect(frame.x, Math.round(frame.y + frame.h * frac), frame.w, 1);
  }
  ctx.globalAlpha = 1;
  ctx.font = "9px ui-monospace, Consolas, monospace";
  ctx.fillStyle = AXIS_COLOR;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  // The top label is the measured maximum, printed rather than assumed: it is
  // the number that says the axis is the data's and not [0, 1].
  // 上端のラベルは実測の最大値。軸がデータ由来であることを述べる数値である。
  ctx.fillText(fmtTick(vMax), frame.x - 4, plot.y);
  ctx.fillText(fmtTick(vMin), frame.x - 4, plot.y + plot.h);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(fmtTick(tMin), frame.x, frame.y + frame.h + 2);
  ctx.textAlign = "right";
  ctx.fillText(`t=${fmtTick(tMax)}`, frame.x + frame.w, frame.y + frame.h + 2);

  // --- the curves
  ctx.save();
  ctx.beginPath();
  ctx.rect(frame.x, frame.y, frame.w, frame.h);
  ctx.clip();

  const xs = grid.map((t) => mapX(Number(t)));
  for (let c = 0; c < rows.length; c++) {
    polyline(ctx, xs, rows[c].map((v) => mapY(Number(v))), CHANNEL_COLORS[c % CHANNEL_COLORS.length], false);
  }

  const gamma = Number(options.gamma);
  const overlay = Number.isFinite(gamma) && gamma > 0 ? gamma : null;
  if (overlay !== null) {
    // The analytic inverse gamma, on the same axis: `LINEARIZERS["gamma"]`
    // raises the linear value to `1 / projector.gamma`, so this is the curve
    // the measured LUT is standing in for, point for point.
    // 同じ軸に解析逆ガンマを重ねる。`LINEARIZERS["gamma"]` は 1/γ 乗するので、
    // これが実測LUTの置き換え対象そのものである。
    polyline(
      ctx,
      xs,
      grid.map((t) => mapY(Math.pow(Math.max(0, Number(t)), 1 / overlay))),
      GAMMA_COLOR,
      true,
    );
  }
  ctx.restore();

  return { plot, tMin, tMax, vMin, vMax, gamma: overlay, curves: rows.length };
}
