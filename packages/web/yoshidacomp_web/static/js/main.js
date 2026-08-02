// Application wiring for the matrix viewer.
//
// Two shapes of view are drawn: a scalar one, one tile per channel (the 12
// coefficients), and an rgb one, three channels on a single colour tile (the
// projection surface). Everything that differs between them is one row of
// contract.js's KIND_TRAITS, read here rather than re-decided per site.
// 両者の違いは contract.js の KIND_TRAITS の1行にまとまっており、各所で
// 判断し直さずそこを読む。
//
// The load-bearing rule here is the separation between *display* and
// *recompute*. Exposure, gamma, colormap and dither only push uniforms into the
// GLViews; they must never touch the network. Only a server-side view
// parameter (`normalize` for the coefficient tiles, the projection command for
// the surface) re-issues the binary fetch -- and those are staged and submitted
// as a group by 適用, one render per gesture rather than one per control.
// 露出・γ・カラーマップ・ディザは uniform を更新するだけでサーバへ行かない。
// サーバ側ビューパラメータだけが再フェッチを起こし、それも 適用 でまとめて
// 送る（コントロールごとではなく操作ごとに1回のレンダ）。
//
// The fetched Float32Array stays on the CPU for the whole session, so the
// hovered pixel's *view* values are read locally. The body is planar (C, H, W)
// -- channel c of pixel (row, col) is data[c * width * height + row * width +
// col] -- with the older interleaved (H, W, C) still handled wherever
// `payload.layout` says so. The hovered pixel's *matrix* coefficients are not
// in that array at all under a view like `surface`, and are fetched one pixel
// at a time instead; see the readout section.
// 取得した Float32Array は保持し続け、ホバー画素の「ビューの値」はローカルで
// 読む。本体は planar (C, H, W)。従来のインタリーブも layout を見て扱う。
// ホバー画素の「行列の係数」は `surface` のようなビューではこの配列に無いので、
// 1画素ずつ取りに行く（読み取りの節を参照）。

import {
  loadMatrix, listViews, fetchView, probeMatrix,
  loadTarget, loadLut, fetchLutCurve,
} from "./api.js";
import { GLView, NAN_COLOR, nanColorCss } from "./glview.js";
import { LAYOUT_PLANAR, requireSupportedKind, traitsFor } from "./contract.js";
// The class palette is imported from the shader module, not restated here.
// The legend's swatches are `classColorCss(i)` of the same array the fragment
// shader's classColor() is generated from, so a palette edit moves both or
// neither -- see the note on CLASS_COLORS in shaders.js.
// パレットはシェーダのモジュールから読む（ここに書き写さない）。
import { CLASS_COLORS, classColorCss } from "./shaders.js";
import { drawLutCurve } from "./plot.js";

const $ = (id) => document.getElementById(id);

// Startup fallback only -- the real axis labels always arrive from the server
// (axesFromLabels over the X-Labels-B64 header). These mirror
// packages/web/yoshidacomp_web/views/coef_tiles.py's _INPUT_LABELS and
// _OUTPUT_LABELS so the empty grid drawn before the first load has the right
// shape; that Python file is the source of truth.
// 起動時の仮置きのみ。実際の軸ラベルは常にサーバから来る。値は
// views/coef_tiles.py の _INPUT_LABELS / _OUTPUT_LABELS の写しで、
// 正はあちら側。
const DEFAULT_ROWS = ["in_R", "in_G", "in_B", "bias"];
const DEFAULT_COLS = ["out_R", "out_G", "out_B"];

// The corner cell of a coefficient grid, where the row and column axes meet.
// An rgb view has no such axes, so it gets an empty corner rather than a
// caption that describes a matrix it is not. It doubles as the marker for
// "this layout's axes came from a matrix", which is what layoutFor's
// keep-current-axes fallback is allowed to inherit from.
// 係数グリッドの左上、行と列の軸が交わるセル。rgbビューにはその軸が無いので
// 空にする。同時に「この配置の軸は行列由来である」という目印でもあり、
// layoutFor が軸を引き継いでよいのはこれが付いている場合だけである。
const MATRIX_CORNER = "in \\ out";

// rows x cols is the *tile* grid -- the shape of whatever view is on screen.
// The readout table is not derived from it; it is always the matrix's own
// 4x3, built once from DEFAULT_ROWS x DEFAULT_COLS. See buildProbeGrid.
// rows x cols はタイルの並び、すなわち表示中のビューの形である。読み取り表は
// ここから導かれず、常に行列自身の4x3である。
const DEFAULT_LAYOUT = {
  rows: DEFAULT_ROWS,
  cols: DEFAULT_COLS,
  corner: MATRIX_CORNER,
};

// The readout table's column widths, in CSS pixels.
//
// Fixed lengths, not fractions. The table is always the matrix's own 4x3 and a
// cell holds seven characters, so `minmax(0, 1fr)` stretched `-0.1234` across a
// 500px column on a 1600px window -- 12 numbers occupying the full width of the
// screen, with the display range and the transfer timings pushed into the
// sidebar's overflow because there was nowhere else for them to go. At these
// widths the whole table is about 240px and the rest of the readout fits beside
// it.
//
// Set here rather than in style.css because buildProbeGrid writes the template
// inline, and an inline style wins: a value in the sheet would look live and be
// dead. See the note on .probe-grid there.
//
// 読み取り表の列幅（CSSピクセル）。分数ではなく固定長にする。表は常に行列
// 自身の4x3で、1セルは7文字しかないため、`1fr` では1600pxの窓で1列500pxに
// なっていた——12個の数値が画面の全幅を占め、表示レンジと計測はサイドバーの
// 溢れた先へ追いやられていた。この幅なら表は約240pxで、残りは横に並ぶ。
// style.css ではなくここに書くのは、インライン指定が優先されるためである。
const PROBE_HEAD_PX = 38;
const PROBE_CELL_PX = 66;

// Gamma bounds for the typed input. The shader guards with max(u_gamma, 1e-6),
// so the floor here is about keeping the control usable rather than safe.
// 数値入力で受け付けるγの範囲。シェーダ側は max(u_gamma, 1e-6) で守られているので、
// この下限は安全性ではなく操作性のためのもの。
const GAMMA_MIN = 0.01;
const GAMMA_MAX = 10;

const state = {
  matrixId: null,
  // The matrix the *tiles* are currently drawn from, which is not the same
  // thing as `matrixId` and must not be confused with it.
  //
  // `matrixId` is what the newest load registered; it changes the moment
  // `POST /session/matrix` answers, seconds before the view transfer that will
  // put that matrix on screen has finished. The readout describes the picture,
  // so it has to ask about the picture: a hover during those seconds used to
  // be answered by the arriving matrix while the tiles still held the previous
  // one, and `inside_active_region` is per-matrix -- two matrices captured
  // either side of a config edit have different active regions, which is
  // routine here. The tile under the cursor would be grey (masked in the
  // matrix on screen) while the readout showed finite numbers and no marker
  // (inside the region of the matrix still arriving), or the reverse.
  //
  // Set beside `state.payload`, because the two are the same fact.
  //
  // タイルが今描いている行列。`matrixId`（最新の読み込みが登録したもの）とは
  // 別物である。後者は `POST /session/matrix` が返った時点で変わり、その行列が
  // 画面に出るまでには数秒の転送がある。読み取り欄は画面を説明するものなので、
  // 画面に出ている行列に問い合わせなければならない。以前は転送中のホバーが
  // 到着中の行列に答えられており、`inside_active_region` は行列ごとの性質
  // なので、設定を変えた前後で撮った2つの行列（このリポジトリでは日常）だと
  // 印と絵が転送のあいだずっと食い違う。`state.payload` と同じ場所で更新する。
  shownMatrixId: null,
  meta: null,
  specs: [],        // every ViewSpec from /api/views
  spec: null,       // the active one
  payload: null,    // { data, width, height, channels, layout, vmin, vmax, labels }
  // The KIND_TRAITS row `state.payload` was rendered through.
  //
  // Set beside the payload, because the two are the same fact: the traits are
  // read off the payload's own `kind`, and a copy that could describe a
  // different kind from the pixels on screen would put the class-index readout
  // over a coefficient tile. Kept rather than re-derived at each reader for
  // the same reason `shownMatrixId` is kept -- one place decides, everyone
  // else asks. The hover readout is the reader that needs it: it fires between
  // renders, when `state.spec` may already name a view whose transfer has not
  // landed.
  // 描かれているペイロードの特性行。ペイロードと同時に確定する（同じ事実で
  // あり、食い違うとクラス番号の読み取り欄が係数タイルの上に出る）。ホバーは
  // レンダの合間に起きるので `state.spec` からは導けない。
  traits: null,
  ranges: null,     // [{ vmin, vmax }] per *tile* (per channel except for rgb)
  layout: DEFAULT_LAYOUT,
  pinned: null,     // { col, row } while the readout is frozen
  sentParams: null, // the server-side params the newest request carried
  // Monotonic request id. Fetches are not cancelled, they are *ignored*: a
  // 95 MB transfer takes seconds, and a second 読み込み during the first one
  // used to leave the header describing matrix B while state.payload -- which
  // backs all 12 tiles and the hover readout -- still held matrix A. Two
  // clicks, no warning, coefficients read off the wrong matrix. Every async
  // entry point bumps this, captures it, and drops its result after any await
  // where it no longer matches.
  // 連続フェッチの順序保証。最新のリクエストIDと一致しない結果は捨てる。
  reqId: 0,
  // Hover readout. Fetches are coalesced rather than queued: a mousemove can
  // fire dozens of times a second and every one of them would otherwise be a
  // round trip. At most one request is in flight; the newest position waits
  // for it and fires once, so a fast drag costs two requests, not fifty.
  //
  // Deliberately a series of its own rather than `reqId`. That one guards the
  // view transfers, and sharing it would mark the in-flight view fetch stale
  // on every mousemove -- the tiles would simply stop arriving while the
  // pointer was over them, with nothing to say why.
  //
  // ホバーの点クエリは1つずつに合流させる。飛んでいる間の移動は最新の位置だけを
  // 残し、戻ってきたら1回だけ追加で投げる。`reqId` とは別系列にすること。
  // 流用するとホバーのたびに転送中のビュー取得がスタール扱いになり、ポインタを
  // 載せている間タイルが届かなくなる（しかも理由がどこにも出ない）。
  probeSeq: 0,        // monotonic; a response older than this is dropped
  probeInFlight: false,
  probeWanted: null,  // { col, row, seq } claimed while a fetch was in flight
  // The runtime ids the two sidebar loaders have obtained, and what they
  // loaded. Kept here rather than only in the `sp-*` boxes because those are
  // rebuilt from scratch on every view switch: without this, loading a target
  // and then looking at `surface` and back would silently empty the ターゲットID
  // box, and the `compensation` view would answer "needs a target image" about
  // one that is loaded and still in the store.
  // 2つの読み込み欄が得たIDと、その出所。`sp-*` はビュー切替のたびに作り直され
  // るので、ここに残さないと「読み込んだのに欄が空に戻る」ことになる。
  targetId: null,
  targetMeta: null,   // the POST /api/target meta: path, images, index, shape
  lutId: null,
  lutCurve: null,     // the GET /api/lut/{id}/curve body, kept for redraws
};

// What `drawLutCurve` last returned: the data rectangle and the mapping from
// LUT values into it.
//
// Not in `state`. Nothing in this file reads it -- the plot is redrawn from
// `state.lutCurve`, never from this -- and its only consumer is
// tests/js/frontend_harness.mjs, which needs the mapping to say *where* a
// value should have been drawn. `state` is what the viewer runs on, so a
// field there that no code path reads is an invitation to wire something to
// it; the automation handle `window.__viewer` is already the declared place
// for things that exist for a driver rather than for the page.
// `state` には置かない。このファイルのどこもこれを読まない（描き直しは
// `state.lutCurve` から行う）。読み手はハーネスだけであり、そういうものの
// 置き場は既に `window.__viewer` と決まっている。
let lastLutPlot = null;

// Which runtime-id parameter each loader fills. Read by buildServerParams so a
// view switch does not throw away an id that is still loaded.
// どの読み込み欄がどのパラメータを埋めるか。ビュー切替でIDを捨てないため
// buildServerParams が読む。
const LOADED_ID_FOR_PARAM = { target: "targetId", lut: "lutId" };

// Uniform-only display parameters. Changing any of these must not fetch.
const display = {
  exposure: 0, gamma: 1, colormap: 0, dither: 1, nanColor: NAN_COLOR,
};

const views = [];       // one GLView per channel
const captions = [];    // per-tile caption elements
const probeCells = [];  // the 12 readout cells, row-major over DEFAULT_ROWS x DEFAULT_COLS
let syncBusy = false;   // guards onViewChange against re-entry

// ---------------------------------------------------------------- utilities

function status(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  if (isError) console.error(msg);
}

function errorText(e) {
  return String(e && e.message ? e.message : e);
}

// Claim the newest request slot. Anything already in flight becomes stale.
function beginRequest() {
  state.reqId += 1;
  return state.reqId;
}

function isStale(req) {
  return req !== state.reqId;
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (!Number.isFinite(v)) return "NaN";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(3);
  return v.toFixed(4);
}

function nextFrame() {
  // Two frames: one to let the browser lay the grid out, one to be sure the
  // ResizeObserver has already sized every backing store.
  return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
}

// 99th percentile of |value| per channel, via a two-pass histogram.
//
// Sorting 2M floats twelve times would cost seconds; a 2048-bin histogram over
// two sequential passes costs a fraction of that and is accurate to ~0.05% of
// the range, which is far below what the eye reads off a colour map.
// ソートは高すぎるのでヒストグラムで分位点を取る。
//
// `layout` decides where channel `c`'s samples live: under planar (C, H, W)
// they are the contiguous run [c * stride, (c + 1) * stride); under interleaved
// (H, W, C) they sit every `channels` elements. Reading the wrong one does not
// throw -- it just attributes every value to the wrong channel and produces
// twelve plausible-looking but wrong display ranges.
// planar なら各チャンネルは連続、インタリーブなら channels 間隔。取り違えても
// 例外にはならず、12枚のレンジが静かに入れ替わるだけなので分岐を明示する。
function percentileAbsPerChannel(data, channels, pct, layout) {
  const BINS = 2048;
  const planar = layout === LAYOUT_PLANAR;
  // Pixels per channel (planar only). Math.floor rather than `| 0`: the
  // camera-space payload this is headed for is ~390M elements and `| 0` is a
  // 32-bit trap waiting for the next size increase.
  // planar のチャンネルあたり画素数。`| 0` は32bitで折り返すため使わない。
  const stride = Math.floor(data.length / channels);
  const maxAbs = new Float64Array(channels);
  const counts = new Float64Array(channels);
  const hist = new Int32Array(channels * BINS);

  // The two passes are written out per layout rather than behind a callback:
  // this walks ~25M floats twice on a 1080p payload and a per-sample closure
  // is not worth the risk on that path.
  // 1080pで約2500万要素を2周するため、走査は layout ごとに素直に展開する。
  if (planar) {
    for (let c = 0; c < channels; c++) {
      const base = c * stride;
      for (let i = 0; i < stride; i++) {
        const v = data[base + i];
        if (!Number.isFinite(v)) continue;
        const a = v < 0 ? -v : v;
        if (a > maxAbs[c]) maxAbs[c] = a;
        counts[c]++;
      }
    }
    for (let c = 0; c < channels; c++) {
      const m = maxAbs[c];
      if (m === 0) continue;
      const base = c * stride, hbase = c * BINS;
      for (let i = 0; i < stride; i++) {
        const v = data[base + i];
        if (!Number.isFinite(v)) continue;
        hist[hbase + ((((v < 0 ? -v : v) * (BINS - 1)) / m) | 0)]++;
      }
    }
  } else {
    for (let p = 0, n = data.length; p < n; p += channels) {
      for (let c = 0; c < channels; c++) {
        const v = data[p + c];
        if (!Number.isFinite(v)) continue;
        const a = v < 0 ? -v : v;
        if (a > maxAbs[c]) maxAbs[c] = a;
        counts[c]++;
      }
    }
    for (let p = 0, n = data.length; p < n; p += channels) {
      for (let c = 0; c < channels; c++) {
        const v = data[p + c];
        if (!Number.isFinite(v)) continue;
        const m = maxAbs[c];
        if (m === 0) continue;
        const bin = (((v < 0 ? -v : v) * (BINS - 1)) / m) | 0;
        hist[c * BINS + bin]++;
      }
    }
  }

  const out = new Array(channels);
  for (let c = 0; c < channels; c++) {
    const m = maxAbs[c];
    if (!counts[c] || m === 0) { out[c] = 1e-6; continue; }
    const target = (counts[c] * pct) / 100;
    let acc = 0;
    let val = m;
    for (let b = 0; b < BINS; b++) {
      acc += hist[c * BINS + b];
      if (acc >= target) { val = ((b + 1) * m) / (BINS - 1); break; }
    }
    out[c] = Math.max(Math.min(val, m), 1e-6);
  }
  return out;
}

// ------------------------------------------------------------- tile grid

// Split "in_R -> out_R" labels back into row/column axes so the grid follows
// whatever the view declares rather than a hard-coded 4x3.
function axesFromLabels(labels, channels) {
  const rows = [], cols = [];
  for (const label of labels) {
    const parts = String(label).split("->");
    if (parts.length !== 2) return null;
    const a = parts[0].trim(), b = parts[1].trim();
    if (!rows.includes(a)) rows.push(a);
    if (!cols.includes(b)) cols.push(b);
  }
  return rows.length * cols.length === channels ? { rows, cols } : null;
}

// A view whose labels are not "a -> b" still has to produce exactly `channels`
// tiles; anything else would make the grid rebuild on every reload and churn
// through the browser's WebGL context budget. One row of N is always exact.
// ラベルが行列形式でなくてもタイル数はチャンネル数と一致させる。
function fallbackAxes(labels, channels) {
  return {
    rows: [""],
    cols: Array.from({ length: channels }, (_, i) => labels[i] || String(i)),
  };
}

// What grid a payload needs: how many tiles, and what to write on them.
//
// A scalar view draws one tile per channel -- 12 coefficients are 12 unrelated
// numbers, and the grid's job is to name each one. An rgb view is the opposite:
// its three channels *are* one colour, and drawing them as three greyscale
// tiles would hide the only thing the view exists to show.
//
// So the tile grid branches on `kind` rather than being derived from the
// labels, and it has to. `axesFromLabels` cannot express "one tile, three
// channels": `surface`'s labels are ["R","G","B"], which have no "->", so it
// returns null and `fallbackAxes` produces exactly the 1x3 grid of greyscale
// tiles we do not want -- and it does so silently, drawing a viewer that looks
// like it works. `kind` is the thing that actually says how many pictures are
// in the payload, so that is what decides.
//
// タイル配置は kind で分岐させる。scalar はチャンネル数だけタイルを並べるが、
// rgb の3チャンネルは1枚の絵であり、3枚のグレースケールにしては意味が無い。
// ラベル ["R","G","B"] は `axesFromLabels` では解釈できず、フォールバックだと
// まさにその1行3列になってしまう（しかも黙って描けてしまう）。ペイロードに
// 絵が何枚あるかを知っているのは kind だけなので、それで決める。
function layoutFor(payload, current, traits) {
  // The channel count as well as the trait: a single-tile kind is defined by
  // how many channels that one tile consumes -- three for `rgb` (one colour
  // per pixel), one for `categorical` (one class index per pixel) -- and a
  // single-tile layout is only correct for a payload the matching upload path
  // can actually consume. Both numbers come from the same trait row as
  // `packed` and `texture`, so "one tile", "how it is packed" and "how it is
  // uploaded" cannot come apart. The active guards are elsewhere
  // (requireRgbHasThreeChannels here, routers/view.py's categorical clause on
  // the server); this is defence in depth.
  //
  // The caption is a trait too, and that is the interesting half. `labels` is
  // ["R","G","B"] for `surface` and the four *class* names for `gamut`, so the
  // label-derived fallback would head an rgb tile "R" and a categorical one
  // "範囲内" -- the name of class 0, written across a picture of all four
  // classes. It renders; it is simply a lie in 3-point type.
  //
  // 特性だけでなくチャンネル数も見る。1枚タイルの kind は「その1枚が何
  // チャンネルを食うか」で定義される（rgb は3、categorical は1）。見出しも
  // 特性から取る: ラベル由来にすると rgb は "R"、categorical はクラス0の名前
  // が4クラスの絵の上に載る（描けてしまうが嘘である）。
  if (traits.singleTile && payload.channels === traits.tileChannels) {
    return { rows: [""], cols: [traits.tileCaption], corner: "" };
  }
  // Keeping the current axes is only sound when they *are* axes -- that is,
  // when the layout on screen came from a matrix-shaped set of labels. The
  // corner marker is what records that.
  //
  // Without the gate, the counting condition alone is enough: an rgb layout is
  // 1 row x 1 col, so a hypothetical 1-channel scalar view arriving straight
  // after `surface` would satisfy `1 * 1 === 1` and inherit `cols = ["RGB"]`,
  // captioning its single greyscale tile as a colour. The fallback exists to
  // stop the grid churning through WebGL contexts on a re-fetch of the *same*
  // shape, not to hand one view another view's vocabulary.
  // 現在の軸を引き継いでよいのは、それが本当に軸であるとき——つまり画面上の
  // 配置が行列形式のラベルから来ているときだけで、それを記録しているのが
  // corner である。この条件が無いと枚数の一致だけで通ってしまい、rgb の
  // 1行1列の直後に来た1チャンネルの scalar ビューが cols=["RGB"] を継承して、
  // グレースケール1枚に「RGB」という見出しを付ける。このフォールバックは
  // 同じ形の再取得でグリッドを組み直さないためのものであって、あるビューに
  // 別のビューの語彙を渡すためのものではない。
  const inheritable = current.corner === MATRIX_CORNER
    && current.rows.length * current.cols.length === payload.channels;
  const axes = axesFromLabels(payload.labels, payload.channels)
    || (inheritable
      ? { rows: current.rows, cols: current.cols }
      : fallbackAxes(payload.labels, payload.channels));
  return { rows: axes.rows, cols: axes.cols, corner: MATRIX_CORNER };
}

function buildGrid(layout) {
  const rows = layout.rows, cols = layout.cols;
  for (const v of views) v.dispose();
  views.length = 0;
  captions.length = 0;

  const grid = $("tile-grid");
  grid.textContent = "";
  grid.style.gridTemplateColumns = `var(--rowhead) repeat(${cols.length}, minmax(0, 1fr))`;
  grid.style.gridTemplateRows = `auto repeat(${rows.length}, minmax(0, 1fr))`;

  const corner = document.createElement("div");
  corner.className = "corner mono";
  corner.textContent = layout.corner;
  grid.appendChild(corner);
  for (const c of cols) {
    const h = document.createElement("div");
    h.className = "col-head mono";
    h.textContent = c;
    grid.appendChild(h);
  }

  for (let ri = 0; ri < rows.length; ri++) {
    const rh = document.createElement("div");
    rh.className = "row-head mono";
    rh.textContent = rows[ri];
    grid.appendChild(rh);
    for (let ci = 0; ci < cols.length; ci++) {
      const cell = document.createElement("div");
      cell.className = "tile";
      const canvas = document.createElement("canvas");
      cell.appendChild(canvas);
      const cap = document.createElement("div");
      cap.className = "tile-cap mono";
      cell.appendChild(cap);
      grid.appendChild(cell);
      captions.push(cap);
      views.push(new GLView(canvas));
    }
  }
  linkViews();
  buildProbeGrid();
}

// Pan/zoom is shared by all tiles. The guard stops the propagation loop from
// re-entering: every synced view would otherwise fire its own onViewChange.
// 連動時の再帰をフラグで止める。
function linkViews() {
  views.forEach((v, i) => {
    v.onViewChange = () => {
      if (syncBusy) return;
      syncBusy = true;
      try {
        for (let j = 0; j < views.length; j++) {
          if (j === i) continue;
          const o = views[j];
          o.view.scale = v.view.scale;
          o.view.tx = v.view.tx;
          o.view.ty = v.view.ty;
          o.draw();
        }
      } finally {
        syncBusy = false;
      }
    };
    v.onHover = (hit) => { if (!state.pinned) showProbe(hit); };
    v.onPixel = (col, row) => {
      if (state.pinned && state.pinned.col === col && state.pinned.row === row) {
        state.pinned = null;
        $("probe-pin").textContent = "";
      } else {
        state.pinned = { col, row };
        $("probe-pin").textContent = "（固定中・もう一度クリックで解除）";
        showProbe({ col, row });
      }
    };
  });
}

// The readout table is the *matrix*, so it is always 4 inputs x 3 outputs --
// no argument, and nothing about the view on screen reaches it. It used to be
// built from the payload's own axes, which is why an rgb view produced a 1x3
// table of surface colours under headings that named coefficients.
// 読み取り表は行列そのものなので、常に入力4×出力3で組む。画面上のビューは
// 一切影響しない。以前はペイロードの軸から組んでいたため、rgbビューでは
// 係数の見出しの下に投影面のRGBが3つ並んでいた。
function buildProbeGrid() {
  const rows = DEFAULT_ROWS, cols = DEFAULT_COLS;
  const grid = $("probe-grid");
  grid.textContent = "";
  grid.style.gridTemplateColumns =
    `${PROBE_HEAD_PX}px repeat(${cols.length}, ${PROBE_CELL_PX}px)`;
  probeCells.length = 0;

  grid.appendChild(document.createElement("div"));
  for (const c of cols) {
    const h = document.createElement("div");
    h.className = "probe-head";
    h.textContent = c;
    grid.appendChild(h);
  }
  for (let ri = 0; ri < rows.length; ri++) {
    const rh = document.createElement("div");
    rh.className = "probe-head";
    rh.textContent = rows[ri];
    grid.appendChild(rh);
    for (let ci = 0; ci < cols.length; ci++) {
      const cell = document.createElement("div");
      cell.className = "probe-cell";
      cell.textContent = "—";
      grid.appendChild(cell);
      probeCells.push(cell);
    }
  }
}

// The readout is two panels with different sources, and keeping them apart is
// the whole of this section.
//
// `#view-values` is what the *displayed view* holds at that pixel. It comes
// out of `state.payload`, which is already in hand, so it is written
// synchronously and costs nothing.
//
// `#probe-grid` is the *matrix*'s own 4x3, which the client does not have:
// under `surface` there is no matrix on this side at all, and shipping a
// second 99 MB copy to get one would undo the heap the planar transfer bought.
// So it is fetched a pixel at a time. Until it was, the table showed the
// payload too -- hovering `surface` printed three projection-surface colours
// beneath headings that read `in_R -> out_R`.
//
// 読み取り欄は出所の違う2枚の板でできており、それを混ぜないことがこの節の要点。
// `#view-values` は表示中のビューの値で、手元の `state.payload` から同期的に
// 作れる。`#probe-grid` は行列そのものの4x3で、こちら側には無い（`surface` を
// 見ているときクライアントに行列は存在せず、2つ目の99MBを送れば計画B2で
// 半減させたヒープが元に戻る）。だから1画素ずつ取りに行く。それ以前は表も
// ペイロードを読んでいたため、`surface` をホバーすると `in_R -> out_R` という
// 見出しの下に投影面のRGBが並んでいた。

function clearMatrixCells() {
  for (const cell of probeCells) { cell.textContent = "—"; cell.className = "probe-cell"; }
}

// The position line, and the one warning the table cannot give by itself.
//
// `outside` says the pixel is excluded from the mask coef_tiles.py and
// surface.py paint -- and the numbers below are still printed, because they are
// what the file holds. That is the whole problem: the mask is applied at
// *render* time, so the tile is grey while the stored array keeps a finite,
// meaningless coefficient underneath. Measured on
// data/linear_pipeline_1080/matrices_linear_gamma.npy (active region
// [613, 201, 722, 722]), row 5 col 5 reads 13.440 / 1.197 / 0.017 -- set in the
// same type, the same colour and the same four decimals as a real one. Nothing
// distinguished them, and they are plausible enough to be written down.
//
// They are labelled rather than hidden: blanking them would make this
// indistinguishable from "nothing has been read yet" (the "—" clearReadout
// writes), and would also hide the one thing worth seeing here, which is that
// the file has values out there at all.
//
// Written from `want`/`hit` in full every time rather than appended to, so the
// text cannot end up describing a different pixel from the one that produced
// the numbers, and a second answer for the same pixel cannot double the suffix.
//
// 位置の表示と、表だけでは出せない唯一の警告。`outside` は
// coef_tiles.py / surface.py が塗るマスクの外側であることを指す。値は消さずに
// 出す——マスクは描画時に掛かるものなので、タイルが灰色でも配列には有限で
// 無意味な係数が残っているからである（実測: 上記の行列の row 5 col 5 で
// 13.440 / 1.197 / 0.017。本物と同じ字体・色・桁数で出ていた）。消さないのは、
// 消すと「まだ何も読んでいない」を意味する「—」と区別が付かなくなるうえ、
// 「ファイルには領域外にも値が入っている」という見るべき事実まで隠れるため。
// 毎回まるごと書き直すのは、数値を出した画素と表示が食い違わないようにする
// ためと、同じ画素への2度目の応答で接尾辞が重ならないようにするため。
//
// The wording says what is true of the numbers, which is not that they failed
// to be read. They are real values the file holds; what is wrong with them is
// that the mask covers those pixels, so nothing downstream uses them. 「無効」
// read as "invalid / the read failed" and contradicted the sentence above about
// why they are worth showing. 「マスク対象」 is also the vocabulary the sidebar
// legend already uses for the same pixels.
// 文言は「読めなかった」ではなく「その値がどういうものか」を述べる。値は
// ファイルに実在し、問題はその画素がマスクの対象で誰も使わないことにある。
// 「無効」は読み取り失敗と読め、値を出す理由と矛盾していた。「マスク対象」は
// サイドバーの凡例が同じ画素に使っている語でもある。
function setProbePos(col, row, outside) {
  const el = $("probe-pos");
  el.textContent = `col ${col}, row ${row}${outside ? " — 有効投影領域の外（マスク対象）" : ""}`;
  el.classList.toggle("outside", !!outside);
}

// Reset both panels, and retire whatever claim is in flight.
//
// Called on pointer-leave and after a render. Without the `probeSeq` bump a
// response already on its way would repaint the table a moment after it was
// cleared, over a `#probe-pos` that says "—".
// 両方の板を戻し、飛んでいる要求の権利を失効させる。ポインタが外れたときと
// 再描画のあとに呼ぶ。`probeSeq` を進めないと、消した直後に飛来した応答が
// 「—」と表示された位置の下に表を描き直す。
function clearReadout() {
  $("probe-pos").textContent = "—";
  $("probe-pos").classList.remove("outside");
  $("view-values").textContent = "";
  clearMatrixCells();
  state.probeSeq += 1;
  state.probeWanted = null;
}

// The position and the view's own values, both from data already here.
//
// The marker is carried over from whatever it already said, and that is the
// point: the twelve cells are *not* cleared here, so until the point query
// answers they still hold the previous pixel's coefficients -- and the marker
// disclaims those numbers, not the heading above them. Writing `false` here
// (which is what this did) cleared the marker while the out-of-region values
// it referred to were still on screen, in the same type, colour and precision
// as real ones: exactly the state setProbePos exists to make impossible, for
// one round trip, every time the pointer left a masked pixel.
//
// Guessing "inside" is the direction this marker exists to break, so the
// wrong-but-safe direction is to keep warning until the answer arrives. The
// alternative -- blanking the twelve cells on every mousemove -- would strobe
// the table during a drag and collide with the "—" that means "nothing read".
//
// 位置と、表示中のビューの値。印は「今出ている状態」をそのまま持ち越す。ここで
// 12個のセルは消さないので、応答が来るまで表には前の画素の係数が残っており、
// 印が打ち消しているのはその数値であって、上の見出しではない。以前のように
// `false` を書くと、領域外の値が本物と同じ字体・色・桁数で表に残ったまま印だけ
// が消える——マスクされた画素からポインタを動かすたびに、1往復のあいだ、
// setProbePos が無くそうとしている状態そのものが再現していた。「内側」と
// 推量する側こそこの印が破ろうとしている沈黙なので、誤るなら警告を残す側に
// 倒す。12個を毎回消す案は、ドラッグ中に表が点滅し、「まだ何も読んでいない」を
// 意味する「—」とも衝突する。
function showProbe(hit) {
  if (!hit) { clearReadout(); return; }
  setProbePos(hit.col, hit.row, $("probe-pos").classList.contains("outside"));
  showViewValues(hit);
  requestMatrixProbe(hit);
}

// One row: a caption and a number, in the classes the stylesheet colours.
// Shared by both readings below so "what a masked value looks like here" is
// decided once.
// 1行ぶんの生成。マスクされた値の見た目を決める場所を1つにするため共有する。
function appendViewValue(host, label, text, masked) {
  const item = document.createElement("div");
  item.className = "view-value";
  const labelEl = document.createElement("span");
  labelEl.className = "vv-label";
  labelEl.textContent = label;
  const value = document.createElement("span");
  value.className = masked ? "vv-num nan" : "vv-num";
  value.textContent = text;
  item.appendChild(labelEl);
  item.appendChild(value);
  host.appendChild(item);
}

function showViewValues(hit) {
  const host = $("view-values");
  host.textContent = "";
  const p = state.payload;
  if (!p) return;
  const traits = state.traits;

  // A categorical payload is read down a different axis, and this is the site
  // the brief for this task warns about by name.
  //
  // Every other kind has one label per channel, so the loop below prints
  // `labels[c]` beside channel c's value. Here there is one channel and four
  // labels, and they name the *classes*: `labels[0]` is the name of class 0,
  // so the per-channel reading captions every pixel in the frame "範囲内"
  // regardless of which class it actually is -- and it does so beside the
  // correct number, in the correct type, under the correct heading. The class
  // name is `labels[value]`, which is the only reading that changes as the
  // pointer moves.
  //
  // The index is rounded rather than truncated, matching the shader's
  // `int(floor(v + 0.5))`: the readout and the picture must agree about which
  // class a pixel is, and two different roundings is a way for them not to.
  //
  // 他の kind はチャンネル1つにラベル1つなので `labels[c]` を出すが、
  // categorical はチャンネル1・ラベル4で、しかもラベルはクラスの名前である。
  // チャンネルで引くと全画素がクラス0の名前を名乗る（数値も見出しも正しい
  // まま、名前だけが常に同じになる）。引くべきは `labels[値]` である。
  // 丸め方はシェーダと揃える（読み取りと絵が別のクラスを指さないため）。
  if (traits && traits.labelsAreClasses) {
    // No layout branch: `tileChannels` is 1, and a one-channel body is the
    // same run of pixels under both layouts (C=1 makes the planar stride and
    // the interleaved stride the same walk). Writing one anyway would be a
    // branch neither side of which could ever differ from the other.
    // 1チャンネルなので planar とインタリーブで並びは同一であり、分岐しない。
    const v = p.data[hit.row * p.width + hit.col];
    if (!Number.isFinite(v)) {
      appendViewValue(host, "クラス", "NaN", true);
      return;
    }
    const index = Math.round(v);
    const name = p.labels[index];
    appendViewValue(host, "クラス", `${index} ${name ?? "（未定義のクラス）"}`, !name);
    return;
  }

  // Planar (C, H, W) puts the channel index outermost; interleaved (H, W, C)
  // puts it innermost. Getting this backwards misreports every hovered value
  // while the tiles themselves still look right -- nothing would flag it.
  // planar は channel が最外、インタリーブは最内。取り違えても絵は正しいまま
  // 読み取り値だけが狂うので、どこも失敗しない。
  const planar = p.layout === LAYOUT_PLANAR;
  const stride = p.width * p.height;
  const offset = hit.row * p.width + hit.col;
  for (let c = 0; c < p.channels; c++) {
    const v = planar
      ? p.data[c * stride + offset]
      : p.data[offset * p.channels + c];
    appendViewValue(host, String(p.labels[c] ?? c), fmt(v), !Number.isFinite(v));
  }
}

// The class legend: one row per class, in class order, from `payload.labels`.
//
// **The labels come from the payload and the colours from CLASS_COLORS**, and
// neither is written down twice. The obvious build -- swatches in JavaScript,
// Japanese class names in index.html -- is two sources of truth for one
// mapping, and they drift in silence: views/gamut.py's LABELS are the wire
// contract (their *order* is the class index, per that file), so a server-side
// relabelling or a reordering leaves an HTML legend confidently explaining the
// previous meaning of the same four colours. Nothing fails, nothing looks
// wrong, and the picture now means something else.
//
// Hidden entirely for a kind whose labels are channels: `surface`'s three
// labels are not classes and a legend of them would be three colour swatches
// for a picture that has no classes in it.
//
// 凡例はラベルを `payload.labels` から、色を CLASS_COLORS から取る。どちらも
// 二重に書かない。色をJSに・和文ラベルをHTMLに書くと真実源が2つになり、
// サーバ側でラベルを変えたり並べ替えたりしても画面は前の意味を説明し続ける
// （何も失敗せず、絵の意味だけが変わる）。ラベルがチャンネル名である kind
// では丸ごと隠す。
function renderClassLegend(payload, traits) {
  const host = $("class-legend");
  host.textContent = "";
  host.hidden = !traits.labelsAreClasses;
  if (!traits.labelsAreClasses) return;
  for (let i = 0; i < payload.labels.length; i++) {
    const row = document.createElement("div");
    row.className = "legend class-row";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    // Derived from the shader's own palette, quantised the way the framebuffer
    // quantises it -- see classColorCss in shaders.js. Set inline rather than
    // in style.css for the same reason `--nan` is injected at runtime: a
    // colour in the sheet would look live and be a copy.
    // シェーダのパレットから導出してインラインで置く（シートに書くと写しになる）。
    swatch.style.background = classColorCss(i);
    const text = document.createElement("span");
    text.className = "muted";
    text.textContent = payload.labels[i];
    row.appendChild(swatch);
    row.appendChild(text);
    host.appendChild(row);
  }
}

// More classes than the palette has colours has to be refused, not wrapped.
//
// Reached only if a view is registered with more labels than CLASS_COLORS has
// entries. Two classes sharing a colour is the failure this whole view exists
// to avoid -- the picture still looks like a working classification, and two
// of its regions have silently become one. The shader would paint the extra
// indices in the mask colour, which is at least not a class, but the legend
// would still list them with `classColorCss` returning null; say so instead.
// パレットの色数を超えるクラス数は、回り込ませず断る。2つのクラスが同じ色に
// なると、絵は正しく見えたまま2つの領域が1つに融ける。
function requireClassesFitThePalette(labels, where) {
  if (labels.length > CLASS_COLORS.length) {
    throw new Error(
      `${where} は ${labels.length} クラスありますが、離散パレットの色は ` +
      `${CLASS_COLORS.length} 色しかありません` +
      "（色が足りないまま描くと、別のクラスが同じ色になります）。"
    );
  }
}

// Write one 4x3 into the table.
//
// `null` arrives for every non-finite coefficient. That is the point query's
// contract, not an accident (routers/session.py::_finite_or_none). It is a
// contract for matrices that carry NaN, though, and not a description of the
// ones loaded here: the stored array is finite everywhere, including outside
// the active projection region, and the NaN on screen is written by the views
// at render time. That is the whole reason `#probe-pos` carries an
// out-of-region marker -- see setProbePos, and the measurement quoted there.
// A `null` here therefore means "this matrix really does hold a non-finite
// coefficient at that pixel", which is rare and worth showing as such.
//
// Read back to NaN here rather than branched on in the cell loop, so exactly
// one place decides what a masked coefficient looks like. `fmt(null)` returns
// "—", and in this readout "—" already means "nothing has been read yet" --
// it is the initial text and what a failed query restores. Reusing it would
// spell "outside the projection area" and "no answer" with the same glyph.
// NaN prints "NaN" and takes the mask colour the tiles already paint those
// same pixels in, so the table and the picture agree.
//
// 非有限係数は `null` で届く。これは点クエリの契約であって偶然ではないが
// （routers/session.py::_finite_or_none）、NaNを持ちうる行列のための契約で
// あって、ここで読んでいる行列の説明ではない。保存された配列は有効投影領域の
// 外も含めて全て有限で、画面上のNaNはビューが描画時に書いている。`#probe-pos`
// が独自の「領域外」の印を持つ理由がこれである（setProbePos の実測値を参照）。
// したがってここで `null` が来るということは「その画素に本当に非有限の係数が
// 入っている」を意味し、それは稀であり、そう見えるべきである。
// セル側で分岐させずここで NaN に読み替えるのは、マスクされた係数の見た目を
// 決める場所を1つにするため。`fmt(null)` は「—」を返すが、この欄で「—」は
// 既に「まだ何も読んでいない」を意味しており（初期値であり、失敗時に戻す値
// でもある）、流用すると「投影領域の外」と「答えが無い」が同じ字面になる。
// NaN ならタイルが同じ画素に塗っているマスク色が付き、表と絵が一致する。
function showMatrixValues(values) {
  for (let i = 0; i < DEFAULT_ROWS.length; i++) {
    for (let j = 0; j < DEFAULT_COLS.length; j++) {
      const cell = probeCells[i * DEFAULT_COLS.length + j];
      if (!cell) continue;
      const raw = (values[i] || [])[j];
      const v = raw === null || raw === undefined ? NaN : raw;
      cell.textContent = fmt(v);
      cell.className = Number.isFinite(v) ? "probe-cell" : "probe-cell nan";
    }
  }
}

// The status line is shared with the loader and a hover fires dozens of times
// a second, so repeating one failure would strobe it. Only a message that
// differs from the last one is worth writing.
//
// A success has to put the line back, not merely stop writing to it: leaving
// it would mark the bar as an error over a readout that is working again, for
// as long as nothing else happens to write there -- which, once a matrix is
// loaded and the user is only hovering, is indefinitely. So the sentence the
// line held before the readout claimed it is restored. Only if it is still the
// sentence *this* wrote, though; if the loader has since taken the line, that
// is the newer truth and it stays.
//
// ステータス行は読み込みと共用で、ホバーは毎秒何十回も起きるため、同じ失敗を
// 連投すると点滅する。書くのは前回と違うときだけ。復旧時は「書かない」では
// 足りず、行を元に戻す必要がある——放置すると、既に直っている読み取りについて
// 行が赤いまま残り、他の誰かが書くまで（行列を読み込んだあとホバーしている
// だけなら永久に）そのままになる。ただし戻すのは、行の内容がまだこちらの
// 書いた文である場合に限る。読み込み側が既に書いていればそちらが新しい。
let lastProbeError = null;      // the failure the readout currently claims the line for
let probeErrorText = null;      // exactly what was written, to tell whether it is still ours
let statusBeforeProbe = null;   // what the line said before the readout took it

function reportProbeError(message) {
  if (message === lastProbeError) return;
  if (message) {
    if (lastProbeError === null) statusBeforeProbe = $("status").textContent;
    lastProbeError = message;
    probeErrorText = `読み取りに失敗: ${message}`;
    status(probeErrorText, true);
    return;
  }
  lastProbeError = null;
  if (probeErrorText !== null && $("status").textContent === probeErrorText) {
    status(statusBeforeProbe ?? "");
  }
  probeErrorText = null;
  statusBeforeProbe = null;
}

// Claim the readout for this pixel, and fetch unless something is already in
// flight -- in which case the claim is what the in-flight request picks up
// when it returns.
//
// Every hover claims a new `probeSeq`, whether or not it gets to issue a
// request. That is what keeps the position on screen and the numbers under it
// describing the same pixel: `#probe-pos` is written synchronously, so a
// response for a superseded claim would put one pixel's coefficients under
// another pixel's heading.
//
// ホバーのたびに `probeSeq` を進める（要求を出せたかどうかに関わらず）。
// 位置の表示は同期的に書き換わるので、追い越された要求の応答を描くと
// 「ある画素の係数が別の画素の見出しの下に出る」ことになる。
function requestMatrixProbe(hit) {
  // The matrix on screen, not the newest one registered. See state.shownMatrixId.
  // 画面に出ている行列。最新の登録ではない。
  if (!state.shownMatrixId) { clearMatrixCells(); return; }
  state.probeSeq += 1;
  state.probeWanted = { col: hit.col, row: hit.row, seq: state.probeSeq };
  if (state.probeInFlight) return;
  // Not awaited -- nothing here has anything to wait for. The loop reports its
  // own failures; this catch is the net for a throw that is not a failed
  // fetch, routed through the same de-duplication so a bug on the hover path
  // cannot strobe the status line either.
  // 待つものが無いので await しない。fetch の失敗はループ側が報告するので、この
  // catch はそれ以外の例外のための網であり、同じ重複除去を通す。
  runMatrixProbes().catch((e) => reportProbeError(errorText(e)));
}

async function runMatrixProbes() {
  state.probeInFlight = true;
  try {
    while (state.probeWanted) {
      const want = state.probeWanted;
      state.probeWanted = null;
      const matrixId = state.shownMatrixId;
      try {
        const body = await probeMatrix(matrixId, want.row, want.col);
        if (want.seq !== state.probeSeq) continue;   // the pointer has moved on
        // Only an explicit `false` is a claim. A response without the field is
        // a server that does not answer this question, and "it did not say"
        // must not be rendered as "it said you are inside".
        // 明示的な false のときだけ主張する。フィールドが無い応答は「この問いに
        // 答えないサーバ」であって、「内側だと答えた」ではない。
        setProbePos(want.col, want.row, body.inside_active_region === false);
        showMatrixValues(body.values);
        reportProbeError(null);
      } catch (e) {
        if (want.seq !== state.probeSeq) continue;
        clearMatrixCells();
        reportProbeError(errorText(e));
      }
    }
  } finally {
    state.probeInFlight = false;
  }
}

// --------------------------------------------------------- server params

function serverParams() {
  const out = {};
  if (!state.spec) return out;
  for (const p of state.spec.params) {
    const el = $(`sp-${p.name}`);
    out[p.name] = el ? el.value : p.default;
  }
  return out;
}

// The whole server-side parameter group is submitted at once, by 適用.
//
// Each control used to fetch on its own `change`, so moving the projection
// command from (1,1,1) to (0,0,0) cost three full renders -- about 8 s and 2 GB
// transient on the 1080p matrix, two thirds of it thrown away. `reqId` already
// discarded the stale *results*, but the server still computed all three, and
// concurrently: `render_view` is a sync `def`, so FastAPI runs it in a
// threadpool, and NumPy's heavy kernels release the GIL. The waste was real
// work on real cores, not a queue.
//
// Applied to every view uniformly rather than only to the expensive ones. A
// "cheap views stay instant" rule would need a second source of truth for which
// renders are cheap, kept in the frontend, agreeing with the backend by
// convention -- and it would be wrong the first time a view's cost changed
// without the frontend hearing about it. One rule, no exceptions: a control
// stages a value, 適用 sends the group.
//
// サーバ側パラメータは1つずつではなく、グループごとに 適用 で送る。以前は
// コントロールの change ごとにフェッチしていたため、投影コマンドを
// (1,1,1) から (0,0,0) へ動かすだけで3回のフルレンダになっていた（1080pで
// 約8秒・一時的に約2GB、うち2/3は捨てられる）。reqId が捨てるのは結果だけで、
// サーバは3回とも計算する。しかも render_view は同期関数なので FastAPI は
// スレッドプールで走らせ、NumPy の重い処理は GIL を離すため、3回は並行に
// 走って実際にコアを食う。
//
// 安いビューだけ即時にする、という区別は付けない。それには「どのレンダが
// 安いか」という第二の情報源をフロント側に持ち、規約でバックエンドと
// 一致させ続ける必要があり、ビューのコストが変わった瞬間に嘘になる。
// 規則は1つ——コントロールは値を置くだけ、送るのは 適用。
// Whether the controls on screen differ from what the last request sent --
// *derived*, not tracked by events.
//
// This started as a pair of set/clear calls: `change` armed the ring, and
// reload() cleared it. That is unfixable by ordering, because the browser fires
// `keydown` before it commits the value and fires `change`. Pressing Enter
// therefore ran apply -> reload -> clear, and only *then* the `change` that
// re-armed the ring -- leaving the button asserting "not applied" about a value
// that had just been applied, with nothing left to clear it. A UI lying about
// its own state is the exact failure class the rest of this branch removes.
//
// Comparing the two values has no such ordering to get wrong: whenever anything
// happens, ask the question again. `keydown` reads the input's live `value`
// (already updated by the time the key event fires), so the request carries the
// new value and the `change` that follows finds nothing pending.
//
// Guarded on `state.matrixId` because with no matrix loaded nothing is on
// screen, so no control can differ from what is displayed.
//
// 画面上のコントロールが、最後に送った値と違うかどうか。イベントで追うのでは
// なく比較で導く。以前は `change` で立て reload() で降ろしていたが、これは
// 順序では直せない——ブラウザは値を確定して `change` を出す*前*に `keydown` を
// 出すため、Enter では 適用→reload→解除 のあとに `change` が来て旗を立て直し、
// 「適用済みの値について未適用と主張するボタン」が残る。UIが自分の状態について
// 嘘をつくのは、このブランチが他所で潰してきた失敗そのものである。
// 比較なら間違える順序が無い。何か起きるたびに問い直せばよい。`keydown` の
// 時点で入力欄の `value` は既に新しいので、送られるのは新しい値であり、後続の
// `change` は「差分なし」を見る。行列未読み込みなら画面に何も無いので比較しない。
function refreshParamsPending() {
  const apply = $("sp-apply");
  if (!apply) return;
  const pending = !!state.matrixId
    && JSON.stringify(serverParams()) !== JSON.stringify(state.sentParams);
  apply.className = pending ? "primary pending" : "primary";
  // Only claim the status line when there is something to say. Clearing is
  // reload()'s business -- it owns the line once a fetch is under way.
  // 何か言うことがあるときだけステータス行を取る。消すのは reload() の仕事。
  if (pending) status("パラメータ変更あり — 適用 を押すと再取得します");
}

function applyServerParams() {
  reload().catch((e) => status(errorText(e), true));
}

function buildServerParams(spec) {
  const host = $("server-params");
  host.textContent = "";
  if (!spec.params.length) {
    host.innerHTML = '<div class="muted">なし</div>';
    return;
  }
  for (const p of spec.params) {
    const label = document.createElement("label");
    label.className = "lbl";
    label.textContent = p.label || p.name;
    host.appendChild(label);
    let el;
    if (p.type === "select") {
      el = document.createElement("select");
      for (const choice of p.choices) {
        const opt = document.createElement("option");
        opt.value = choice;
        opt.textContent = choice;
        el.appendChild(opt);
      }
    } else {
      el = document.createElement("input");
      el.type = p.type === "number" ? "number" : "text";
      el.step = "any";
      el.className = "mono";
    }
    el.id = `sp-${p.name}`;
    el.value = p.default;
    // A runtime id the sidebar has already loaded wins over the declared
    // default, which for these is the empty string. The controls are rebuilt
    // on every view switch, so without this the ターゲットID box would come
    // back empty for a target that is still loaded, and the view would refuse
    // a request the user has no reason to think incomplete.
    // 既に読み込まれている実行時IDは宣言側の既定（空文字）より優先する。
    // コントロールはビュー切替のたびに作り直されるため、これが無いと
    // 読み込み済みのターゲットについて欄だけが空に戻る。
    const loadedId = state[LOADED_ID_FOR_PARAM[p.name]];
    if (loadedId) el.value = loadedId;
    // Stages the value; it does not fetch. See refreshParamsPending above.
    // 値を置くだけでフェッチはしない。
    el.addEventListener("change", () => { refreshParamsPending(); });
    // Enter in a typed field is "I am done with this value", so it submits the
    // group -- otherwise a keyboard user has to reach for the mouse.
    // 入力欄での Enter は「この値でよい」の意なのでグループを送る。
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") applyServerParams();
    });
    host.appendChild(el);
  }
  // The one control allowed to hit the network.
  // ネットワークへ行ってよい唯一のコントロール。
  const apply = document.createElement("button");
  apply.id = "sp-apply";
  apply.className = "primary";
  apply.textContent = "適用";
  apply.addEventListener("click", () => { applyServerParams(); });
  host.appendChild(apply);
}

// -------------------------------------------------- target and LUT loaders
//
// The `compensation` view needs two ids that only exist at runtime, and until
// these controls existed there was no way to obtain one from a browser: the
// view shipped complete and reachable only by POSTing to /api/target with curl.
// These load a path, and write the id the server hands back straight into the
// matching `sp-*` box -- the id is never something to copy by hand.
//
// **They stage; they do not render.** The one rule of the server-parameter
// group is that only 適用 hits the view endpoint (see refreshParamsPending's
// note and the cost measurement above it), and a loader that re-rendered on
// its own would be a second way in. So a load fills the box, lights 適用, and
// stops. The user presses 適用 once, for the target and the LUT together,
// instead of paying for a full compensation render per loaded file.
//
// The loaders themselves *do* fetch, and that is not the same thing: they are
// 読み込み buttons like the matrix's, not parameter controls.
//
// ターゲットとLUTの読み込み欄。`compensation` は実行時にしか存在しない2つの
// IDを必要とし、これらが無い間はブラウザから取得する手段が無かった。読み込む
// と返ってきたIDを対応する `sp-*` 欄へ自分で書き込む（手でコピーさせない）。
// **置くだけで再描画はしない。** ビュー取得へ行ってよいのは 適用 だけという
// 規則があり、読み込みが独自に再描画すると2つ目の入口になる。ターゲットとLUT
// をまとめて 適用 1回で反映させる。読み込み自体がフェッチするのは別の話で、
// これらは行列と同じ「読み込み」ボタンであってパラメータ欄ではない。

// Write a value into a server-parameter box, if the active view has one.
//
// Returns whether it did. A view that does not declare the parameter is not an
// error -- `surface` has no ターゲットID -- and the id is still remembered in
// `state`, so switching to `compensation` afterwards picks it up through
// buildServerParams.
// 値をサーバ側パラメータ欄へ書く。宣言していないビューでは書けないが、それは
// 誤りではない（`state` に残るので切替時に反映される）。
function setServerParam(name, value) {
  const el = $(`sp-${name}`);
  if (!el) return false;
  el.value = value;
  refreshParamsPending();
  return true;
}

// Counted, not a boolean, for the same reason setLoadBusy is: two loads can
// overlap (the button and the image dropdown), and a boolean would be cleared
// by whichever finished first.
// 真偽値ではなく数える（ボタンと一覧の2経路が重なりうる）。
let targetLoadsInFlight = 0;
let lutLoadsInFlight = 0;

function setTargetBusy(busy) {
  targetLoadsInFlight += busy ? 1 : -1;
  const disabled = targetLoadsInFlight > 0;
  $("btn-load-target").disabled = disabled;
  $("target-images").disabled = disabled;
}

function setLutBusy(busy) {
  lutLoadsInFlight += busy ? 1 : -1;
  $("btn-load-lut").disabled = lutLoadsInFlight > 0;
}

// The image picker for a directory of targets.
//
// Hidden for a single file, which has exactly one selectable entry: a dropdown
// offering one option is a control that cannot do anything, and this sidebar
// already argues elsewhere against showing those.
// ディレクトリのときだけ出す。単一ファイルには選択肢が1つしか無く、何もでき
// ないコントロールを見せることになる。
function showTargetImages(meta) {
  const select = $("target-images");
  select.textContent = "";
  const names = meta.images || [];
  select.hidden = !meta.is_directory;
  for (let i = 0; i < names.length; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = names[i];
    select.appendChild(opt);
  }
  select.value = String(meta.index ?? 0);
}

// Load one target image (or one entry of a directory of them).
//
// `path` is passed in rather than read from the box here, because the two
// callers mean different things by it: the button means "whatever is typed
// now", and the image dropdown means "the directory this listing came from".
// Reading the box in both would let an edited path be combined with an index
// chosen from the previous directory's listing -- a request that succeeds and
// loads the wrong image.
// `path` を引数で受けるのは、呼び出し側2つの意味が違うため。ボタンは「今
// 入力されているパス」、一覧は「この一覧の出所であるディレクトリ」を指す。
// 両方が入力欄を読むと、編集後のパスと前の一覧のindexが組み合わさり、成功
// するが別の画像を読む要求になる。
async function loadTargetImage(path, index = 0) {
  if (!path) { status("ターゲット画像のパスを入力してください", true); return; }
  setTargetBusy(true);
  try {
    status(`ターゲットを読み込み中: ${path}`);
    const res = await loadTarget(path, index);
    state.targetId = res.id;
    state.targetMeta = res.meta;
    showTargetImages(res.meta);
    setServerParam("target", res.id);
    const [height, width] = res.meta.shape;
    const name = (res.meta.images || [])[res.meta.index] ?? res.meta.path;
    $("target-info").textContent = `${res.id} / ${width}×${height}`;
    status(`ターゲット ${res.id}（${name}, ${width}×${height}）— 適用 で反映`);
  } finally {
    setTargetBusy(false);
  }
}

// Load a measured inverse LUT and plot it.
//
// Two requests, and both are needed before anything is published: the POST
// registers the file and the GET brings back the samples to draw. `state.lutId`
// is written only after the curve is in hand, so a LUT whose curve could not be
// fetched does not end up named in the LUT ID box as though it were ready to
// use.
// POST で登録し、GET でカーブを取る。`state.lutId` を書くのはカーブが手元に
// 来てからで、カーブを取れなかったLUTが「使える」顔で欄に入ることはない。
async function loadInverseLut() {
  const path = $("lut-path").value.trim();
  if (!path) { status("逆LUTのパスを入力してください", true); return; }
  setLutBusy(true);
  try {
    status(`逆LUTを読み込み中: ${path}`);
    const res = await loadLut(path);
    const curve = await fetchLutCurve(res.id);
    state.lutId = res.id;
    state.lutCurve = curve;
    setServerParam("lut", res.id);
    $("lut-gamma-value").textContent = `γ=${curve.gamma}`;
    redrawLutCurve();
    const source = (curve.meta && curve.meta.source) || "?";
    const [channels, samples] = res.meta.shape;
    $("lut-info").textContent = `${res.id} / ${channels}×${samples} / ${source}`;
    status(`逆LUT ${res.id}（source=${source}, ${channels}×${samples}）— 適用 で反映`);
  } finally {
    setLutBusy(false);
  }
}

// Redraw from the curve already in hand. Called on load, on the overlay
// toggle, and on a window resize -- the canvas is sized by CSS, so its backing
// store has to be re-established whenever the box changes size or the display
// changes DPR.
// 手元のカーブから描き直す。読み込み時・重ね描きの切替時・窓のリサイズ時に
// 呼ぶ（canvasはCSSで伸縮するので、そのたびにバッキングストアを取り直す）。
function redrawLutCurve() {
  if (!state.lutCurve) return;
  const overlay = $("lut-gamma-overlay").checked;
  lastLutPlot = drawLutCurve($("lut-curve"), state.lutCurve, {
    gamma: overlay ? state.lutCurve.gamma : null,
  });
}

function wireLoaders() {
  $("btn-load-target").addEventListener("click", () => {
    loadTargetImage($("target-path").value.trim())
      .catch((e) => status(errorText(e), true));
  });
  $("target-path").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btn-load-target").click();
  });
  // Selecting an image re-loads *that* index from the directory the current
  // listing came from, so `meta.index` on the server and the id in the box
  // stay the same fact. Nothing is derived locally from the previous response.
  // 一覧から選ぶと、その一覧の出所のディレクトリからその index を読み直す。
  // ローカルで前の応答から導かない。
  $("target-images").addEventListener("change", () => {
    const meta = state.targetMeta;
    if (!meta) return;
    loadTargetImage(meta.path, Number($("target-images").value) || 0)
      .catch((e) => status(errorText(e), true));
  });

  $("btn-load-lut").addEventListener("click", () => {
    loadInverseLut().catch((e) => status(errorText(e), true));
  });
  $("lut-path").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btn-load-lut").click();
  });
  $("lut-gamma-overlay").addEventListener("change", () => { redrawLutCurve(); });
  window.addEventListener("resize", () => { redrawLutCurve(); });
}

// ------------------------------------------------------------- rendering

function applyDisplay() {
  for (let c = 0; c < views.length; c++) {
    const r = state.ranges ? state.ranges[c] : { vmin: -1, vmax: 1 };
    views[c].setParams({ ...display, vmin: r.vmin, vmax: r.vmax });
  }
}

// The controls that make up the tone chain: exposure, γ (both its input and
// its slider) and dither. Enabled exactly when the shader branch that is going
// to draw actually applies one -- `KIND_TRAITS[kind].toneChain`.
//
// The colormap dropdown is *not* in this list even though it is disabled for
// the same two kinds. It answers a different question -- "does this kind turn
// a number into a colour" -- and rgb answers no to that while answering yes to
// the tone chain. One list for both would be a coincidence of today's three
// kinds, not a rule, and the first kind that broke it would break it silently.
//
// Values are left alone, not reset. A user who set -1 EV on the coefficient
// tiles, looked at the gamut view and came back has not asked for 0 EV; the
// slider is dead while the categorical picture is up and alive again with the
// same value afterwards.
//
// トーンチェーンを構成するコントロール（露出・γ・ディザ）。描画する
// シェーダ経路が実際にトーンチェーンを掛けるときだけ有効にする。
// カラーマップはこの一覧に入れない——同じ2つの kind で無効になるが問いが
// 違い（rgb はトーンチェーンを掛けるがカラーマップは通さない）、まとめると
// 今日の3つの kind でたまたま一致しているだけの規則になる。値は消さない。
const TONE_CONTROL_IDS = ["ev-slider", "gamma-input", "gamma-slider", "dither-check"];

function setToneControlsEnabled(enabled) {
  for (const id of TONE_CONTROL_IDS) {
    const el = $(id);
    if (el) el.disabled = !enabled;
  }
}

// Does the *active* view declare this server-side parameter?
//
// `normalize` belongs to coef_tiles, not to views in general, and a view that
// does not declare it must not inherit it -- that leak is what breaks the
// "register in Python, no JavaScript change" promise.
// `normalize` は coef_tiles 固有のパラメータであり、全ビュー共通ではない。
// 宣言していないビューがそれを引き継いではならない。
function specDeclares(name) {
  return !!state.spec && state.spec.params.some((p) => p.name === name);
}

function computeRanges(traits) {
  const p = state.payload;

  // An rgb view has one tile and therefore one range, and the three components
  // must share it. A per-channel range would rescale R, G and B independently,
  // which is a white balance the data does not have -- the picture would still
  // look like a plausible surface colour while being the wrong colour.
  // Displayed as [vmin, vmax] rather than ±vmax because this range is not
  // symmetric: vmin is 0 for a positive response and negative for an
  // ambient-only one (see views/surface.py).
  // rgbビューはタイル1枚＝レンジ1組で、3成分はそれを共有しなければならない。
  // チャンネルごとに正規化すると、データに無いホワイトバランスを勝手に掛けた
  // 「もっともらしいが違う色」になる。レンジは対称でないので ± ではなく
  // [vmin, vmax] で出す。
  if (traits.sharedRange) {
    state.ranges = [{ vmin: p.vmin, vmax: p.vmax }];
    // A categorical range is a count, not a measurement, and it is written as
    // one. `[0.0000, 3.0000]` invites the reading that 1.5 is a value this
    // view can hold; and the caption cannot be `labels.join(" ")` either,
    // because for this kind that is the four class names crammed under the
    // tile, restating the legend in the one place with no room for it.
    // クラスの範囲は測定値ではなく個数なので、そのように書く。ラベルを
    // 並べる書き方も使えない（凡例を最も狭い場所で繰り返すことになる）。
    const classes = traits.labelsAreClasses;
    const text = classes
      ? `クラス ${p.vmin}–${p.vmax}`
      : `[${fmt(p.vmin)}, ${fmt(p.vmax)}]`;
    if (captions.length) {
      captions[0].textContent = classes
        ? `${p.labels.length} クラス（凡例参照）`
        : `${p.labels.join(" ")}  ${text}`;
    }
    $("range-info").textContent = text;
    return;
  }

  // Defence in depth, not the active guard. What actually keeps `normalize`
  // out of a view that does not declare it is serverParams(), which iterates
  // state.spec.params -- so today this condition is unreachable and no test
  // covers it (removing it changes nothing observable; verified). It is here
  // so that a future serverParams() which reads controls out of the DOM, or
  // carries values across a view switch, cannot silently re-open the leak.
  // Do not read it as tested behaviour.
  // これは多重防御であり現に効いているガードではない。実際に守っているのは
  // state.spec.params を走査する serverParams() の側で、この条件は今日
  // 到達不能であり、対応するテストも無い（外しても観測可能な差は出ない）。
  // 将来 serverParams() がDOMを直接読むようになった場合の保険である。
  const mode = specDeclares("normalize") ? serverParams().normalize : "shared";
  if (mode === "per_tile") {
    // The wire format carries a single (vmin, vmax) pair, so 12 independent
    // ranges have to be derived here from the array we already hold.
    // ヘッダは1組のレンジしか運べないので、タイルごとのレンジは手元で出す。
    const abs = percentileAbsPerChannel(p.data, p.channels, 99, p.layout);
    state.ranges = abs.map((a) => ({ vmin: -a, vmax: a }));
  } else {
    state.ranges = Array.from({ length: p.channels }, () => ({ vmin: p.vmin, vmax: p.vmax }));
  }
  for (let c = 0; c < captions.length && c < p.channels; c++) {
    const r = state.ranges[c];
    captions[c].textContent = `${p.labels[c] ?? c}  ±${fmt(r.vmax)}`;
  }
  const shared = state.ranges.every((r) => r.vmax === state.ranges[0].vmax);
  $("range-info").textContent = shared
    ? `±${fmt(state.ranges[0].vmax)}`
    : "タイルごと";
}

function renderMeta() {
  const m = state.meta;
  $("meta-info").textContent =
    `${m.shape[1]}×${m.shape[0]} ${m.dtype} / ${m.space} / ` +
    (m.active_region ? `有効領域 [${m.active_region.join(", ")}]` : "有効領域なし");

  const note = $("space-note");
  if (m.space === "camera") {
    note.textContent =
      "カメラ座標の行列です。プロジェクタ座標へのワープは未実装のため、" +
      "係数はカメラ画素のまま表示しています。";
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

// An rgb view is three channels by definition -- one per colour.
//
// Without this the failure is loud but undiagnostic. A shorter payload reaches
// packRgb, whose planar loop reads past the end of the array, and JavaScript
// answers `undefined` rather than throwing; that becomes NaN in the Float32Array
// and the NaN mask paints the entire tile. Nothing is drawn wrong -- but the
// user sees "everything is masked" with no route back to "this view declared
// the wrong channel count". Say which it is.
// rgb は定義上3チャンネル。これが無いと packRgb が配列の終端を越えて読み、
// undefined → NaN → タイル全面がマスク色になる。誤った絵は出ないが、
// 「全部マスク」から「チャンネル数の宣言ミス」へは辿り着けない。
function requireRgbHasThreeChannels(kind, channels, where) {
  if (kind === "rgb" && channels !== 3) {
    throw new Error(
      `${where} は kind="rgb" ですがチャンネル数が ${channels} です` +
      "（rgb は R/G/B の3チャンネルのみ描画できます）。"
    );
  }
}

// `req` is inherited when load() drives the reload, so the whole click stays
// one logical request; a bare reload() (a server-parameter change) claims a
// new one.
async function reload(req = beginRequest()) {
  if (!state.matrixId || !state.spec) return;
  // Captured once, so the id this request fetched from is the id published
  // with its payload. Re-reading `state.matrixId` after the await would let a
  // newer load's id be attached to this transfer's pixels.
  // 発行時に控える。await のあとに読み直すと、この転送の画素に新しい行列のIDが
  // 付きうる。
  const matrixId = state.matrixId;
  // Checked from the registry first, so a view that cannot be drawn is refused
  // before the transfer rather than after it.
  // まず登録内容で判定し、描けないビューは転送前に断る。
  const specName = `ビュー "${state.spec.label || state.spec.key}"`;
  requireSupportedKind(state.spec.kind, specName);
  requireRgbHasThreeChannels(state.spec.kind, state.spec.channels, specName);
  const params = serverParams();
  // What the screen is about to be showing. Recorded at issue time rather than
  // on arrival, so the indicator answers "are the controls ahead of the last
  // request?" -- the in-flight state is the status line's job, not the ring's.
  // これから画面が表す値。到着時ではなく発行時に記録する。指標が答えるのは
  // 「コントロールは最後の要求より先に進んでいるか」であり、転送中かどうかは
  // ステータス行の担当である。
  state.sentParams = params;
  refreshParamsPending();
  status(`ビュー取得中… (${state.spec.key} ${JSON.stringify(params)})`);

  const t0 = performance.now();
  const payload = await fetchView(matrixId, state.spec.key, params);
  const t1 = performance.now();
  if (isStale(req)) return;   // a newer request owns the screen now
  // Backstop: the registry and the response could disagree. `traitsFor` is the
  // supported-kind check *and* the single source for everything decided by
  // kind below -- the tile layout, the range calculation, the colormap control
  // and the texture upload all read this one object.
  // レジストリと応答が食い違いうるための保険。`traitsFor` は対応kindの検査を
  // 兼ねており、以下で kind によって決まるもの（タイル配置・レンジ・
  // カラーマップ・テクスチャ）はすべてこの1つのオブジェクトを読む。
  const traits = traitsFor(payload.kind, "サーバの応答");
  requireRgbHasThreeChannels(payload.kind, payload.channels, "サーバの応答");
  // Refused before anything is published, like the rgb channel count above: a
  // legend with two identically-coloured rows is not something to draw and
  // then explain.
  // 上のチャンネル数と同じく、公開前に断る。
  if (traits.labelsAreClasses) {
    requireClassesFitThePalette(payload.labels, "サーバの応答");
  }
  state.payload = payload;
  // The traits the pixels were rendered through, published with them. See
  // `state.traits`.
  state.traits = traits;
  // The pixels and the matrix they came from, published together. The readout
  // asks this id, so it can never describe a matrix other than the one drawn.
  // 画素と、その出所の行列IDを同時に確定する。読み取り欄はこのIDに問い合わせる。
  state.shownMatrixId = matrixId;
  // The heading over #view-values, set with the payload it describes.
  //
  // Two panels of numbers now sit side by side and only one of them is the
  // matrix; without this the three values under `surface` are just three
  // unlabelled numbers next to twelve coefficients. Written here rather than
  // when the selector changes, so a view that is refused -- an unsupported
  // kind, a server-side 400 -- leaves the heading naming what is still on
  // screen instead of a view that never rendered.
  // #view-values の見出し。数値の板が2枚並ぶが行列はそのうち1枚だけであり、
  // これが無いと `surface` の3つは12個の係数の隣に並んだ無名の数値になる。
  // セレクタの変更時ではなくペイロード確定時に書くので、拒否されたビューの
  // 名前が画面に残ることはない。
  $("view-title").textContent = state.spec.label || state.spec.key;
  // ...and whether that panel belongs on screen at all.
  //
  // The readout table already shows the matrix on every view. Where the picture
  // *is* the matrix, this panel is the same twelve numbers again -- measured at
  // 974px of a 1600px strip, the widest element in it, and the reason the strip
  // doubled to 224px at 900px wide, taking 113px back off the tiles to restate
  // what was already 20px to the left.
  //
  // Read from the registry rather than sniffed from the payload, and never from
  // `kind`: `coef_tiles` and `plain_tiles` are both `scalar`, and so would be a
  // residual-error view, which is not the matrix. Only an explicit `true` hides
  // anything, so a view that does not declare the field -- an older server, a
  // newly registered one -- keeps its panel rather than being silently blanked.
  //
  // The one thing the panel used to add on a coefficient view was that a masked
  // pixel read `NaN` here while the table printed the stored value. The
  // out-of-region marker on `#probe-pos` now says that in words, in red, above
  // the numbers it disclaims.
  //
  // その板を出すかどうか。読み取り表はどのビューでも行列を出しているので、
  // 描かれているものが行列そのものなら、この板は同じ12個の繰り返しである
  // （1600pxの帯で974px＝最も広い要素。900pxでは帯が224pxになり、その113pxは
  // タイルから奪われていた）。判断はレジストリの宣言に従う。`kind` では
  // 決めない——`coef_tiles` も `plain_tiles` も `scalar` であり、残差ビューも
  // またそうだが、それは行列ではない。明示的な `true` のときだけ隠すので、
  // 宣言の無いビュー（古いサーバ、新しく登録されたビュー）は黙って空白に
  // ならない。係数ビューでこの板が唯一足していた情報——マスク画素でここだけ
  // NaN になること——は、`#probe-pos` の印が言葉で述べるようになった。
  $("view-panel").hidden = state.spec.values_are_matrix === true;

  const layout = layoutFor(payload, state.layout, traits);
  // Compared by value, and only then rebuilt: buildGrid disposes and recreates
  // one WebGL context per tile, and the browser caps how many can be live at
  // once (~16 in Chrome, against 12 tiles).
  // 値で比較してから組み直す。タイル1枚につきWebGLコンテキスト1つを捨てて
  // 作り直すため、ブラウザの上限（Chromeで約16、タイルは12枚）に近い。
  const layoutChanged = JSON.stringify(layout) !== JSON.stringify(state.layout);
  state.layout = layout;
  // Tiles, not channels: an rgb view puts three channels on one tile.
  // チャンネル数ではなくタイル数。rgbは3チャンネルを1枚に載せる。
  const tiles = layout.rows.length * layout.cols.length;
  const gridRebuilt = views.length !== tiles || layoutChanged;
  if (gridRebuilt) {
    buildGrid(state.layout);
    await nextFrame();
    if (isStale(req)) return;
  }

  const t2 = performance.now();
  computeRanges(traits);
  const t3 = performance.now();

  // The colormap gives a *number* a colour; rgb data is already a colour and
  // the shader's rgb branch has no colormap in it at all. Disable the control
  // rather than leave a dropdown that silently does nothing.
  // カラーマップは数値に色を与える仕組みで、rgb経路はそもそも通さない。
  // 何も起きないドロップダウンを残さず無効化する。
  $("cmap-select").disabled = !traits.colormap;

  // Exposure, γ and dither, for the kind that is now on screen. See
  // setToneControlsEnabled -- and note that the shader ignoring them is the
  // half that actually holds; this is the half the user can see.
  // 露出・γ・ディザ。実効的な保証はシェーダ側が無視することにあり、こちらは
  // 「効かないコントロールを見せない」ためのもの。
  setToneControlsEnabled(traits.toneChain);

  // The class legend, from this payload's own labels. Cleared and hidden for
  // every other kind, so a legend cannot outlive the view that explained it.
  // クラス凡例。他の kind では消して隠す（説明していたビューより長生きさせない）。
  renderClassLegend(payload, traits);

  // Decide before the upload, while the old image is still around: a reframe
  // is warranted only when there is no view worth keeping -- a rebuilt grid, a
  // first image, or different dimensions. Re-fetching the same matrix with a
  // different server-side parameter is none of those, and refitting there
  // would silently discard the pan and zoom the user had set up.
  // 組み直し・初回・寸法変更のときだけ再フレーミングする。normalize 変更の
  // ような同寸法の再取得でパン/ズームを捨てない。
  const previous = views[0].image;
  const needsFit =
    gridRebuilt ||
    !previous ||
    previous.width !== payload.width ||
    previous.height !== payload.height;

  syncBusy = true;   // suppress 12x11 cross-syncs while every tile is loaded
  try {
    // Iterating tiles rather than channels: the two are the same count for a
    // scalar view, but an rgb view is one tile that consumes all three
    // channels. GLView is handed the `kind` itself and reads the same trait
    // row this function did, so the two cannot reach different conclusions
    // about the same payload.
    // チャンネルではなくタイルを回す。scalar では同数だが、rgb は3チャンネルを
    // 1枚が受け取る。GLView には kind をそのまま渡し、向こうもこの関数と同じ
    // 特性表の行を読む（同じペイロードについて別々の結論に至りようがない）。
    for (let t = 0; t < views.length; t++) {
      views[t].setImage({
        data: payload.data,
        width: payload.width,
        height: payload.height,
        channels: payload.channels,
        channel: t,
        layout: payload.layout,
        kind: payload.kind,
      });
    }
  } finally {
    syncBusy = false;
  }
  const t4 = performance.now();
  applyDisplay();
  for (const v of views) v.finish();   // block until the driver has consumed it
  const t5 = performance.now();
  // One fit, propagated to the other tiles, so all 12 share the exact same
  // transform even if their backing stores round to different integer sizes.
  // GLView.setImage draws the same conclusion per tile; this is what keeps
  // them identical.
  if (needsFit) views[0].fit();
  state.pinned = null;
  $("probe-pin").textContent = "";
  // What the readout was describing is no longer on screen -- a different
  // matrix, or the same one through a different view. Clearing also retires
  // any point query still in flight, so it cannot repaint the table afterwards.
  // 読み取りが説明していたものは画面から消えた。飛んでいる点クエリの権利も
  // ここで失効させる。
  clearReadout();

  const bytes = payload.data.byteLength;
  const perf = {
    bytes,
    megabytes: +(bytes / 1048576).toFixed(1),
    fetchMs: +(t1 - t0).toFixed(1),
    rangesMs: +(t3 - t2).toFixed(1),
    uploadMs: +(t4 - t3).toFixed(1),
    uploadFinishMs: +(t5 - t4).toFixed(1),
    totalMs: +(t5 - t0).toFixed(1),
    mbPerSec: +(bytes / 1048576 / ((t1 - t0) / 1000)).toFixed(1),
  };
  if (performance.memory) {
    perf.jsHeapMB = +(performance.memory.usedJSHeapSize / 1048576).toFixed(1);
    perf.jsHeapLimitMB = +(performance.memory.jsHeapSizeLimit / 1048576).toFixed(1);
  }
  window.__perf = perf;
  console.log("[perf]", perf);

  $("perf-fetch").textContent = `${perf.fetchMs} ms (${perf.megabytes} MB, ${perf.mbPerSec} MB/s)`;
  $("perf-ranges").textContent = `${perf.rangesMs} ms`;
  $("perf-upload").textContent = `${perf.uploadMs} ms + finish ${perf.uploadFinishMs} ms`;
  $("perf-heap").textContent = perf.jsHeapMB ? `${perf.jsHeapMB} MB` : "計測不可";
  // What was rendered, and nothing that is already on screen.
  //
  // This line used to end with `転送 570.8 ms / レンジ 0.3 ms / GPU転送 51.2+0.7
  // ms`, twenty pixels below a 計測 panel showing two of those three timings
  // verbatim, and it still opens with dimensions the toolbar also states. The
  // only datum here that lived nowhere else was レンジ, which is now the panel's
  // second row -- so the line goes back to being what it is everywhere else in
  // this app: the transient-state line, for errors, 「ビュー取得中…」 and the
  // unapplied-parameter notice, saying what is on screen when there is nothing
  // else to say.
  //
  // 画面上の他所にあるものは書かない。この行は以前 計測 パネルの20px下で同じ
  // 転送・GPU転送を繰り返し、寸法もツールバーと重複していた。ここにしかない
  // 値はレンジだけだったので、それをパネルへ移して行を戻す。この行は他の場面
  // では過渡的な状態（エラー、取得中、未適用の通知）を出す場所であり、
  // 言うことが無いときは何を表示しているかだけを述べればよい。
  status(`${payload.width}×${payload.height}×${payload.channels} を表示中`);
}

// How many load() calls are still running. The 読み込み button is disabled
// while that count is above zero and released the moment it reaches zero.
//
// This deliberately does *not* key off state.reqId. Anything that calls
// reload() -- the 正規化 dropdown, the view selector -- bumps that counter, so
// a load running at the time would find itself "stale" in its own finally
// block and skip the re-enable, leaving the button dead until the page is
// reloaded (and Enter in the path box dead with it, since it delegates to
// .click()). Counting in-flight loads answers the question the button
// actually asks -- "is a load running?" -- and no other caller can perturb it.
// ボタンの状態は state.reqId とは別に数える。normalize やビュー切替の
// reload() が reqId を進めるため、それを借りると読み込み中のload()が自分を
// stale と判定してボタンを解放できず、再読み込みするまで押せなくなる。
let loadsInFlight = 0;

function setLoadBusy(busy) {
  loadsInFlight += busy ? 1 : -1;
  $("btn-load").disabled = loadsInFlight > 0;
}

async function load() {
  const path = $("path-input").value.trim();
  if (!path) { status("パスを入力してください", true); return; }

  const req = beginRequest();
  setLoadBusy(true);
  try {
    status(`行列を読み込み中: ${path}`);
    const res = await loadMatrix(path);
    // Nothing about the previous matrix may be published once a newer load has
    // started: the meta header and state.payload have to describe the same
    // matrix at all times, so bail before touching either.
    // 新しい読み込みが始まっていたら、メタも payload も一切書き換えない。
    if (isStale(req)) return;
    state.matrixId = res.id;
    state.meta = res.meta;
    renderMeta();
    await reload(req);
  } finally {
    setLoadBusy(false);
  }
}

// ------------------------------------------------------------------ setup

function wireDisplayControls() {
  const push = () => applyDisplay();

  $("ev-slider").addEventListener("input", (e) => {
    display.exposure = Number(e.target.value);
    $("ev-val").textContent = display.exposure.toFixed(1);
    push();
  });
  // The slider is for coarse dragging at 0.01; the number box accepts whatever
  // precision is typed, because the useful gamma for the p99 range sits near
  // 0.45 and a tenth of a step is visible on a diverging map.
  // スライダは 0.01 刻みの粗調整。数値入力は打ち込まれた桁をそのまま採用する。
  // p99レンジで有用なγは 0.45 付近で、発散カラーマップでは 0.01 の
  // さらに下の桁が見た目に出るため。
  $("gamma-slider").addEventListener("input", (e) => {
    display.gamma = Number(e.target.value);
    $("gamma-input").value = display.gamma.toFixed(2);
    push();
  });
  $("gamma-input").addEventListener("input", (e) => {
    const typed = Number.parseFloat(e.target.value);
    if (!Number.isFinite(typed)) return;   // mid-edit ("", "-", "0.") — wait
    display.gamma = Math.min(Math.max(typed, GAMMA_MIN), GAMMA_MAX);
    // Only follow the slider's own range; a typed value outside it parks the
    // handle at the end rather than silently clamping what the user asked for.
    // スライダの可動域外を打ち込んだ場合はハンドルを端に置くだけで、
    // 入力値そのものは丸めない。
    $("gamma-slider").value = String(display.gamma);
    push();
  });
  $("cmap-select").addEventListener("change", (e) => {
    display.colormap = Number(e.target.value);
    push();
  });
  $("dither-check").addEventListener("change", (e) => {
    display.dither = e.target.checked ? 1 : 0;
    push();
  });
  $("btn-reset-view").addEventListener("click", () => {
    if (views.length && state.payload) views[0].fit();
  });
  $("btn-load").addEventListener("click", () => {
    load().catch((e) => status(errorText(e), true));
  });
  $("path-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btn-load").click();
  });
}

// Populate the view picker from /api/views.
//
// Without it, `state.spec = specs[0]` plus an insertion-ordered REGISTRY (with
// coef_tiles registered first) makes a newly registered view unreachable from
// the GUI, and the spec's promise that a new view needs "a Python registration
// and no JavaScript change" is false on its first use.
// これが無いと、新しく登録したビューはGUIから一生選べない。
function buildViewSelect(specs) {
  const el = $("view-select");
  el.textContent = "";
  for (const spec of specs) {
    const opt = document.createElement("option");
    opt.value = spec.key;
    opt.textContent = spec.label || spec.key;
    el.appendChild(opt);
  }
  el.value = specs[0].key;
  el.addEventListener("change", () => {
    const spec = specs.find((s) => s.key === el.value);
    if (!spec) return;
    state.spec = spec;
    buildServerParams(spec);
    if (!state.matrixId) {
      status(`ビュー "${spec.label || spec.key}" を選択 — パスを入れて読み込み`);
      return;
    }
    reload().catch((e) => status(errorText(e), true));
  });
}

async function main() {
  // Debug/automation handle. The smoke test and any CDP-driven measurement need
  // to reach the views without going through synthetic input events.
  // スモークテストや自動計測から views へ触るためのハンドル。
  // `lutPlot` is a getter, not a copy: it is read after a redraw the driver
  // itself triggered, and a snapshot taken here would describe the first draw
  // forever.
  window.__viewer = {
    state, views, display, applyDisplay, reload,
    get lutPlot() { return lastLutPlot; },
  };

  // Derive the legend swatch from the shader's own mask colour rather than
  // restating it in CSS. See NAN_COLOR in glview.js.
  // 凡例の色はシェーダのマスク色から導出する（CSSに数値を書かない）。
  document.documentElement.style.setProperty("--nan", nanColorCss());

  wireDisplayControls();
  wireLoaders();
  try {
    const specs = await listViews();
    if (!specs.length) { status("ビューが登録されていません", true); return; }
    state.specs = specs;
    state.spec = specs[0];
    buildViewSelect(specs);
    buildServerParams(state.spec);
    buildGrid(state.layout);
    status(`準備完了 — ビュー "${state.spec.label}" / パスを入れて読み込み`);
  } catch (e) {
    status(`起動に失敗: ${errorText(e)}`, true);
  }
}

main();
