// Application wiring for the matrix viewer.
//
// Two shapes of view are drawn: a scalar one, one tile per channel (the 12
// coefficients), and an rgb one, three channels on a single colour tile (the
// projection surface). `kind` from the registry decides which; see layoutFor.
//
// The load-bearing rule here is the separation between *display* and
// *recompute*. Exposure, gamma, colormap and dither only push uniforms into the
// GLViews; they must never touch the network. Only a server-side view
// parameter (`normalize` for the coefficient tiles, the projection command for
// the surface) re-issues the binary fetch.
// 露出・γ・カラーマップ・ディザは uniform を更新するだけでサーバへ行かない。
// サーバ側ビューパラメータの変更だけが再フェッチを起こす。
//
// The fetched Float32Array stays on the CPU for the whole session so pixel
// probing reads it locally. The body is planar (C, H, W) -- channel c of pixel
// (row, col) is data[c * width * height + row * width + col] -- with the older
// interleaved (H, W, C) still handled wherever `payload.layout` says so.
// 取得した Float32Array は保持し続け、画素読み取りはローカルで済ませる。
// 本体は planar (C, H, W)。従来のインタリーブも layout を見て扱う。

import { loadMatrix, listViews, fetchView } from "./api.js";
import { GLView, NAN_COLOR, nanColorCss } from "./glview.js";

const $ = (id) => document.getElementById(id);

// The views this frontend can draw. `categorical` is declared in the Python
// ViewKind but has no implementation here; see requireSupportedKind.
// 描けるのはこの2つ。categorical は未実装。
const SUPPORTED_KINDS = ["scalar", "rgb"];

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
// caption that describes a matrix it is not.
const MATRIX_CORNER = "in \\ out";

// rows x cols is the *tile* grid; probeCols is the readout grid, which for an
// rgb view is not the same thing. See layoutFor.
const DEFAULT_LAYOUT = {
  rows: DEFAULT_ROWS,
  cols: DEFAULT_COLS,
  probeCols: DEFAULT_COLS,
  corner: MATRIX_CORNER,
};

// Gamma bounds for the typed input. The shader guards with max(u_gamma, 1e-6),
// so the floor here is about keeping the control usable rather than safe.
// 数値入力で受け付けるγの範囲。シェーダ側は max(u_gamma, 1e-6) で守られているので、
// この下限は安全性ではなく操作性のためのもの。
const GAMMA_MIN = 0.01;
const GAMMA_MAX = 10;

const state = {
  matrixId: null,
  meta: null,
  specs: [],        // every ViewSpec from /api/views
  spec: null,       // the active one
  payload: null,    // { data, width, height, channels, layout, vmin, vmax, labels }
  ranges: null,     // [{ vmin, vmax }] per *tile* (per channel except for rgb)
  layout: DEFAULT_LAYOUT,
  pinned: null,     // { col, row } while the readout is frozen
  // Monotonic request id. Fetches are not cancelled, they are *ignored*: a
  // 95 MB transfer takes seconds, and a second 読み込み during the first one
  // used to leave the header describing matrix B while state.payload -- which
  // backs all 12 tiles and the hover readout -- still held matrix A. Two
  // clicks, no warning, coefficients read off the wrong matrix. Every async
  // entry point bumps this, captures it, and drops its result after any await
  // where it no longer matches.
  // 連続フェッチの順序保証。最新のリクエストIDと一致しない結果は捨てる。
  reqId: 0,
};

// Uniform-only display parameters. Changing any of these must not fetch.
const display = {
  exposure: 0, gamma: 1, colormap: 0, dither: 1, nanColor: NAN_COLOR,
};

const views = [];       // one GLView per channel
const captions = [];    // per-tile caption elements
const probeCells = [];  // per-channel readout elements
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
  const planar = layout === "planar";
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
// The readout grid is deliberately *not* keyed the same way: a hovered pixel
// still has three numbers, so it keeps one cell per channel and the two grids
// differ in exactly that one place.
//
// タイル配置は kind で分岐させる。scalar はチャンネル数だけタイルを並べるが、
// rgb の3チャンネルは1枚の絵であり、3枚のグレースケールにしては意味が無い。
// ラベル ["R","G","B"] は `axesFromLabels` では解釈できず、フォールバックだと
// まさにその1行3列になってしまう（しかも黙って描けてしまう）。ペイロードに
// 絵が何枚あるかを知っているのは kind だけなので、それで決める。
// 読み取りグリッドだけは別で、ホバー1画素につき数値は3つあるためチャンネル数
// で組む。両者が食い違うのはここだけ。
function layoutFor(payload, current) {
  // `channels === 3` as well as the kind: an rgb view is one colour per pixel,
  // so three channels is what "rgb" means here. requireRgbHasThreeChannels has
  // already refused anything else by the time this runs -- this condition is
  // defence in depth, not the active guard -- but a single-tile layout is only
  // correct for a payload the rgb upload path can actually pack, and the two
  // must not be able to disagree.
  // kind だけでなく channels も見る。rgb は1画素1色＝3チャンネルのこと。
  // 実際に弾いているのは requireRgbHasThreeChannels 側で、ここは多重防御だが、
  // 1枚タイルが正しいのは rgb 経路が詰められるペイロードに限られる。
  if (payload.kind === "rgb" && payload.channels === 3) {
    return {
      rows: [""],
      cols: ["RGB"],
      probeCols: fallbackAxes(payload.labels, payload.channels).cols,
      corner: "",
    };
  }
  const axes = axesFromLabels(payload.labels, payload.channels)
    || (current.rows.length * current.cols.length === payload.channels
      ? { rows: current.rows, cols: current.cols }
      : fallbackAxes(payload.labels, payload.channels));
  return {
    rows: axes.rows, cols: axes.cols, probeCols: axes.cols, corner: MATRIX_CORNER,
  };
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
  buildProbeGrid(rows, layout.probeCols);
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

function buildProbeGrid(rows, cols) {
  const grid = $("probe-grid");
  grid.textContent = "";
  grid.style.gridTemplateColumns = `var(--rowhead) repeat(${cols.length}, minmax(0, 1fr))`;
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

// Read all channels for one pixel straight out of the cached array. No fetch.
// キャッシュ済み配列から全チャンネルを読む。フェッチはしない。
function showProbe(hit) {
  const p = state.payload;
  if (!p || !hit) {
    $("probe-pos").textContent = "—";
    for (const cell of probeCells) { cell.textContent = "—"; cell.className = "probe-cell"; }
    return;
  }
  $("probe-pos").textContent = `col ${hit.col}, row ${hit.row}`;
  // Planar (C, H, W) puts the channel index outermost; interleaved (H, W, C)
  // puts it innermost. Getting this backwards misreports every hovered value
  // while the tiles themselves still look right -- nothing would flag it.
  // planar は channel が最外、インタリーブは最内。取り違えても絵は正しいまま
  // 読み取り値だけが狂うので、どこも失敗しない。
  const planar = p.layout === "planar";
  const stride = p.width * p.height;
  const offset = hit.row * p.width + hit.col;
  for (let c = 0; c < probeCells.length && c < p.channels; c++) {
    const v = planar
      ? p.data[c * stride + offset]
      : p.data[offset * p.channels + c];
    const cell = probeCells[c];
    cell.textContent = fmt(v);
    cell.className = Number.isFinite(v) ? "probe-cell" : "probe-cell nan";
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
    // Server-side parameters are the only controls allowed to hit the network.
    el.addEventListener("change", () => { reload().catch((e) => status(String(e), true)); });
    host.appendChild(el);
  }
}

// ------------------------------------------------------------- rendering

function applyDisplay() {
  for (let c = 0; c < views.length; c++) {
    const r = state.ranges ? state.ranges[c] : { vmin: -1, vmax: 1 };
    views[c].setParams({ ...display, vmin: r.vmin, vmax: r.vmax });
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

function computeRanges() {
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
  if (p.kind === "rgb") {
    state.ranges = [{ vmin: p.vmin, vmax: p.vmax }];
    const text = `[${fmt(p.vmin)}, ${fmt(p.vmax)}]`;
    if (captions.length) captions[0].textContent = `${p.labels.join(" ")}  ${text}`;
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

// Falling through for an unimplemented kind would paint `categorical` data
// with the diverging scalar colormap: plausible pixels for a view this
// frontend does not understand, which is the same class of silent misreading
// the NaN mask exists to prevent.
// 未対応の kind をそのまま描くと、理解していないデータを「それらしい」色で
// 表示してしまう。黙って通さず、はっきり失敗させる。
function requireSupportedKind(kind, where) {
  if (!SUPPORTED_KINDS.includes(kind)) {
    throw new Error(
      `${where} の kind="${kind}" はこのビューアでは未対応です` +
      `（対応しているのは kind="${SUPPORTED_KINDS.join('" / kind="')}" のみ）。`
    );
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
  // Checked from the registry first, so a view that cannot be drawn is refused
  // before the transfer rather than after it.
  // まず登録内容で判定し、描けないビューは転送前に断る。
  const specName = `ビュー "${state.spec.label || state.spec.key}"`;
  requireSupportedKind(state.spec.kind, specName);
  requireRgbHasThreeChannels(state.spec.kind, state.spec.channels, specName);
  const params = serverParams();
  status(`ビュー取得中… (${state.spec.key} ${JSON.stringify(params)})`);

  const t0 = performance.now();
  const payload = await fetchView(state.matrixId, state.spec.key, params);
  const t1 = performance.now();
  if (isStale(req)) return;   // a newer request owns the screen now
  // Backstop: the registry and the response could disagree.
  requireSupportedKind(payload.kind, "サーバの応答");
  requireRgbHasThreeChannels(payload.kind, payload.channels, "サーバの応答");
  state.payload = payload;

  const layout = layoutFor(payload, state.layout);
  // Compared by value, and only then rebuilt: buildGrid disposes and recreates
  // one WebGL context per tile, and the browser caps how many can be live at
  // once (~16 in Chrome, against 12 tiles).
  // 値で比較してから組み直す。タイル1枚につきWebGLコンテキスト1つを捨てて
  // 作り直すため、ブラウザの上限（Chromeで約16、タイルは12枚）に近い。
  const layoutChanged = JSON.stringify(layout) !== JSON.stringify(state.layout);
  state.layout = layout;
  // Tiles, not channels: an rgb view puts three channels on one tile.
  const tiles = layout.rows.length * layout.cols.length;
  const gridRebuilt = views.length !== tiles || layoutChanged;
  if (gridRebuilt) {
    buildGrid(state.layout);
    await nextFrame();
    if (isStale(req)) return;
  }

  const t2 = performance.now();
  computeRanges();
  const t3 = performance.now();

  // The colormap gives a *number* a colour; rgb data is already a colour and
  // the shader's rgb branch has no colormap in it at all. Disable the control
  // rather than leave a dropdown that silently does nothing.
  // カラーマップは数値に色を与える仕組みで、rgb経路はそもそも通さない。
  // 何も起きないドロップダウンを残さず無効化する。
  $("cmap-select").disabled = payload.kind === "rgb";

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
    // channels, and `mode` is what tells GLView which of those it is.
    // チャンネルではなくタイルを回す。scalar では同数だが、rgb は3チャンネルを
    // 1枚が受け取る。どちらかは `mode` で伝える。
    for (let t = 0; t < views.length; t++) {
      views[t].setImage({
        data: payload.data,
        width: payload.width,
        height: payload.height,
        channels: payload.channels,
        channel: t,
        layout: payload.layout,
        mode: payload.kind === "rgb" ? "rgb" : "scalar",
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
  $("perf-upload").textContent = `${perf.uploadMs} ms + finish ${perf.uploadFinishMs} ms`;
  $("perf-heap").textContent = perf.jsHeapMB ? `${perf.jsHeapMB} MB` : "計測不可";
  status(
    `${payload.width}×${payload.height}×${payload.channels} を表示中 — ` +
    `転送 ${perf.fetchMs} ms / レンジ ${perf.rangesMs} ms / ` +
    `GPU転送 ${perf.uploadMs}+${perf.uploadFinishMs} ms`
  );
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
  window.__viewer = { state, views, display, applyDisplay, reload };

  // Derive the legend swatch from the shader's own mask colour rather than
  // restating it in CSS. See NAN_COLOR in glview.js.
  // 凡例の色はシェーダのマスク色から導出する（CSSに数値を書かない）。
  document.documentElement.style.setProperty("--nan", nanColorCss());

  wireDisplayControls();
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
