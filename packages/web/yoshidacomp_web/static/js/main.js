// Application wiring for the 12-coefficient tile viewer.
//
// The load-bearing rule here is the separation between *display* and
// *recompute*. Exposure, gamma, colormap and dither only push uniforms into the
// 12 GLViews; they must never touch the network. Only a server-side view
// parameter (currently `normalize`) re-issues the binary fetch.
// 露出・γ・カラーマップ・ディザは uniform を更新するだけでサーバへ行かない。
// サーバ側ビューパラメータの変更だけが再フェッチを起こす。
//
// The fetched Float32Array stays on the CPU for the whole session so pixel
// probing reads data[(row * width + col) * channels + c] locally.
// 取得した Float32Array は保持し続け、画素読み取りはローカルで済ませる。

import { loadMatrix, listViews, fetchView } from "./api.js";
import { GLView, NAN_COLOR, nanColorCss } from "./glview.js";

const $ = (id) => document.getElementById(id);

// The only view B1 can draw. `rgb` and `categorical` are declared in the
// Python ViewKind but have no implementation here; see requireSupportedKind.
// B1が描けるのはこれだけ。rgb / categorical は未実装。
const SUPPORTED_KIND = "scalar";

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

const state = {
  matrixId: null,
  meta: null,
  specs: [],        // every ViewSpec from /api/views
  spec: null,       // the active one
  payload: null,    // { data, width, height, channels, vmin, vmax, labels }
  ranges: null,     // [{ vmin, vmax }] per channel
  rows: DEFAULT_ROWS,
  cols: DEFAULT_COLS,
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
function percentileAbsPerChannel(data, channels, pct) {
  const BINS = 2048;
  const maxAbs = new Float64Array(channels);
  const counts = new Float64Array(channels);
  for (let p = 0, n = data.length; p < n; p += channels) {
    for (let c = 0; c < channels; c++) {
      const v = data[p + c];
      if (!Number.isFinite(v)) continue;
      const a = v < 0 ? -v : v;
      if (a > maxAbs[c]) maxAbs[c] = a;
      counts[c]++;
    }
  }
  const hist = new Int32Array(channels * BINS);
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

function buildGrid(rows, cols) {
  for (const v of views) v.dispose();
  views.length = 0;
  captions.length = 0;

  const grid = $("tile-grid");
  grid.textContent = "";
  grid.style.gridTemplateColumns = `var(--rowhead) repeat(${cols.length}, minmax(0, 1fr))`;
  grid.style.gridTemplateRows = `auto repeat(${rows.length}, minmax(0, 1fr))`;

  const corner = document.createElement("div");
  corner.className = "corner mono";
  corner.textContent = "in \\ out";
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
  buildProbeGrid(rows, cols);
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
  const base = (hit.row * p.width + hit.col) * p.channels;
  for (let c = 0; c < probeCells.length && c < p.channels; c++) {
    const v = p.data[base + c];
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
// `normalize` belongs to coef_tiles, not to views in general. Reading it
// unconditionally meant every future view inherited another view's control
// name, which is exactly the kind of leak that breaks the "register in Python,
// no JavaScript change" promise.
// `normalize` は coef_tiles 固有のパラメータであり、全ビュー共通ではない。
// 有効なビューが宣言している場合だけ見る。
function specDeclares(name) {
  return !!state.spec && state.spec.params.some((p) => p.name === name);
}

function computeRanges() {
  const p = state.payload;
  const mode = specDeclares("normalize") ? serverParams().normalize : "shared";
  if (mode === "per_tile") {
    // The wire format carries a single (vmin, vmax) pair, so 12 independent
    // ranges have to be derived here from the array we already hold.
    // ヘッダは1組のレンジしか運べないので、タイルごとのレンジは手元で出す。
    const abs = percentileAbsPerChannel(p.data, p.channels, 99);
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

// B1 renders scalar views only. Falling through for an unimplemented kind
// would paint `rgb` or `categorical` data with the diverging scalar colormap:
// plausible pixels for a view this frontend does not understand, which is the
// same class of silent misreading the NaN mask exists to prevent.
// 未対応の kind をそのまま描くと、理解していないデータを「それらしい」色で
// 表示してしまう。黙って通さず、はっきり失敗させる。
function requireSupportedKind(kind, where) {
  if (kind !== SUPPORTED_KIND) {
    throw new Error(
      `${where} の kind="${kind}" はこのビューアでは未対応です` +
      `（対応しているのは kind="${SUPPORTED_KIND}" のみ）。`
    );
  }
}

// `req` is inherited when load() drives the reload, so the whole click stays
// one logical request; a bare reload() (a server-parameter change) claims a
// new one.
async function reload(req = beginRequest()) {
  if (!state.matrixId || !state.spec) return;
  requireSupportedKind(state.spec.kind, `ビュー "${state.spec.label || state.spec.key}"`);
  const params = serverParams();
  status(`ビュー取得中… (${state.spec.key} ${JSON.stringify(params)})`);

  const t0 = performance.now();
  const payload = await fetchView(state.matrixId, state.spec.key, params);
  const t1 = performance.now();
  if (isStale(req)) return;   // a newer request owns the screen now
  // Backstop: the registry and the response could disagree.
  requireSupportedKind(payload.kind, "サーバの応答");
  state.payload = payload;

  const axes = axesFromLabels(payload.labels, payload.channels)
    || (state.rows.length * state.cols.length === payload.channels
      ? { rows: state.rows, cols: state.cols }
      : fallbackAxes(payload.labels, payload.channels));
  const axesChanged =
    axes.rows.join(" ") !== state.rows.join(" ") ||
    axes.cols.join(" ") !== state.cols.join(" ");
  state.rows = axes.rows;
  state.cols = axes.cols;
  const gridRebuilt = views.length !== payload.channels || axesChanged;
  if (gridRebuilt) {
    buildGrid(state.rows, state.cols);
    await nextFrame();
    if (isStale(req)) return;
  }

  const t2 = performance.now();
  computeRanges();
  const t3 = performance.now();

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
    for (let c = 0; c < payload.channels; c++) {
      views[c].setImage({
        data: payload.data,
        width: payload.width,
        height: payload.height,
        channels: payload.channels,
        channel: c,
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
  $("gamma-slider").addEventListener("input", (e) => {
    display.gamma = Number(e.target.value);
    $("gamma-val").textContent = display.gamma.toFixed(2);
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
    buildGrid(state.rows, state.cols);
    status(`準備完了 — ビュー "${state.spec.label}" / パスを入れて読み込み`);
  } catch (e) {
    status(`起動に失敗: ${errorText(e)}`, true);
  }
}

main();
