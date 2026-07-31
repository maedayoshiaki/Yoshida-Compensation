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
import { GLView } from "./glview.js";

const $ = (id) => document.getElementById(id);

const DEFAULT_ROWS = ["in_R", "in_G", "in_B", "bias"];
const DEFAULT_COLS = ["out_R", "out_G", "out_B"];

const state = {
  matrixId: null,
  meta: null,
  spec: null,       // the ViewSpec from /api/views
  payload: null,    // { data, width, height, channels, vmin, vmax, labels }
  ranges: null,     // [{ vmin, vmax }] per channel
  rows: DEFAULT_ROWS,
  cols: DEFAULT_COLS,
  pinned: null,     // { col, row } while the readout is frozen
};

// Uniform-only display parameters. Changing any of these must not fetch.
const display = {
  exposure: 0, gamma: 1, colormap: 0, dither: 1, nanColor: [0.85, 0.1, 0.75],
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

function computeRanges() {
  const p = state.payload;
  const mode = serverParams().normalize || "shared";
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

async function reload() {
  if (!state.matrixId || !state.spec) return;
  const params = serverParams();
  status(`ビュー取得中… (${state.spec.key} ${JSON.stringify(params)})`);

  const t0 = performance.now();
  const payload = await fetchView(state.matrixId, state.spec.key, params);
  const t1 = performance.now();
  state.payload = payload;

  const axes = axesFromLabels(payload.labels, payload.channels)
    || (state.rows.length * state.cols.length === payload.channels
      ? { rows: state.rows, cols: state.cols }
      : fallbackAxes(payload.labels, payload.channels));
  const axesChanged =
    axes.rows.join(" ") !== state.rows.join(" ") ||
    axes.cols.join(" ") !== state.cols.join(" ");
  state.rows = axes.rows;
  state.cols = axes.cols;
  if (views.length !== payload.channels || axesChanged) {
    buildGrid(state.rows, state.cols);
    await nextFrame();
  }

  const t2 = performance.now();
  computeRanges();
  const t3 = performance.now();

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
  views[0].fit();                      // one fit, propagated to the other tiles
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

async function load() {
  const path = $("path-input").value.trim();
  if (!path) { status("パスを入力してください", true); return; }
  status(`行列を読み込み中: ${path}`);
  const res = await loadMatrix(path);
  state.matrixId = res.id;
  state.meta = res.meta;
  renderMeta();
  await reload();
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
    load().catch((e) => status(String(e && e.message ? e.message : e), true));
  });
  $("path-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btn-load").click();
  });
}

async function main() {
  // Debug/automation handle. The smoke test and any CDP-driven measurement need
  // to reach the views without going through synthetic input events.
  // スモークテストや自動計測から views へ触るためのハンドル。
  window.__viewer = { state, views, display, applyDisplay, reload };

  wireDisplayControls();
  try {
    const specs = await listViews();
    if (!specs.length) { status("ビューが登録されていません", true); return; }
    state.spec = specs[0];
    buildServerParams(state.spec);
    buildGrid(state.rows, state.cols);
    status(`準備完了 — ビュー "${state.spec.label}" / パスを入れて読み込み`);
  } catch (e) {
    status(`起動に失敗: ${e && e.message ? e.message : e}`, true);
  }
}

main();
