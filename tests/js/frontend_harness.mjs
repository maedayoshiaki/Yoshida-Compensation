// Drives the shipped viewer frontend under a stub DOM/WebGL, for pytest.
//
// Usage: node frontend_harness.mjs <static-dir> <scenario>
//   <static-dir>  directory holding index.html and js/{main,glview,api}.js
//   <scenario>    "ordering" | "views" | "button" | "rgb"
// Prints one JSON object to stdout: { scenario, checks: [{name, ok, detail}] }.
//
// Why a stub instead of a browser: the behaviours under test -- request
// ordering, view selection, button state -- are all plain JavaScript control
// flow. A headless browser would test the same lines through three more layers
// and could not run inside `pytest -q`, which has to stay the whole gate.
//
// What the stub deliberately does *not* fake: the modules themselves. It
// imports main.js from <static-dir> as shipped, so this can never become a
// mirror that drifts from the code it claims to cover -- the mistake the
// golden shader test was written to stop making. Pointing <static-dir> at a
// mutated copy is how tests/test_web_frontend_js.py proves these checks can
// fail.
//
// ブラウザを使わずスタブで駆動する。検証対象（リクエスト順序・ビュー選択・
// ボタン状態）はすべて素のJS制御フローであり、pytest 一発を唯一のゲートに
// 保つ必要があるため。スタブにしないのはモジュール本体で、main.js は
// <static-dir> から出荷物そのものを import する。写しを作らないことが要点。

import { readFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [, , STATIC_DIR, SCENARIO] = process.argv;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ results
const checks = [];
const check = (name, ok, detail = "") => checks.push({ name, ok: !!ok, detail: String(detail) });

function emit() {
  process.stdout.write(JSON.stringify({ scenario: SCENARIO, checks }, null, 2));
}

// --------------------------------------------------------------- WebGL stub
// Any property is either a GL constant (SCREAMING_CASE -> a number) or a
// callable returning another such proxy, so every gl.* call in glview.js
// succeeds and every returned handle is usable.
//
// One call is not swallowed: `texImage2D`, whose last argument is the plane
// GLView.setImage extracted. That argument is the only externally visible
// output of the layout handling -- which channel ended up on which tile -- and
// without recording it the two branches of setImage (planar subarray vs
// interleaved strided walk) are indistinguishable from outside, so either
// could break with nothing noticing. Recorded as a summary rather than the
// array itself so a `subarray` view cannot pin the whole payload alive.
// `texImage2D` だけは飲み込まず、最後の引数（setImage が切り出した平面）を
// 記録する。どのチャンネルがどのタイルに載ったかは外から見える唯一の出力で
// あり、これを取らないと planar とインタリーブの2分岐は外部から区別できず、
// どちらが壊れても誰も気付かない。`subarray` がペイロード全体を保持しない
// よう、配列そのものではなく要約を記録する。
//
// `head` is what makes the rgb pack observable. Length, first and last alone
// cannot tell a correctly interleaved (R,G,B,R,G,B,...) upload from the planar
// buffer handed over unpacked: both are 3*H*W long, both start at channel 0's
// value and end at channel 2's. The first six elements separate them
// immediately -- [1,2,3,1,2,3] against [1,1,1,1,1,1].
// `head` があってはじめて RGB の詰め替えが外から見える。長さ・先頭・末尾だけ
// では、正しくインタリーブした配列と planar のまま渡した配列を区別できない。
//
// `uniform1i` is the other call not swallowed, for the same reason one step
// later: the texture can be uploaded as RGB32F and the shader still told
// u_mode = 0, which reads only its red component and draws a greyscale tile
// that looks like a working view of a colour. Uniform locations are handed back
// as their own names so the value is attributable to a uniform from out here.
// `uniform1i` も飲み込まない。RGB32F を載せておきながら u_mode を0のままに
// すると、赤成分だけを読んだグレースケールが「動いているビュー」の顔で出る。
// 値をどの uniform のものか判別できるよう、ロケーションは名前そのものを返す。
//
// Both maps are threaded through *every* nested proxy, including the ones the
// fallback and `apply` hand back. They used to be dropped there, so anything
// reached one level deeper -- `gl.getExtension(...).uniform1i(...)`, say --
// carried `uniforms === undefined` and would have thrown on assignment instead
// of recording. Nothing in glview.js does that today; the point is that the
// stub must not be the reason a future call is untestable.
// どちらの記録先も、フォールバックと `apply` が返す入れ子のプロキシまで
// 引き回す。以前はここで落ちており、1段深い呼び出しは `uniforms` が
// undefined のまま——記録ではなく例外になる状態だった。現時点の glview.js に
// そうした呼び出しは無いが、将来の呼び出しが検証できない理由がスタブ側に
// あってはならない。
function glStub(uploads, uniforms) {
  const fn = function () { return glStub(uploads, uniforms); };
  return new Proxy(fn, {
    get(_t, prop) {
      if (prop === Symbol.toPrimitive || prop === "valueOf") return () => 1;
      if (prop === "__uploads") return uploads;
      if (prop === "__uniforms") return uniforms;
      if (prop === "getUniformLocation") return (_prog, name) => name;
      if (prop === "uniform1i") return (name, value) => { uniforms[name] = value; };
      if (prop === "texImage2D") {
        return (...args) => {
          const px = args[args.length - 1];
          uploads.push({
            length: px.length,
            first: px[0],
            last: px[px.length - 1],
            head: Array.from(px.slice(0, 6)),
          });
        };
      }
      if (typeof prop === "string" && /^[A-Z0-9_]+$/.test(prop)) return 1;
      return glStub(uploads, uniforms);
    },
    apply() { return glStub(uploads, uniforms); },
  });
}

// ------------------------------------------------------------------ DOM stub
const registry = new Map();

function makeEl(tag = "div") {
  const listeners = {};
  const el = {
    tagName: tag.toUpperCase(),
    _id: "", _text: "",
    innerHTML: "", value: "", checked: false, disabled: false,
    className: "", hidden: false, type: "", step: "",
    width: 0, height: 0, clientWidth: 480, clientHeight: 240,
    children: [],
    style: { setProperty(k, v) { el.style[k] = v; } },
    classList: { toggle() {}, add() {}, remove() {} },
    appendChild(child) { el.children.push(child); return child; },
    addEventListener(name, fn) { (listeners[name] ||= []).push(fn); },
    removeEventListener() {},
    // Test-side: fire a listener the way a user interaction would.
    __fire(name, event = {}) {
      for (const fn of listeners[name] || []) fn({ target: el, ...event });
    },
    // One context per element, kept, so each tile's uploads accumulate on its
    // own GLView rather than on a fresh proxy per call.
    // 要素ごとに1つのコンテキストを保持し、タイル別に記録が貯まるようにする。
    getContext() { return (el.__gl ||= glStub([], {})); },
    getBoundingClientRect() { return { left: 0, top: 0 }; },
    setPointerCapture() {},
  };
  // Assigning an id registers the element, so dynamically built controls
  // (buildServerParams' `sp-*`) are findable exactly as in a real document.
  Object.defineProperty(el, "id", {
    get: () => el._id,
    set: (v) => { el._id = v; if (v) registry.set(v, el); },
  });
  // Setting textContent replaces all children, as the DOM does -- this is what
  // makes buildServerParams' `host.textContent = ""` actually unregister the
  // previous view's controls instead of leaving them findable.
  Object.defineProperty(el, "textContent", {
    get: () => el._text,
    set: (v) => {
      for (const child of el.children) unregister(child);
      el.children.length = 0;
      el._text = v;
    },
  });
  return el;
}

function unregister(el) {
  if (el._id && registry.get(el._id) === el) registry.delete(el._id);
  for (const child of el.children) unregister(child);
}

// Pre-register exactly the ids that index.html declares, and nothing else, so
// a reference to an id the page does not have throws rather than silently
// finding a blank stub.
// index.html が宣言しているidだけを事前登録する。存在しないidの参照は
// 黙って空要素を拾わず例外になる。
const html = readFileSync(path.join(STATIC_DIR, "index.html"), "utf-8");
for (const m of html.matchAll(/id="([^"]+)"/g)) registry.set(m[1], makeEl());

globalThis.document = {
  documentElement: makeEl("html"),
  getElementById: (id) => registry.get(id) ?? null,
  createElement: (tag) => makeEl(tag),
};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.performance = { now: () => Date.now() };
globalThis.window = {
  devicePixelRatio: 1,
  matchMedia: () => ({ addEventListener() {}, removeEventListener() {} }),
  addEventListener() {}, removeEventListener() {},
  requestAnimationFrame: globalThis.requestAnimationFrame,
};

const el = (id) => registry.get(id) ?? null;

// main.js logs a [perf] object per load and console.error()s on status errors.
// Buffer both so stdout carries only the JSON result.
const consoleLines = [];
console.log = (...a) => consoleLines.push(a.map((x) => String(x)).join(" "));
console.error = (...a) => consoleLines.push("ERR " + a.map((x) => String(x)).join(" "));

// -------------------------------------------------------------- fake server
const H = 4, W = 5;
const TILE_LABELS = ["in_R", "in_G", "in_B", "bias"].flatMap(
  (i) => ["out_R", "out_G", "out_B"].map((o) => `${i} -> ${o}`));

const VIEWS = [
  {
    key: "coef_tiles", label: "12係数タイル", kind: "scalar", channels: 12,
    params: [{
      name: "normalize", label: "正規化", type: "select",
      default: "shared", choices: ["shared", "per_tile"],
    }],
  },
  // A second registration, to check it is reachable at all (F4).
  { key: "plain_tiles", label: "パラメータなしタイル", kind: "scalar", channels: 12, params: [] },
  // Interleaved *and* carrying `normalize`, which is the only combination that
  // reaches percentileAbsPerChannel's interleaved branch. Nothing was both:
  // `coef_tiles` has the parameter but is planar, `plain_tiles` is interleaved
  // but deliberately has no parameters -- it is the fixture that proves a view
  // which does not declare `normalize` is not handed one, so it cannot be given
  // one here. Hence a fourth registration rather than a change to either.
  // Without it that branch has no execution anywhere and could be replaced by
  // the planar walk with nothing failing (verified).
  // インタリーブかつ `normalize` を持つ唯一の組み合わせ。これが
  // percentileAbsPerChannel のインタリーブ側分岐への唯一の入口である。
  // `coef_tiles` は planar、`plain_tiles` は「パラメータを宣言しないビューに
  // `normalize` を渡さない」ことを示すためのフィクスチャなので付けられない。
  // 無いとこの分岐はどこからも実行されず、planar 側の走査に置き換えても
  // 何も落ちない（実際に確認済み）。
  {
    key: "legacy_tiles", label: "旧形式タイル", kind: "scalar", channels: 12,
    params: [{
      name: "normalize", label: "正規化", type: "select",
      default: "shared", choices: ["shared", "per_tile"],
    }],
  },
  // Declared in the Python ViewKind but unimplemented in the frontend (F4).
  { key: "gamut", label: "色域外判定", kind: "categorical", channels: 4, params: [] },
  // Three channels that are one colour, not three pictures -- the rgb scenario.
  {
    key: "surface", label: "投影面の色", kind: "rgb", channels: 3,
    params: [
      { name: "p_r", label: "投影 R", type: "number", default: 1.0, choices: [] },
      { name: "p_g", label: "投影 G", type: "number", default: 1.0, choices: [] },
      { name: "p_b", label: "投影 B", type: "number", default: 1.0, choices: [] },
    ],
  },
  // Three channels that are three unrelated pictures. This differs from
  // `surface` in `kind` and in nothing else, which is the whole reason it is
  // here: layoutFor decides the tile count from `kind` *and* `channels`, and
  // with only `surface` registered a layout keyed on the channel count alone
  // would be indistinguishable from the correct one. Then any 3-channel scalar
  // view collapses onto a single colour tile -- two of its three pictures
  // simply gone, with the remaining one drawn as a colour it is not.
  // `surface` と kind だけが違うビュー。タイル数は kind と channels の両方で
  // 決まるが、`surface` しか無いとチャンネル数だけを見る実装と区別が付かない。
  // その状態では3チャンネルのスカラービューが1枚の色タイルに潰れ、3枚のうち
  // 2枚が消える。
  { key: "tri_scalar", label: "3chスカラー", kind: "scalar", channels: 3, params: [] },
  // kind="rgb" with a channel count "rgb" cannot mean. Registered so the guard
  // that refuses it has something to refuse: without this the channel-count
  // half of the rgb contract is asserted nowhere, and dropping it lets a short
  // payload into packRgb, which reads past the end of the array and paints the
  // whole tile in the NaN mask colour -- a loud failure with no diagnosis in it.
  // rgb を名乗る非3チャンネルのビュー。これが無いとチャンネル数の検査は
  // どこからも検証されず、外した場合は packRgb が終端を越えて読み、タイル全面
  // がマスク色になる（派手に壊れるが原因が分からない）。
  { key: "bad_rgb", label: "4chのrgb", kind: "rgb", channels: 4, params: [] },
  // Served with `X-Layout: "Planar"` -- present, capitalised, and therefore in
  // neither of the two layouts the client knows. It has to be refused loudly.
  // Without the guard nothing throws: every `=== "planar"` test fails, all four
  // consumers fall to the interleaved walk together, and the tiles, the hover
  // readout and the per-tile ranges then agree with each other about the wrong
  // channels. Three independent readings corroborating one mistake is the most
  // convincing wrong answer this viewer can produce.
  // `X-Layout: "Planar"`（存在するが大文字違いで既知のどちらでもない）で返す。
  // ガードが無いと例外は出ず、`=== "planar"` が全て外れて4つの読み手が一斉に
  // インタリーブ側へ落ちる。タイル・読み取り値・タイル別レンジが「同じ
  // 間違い方」で一致するため、このビューアが出しうる最も説得力のある誤答になる。
  { key: "mangled_layout", label: "配置名が壊れたタイル", kind: "scalar", channels: 12, params: [] },
  // One channel, scalar. Exists for layoutFor's "keep the current axes"
  // fallback: an rgb layout is 1 row x 1 col, so the channel count alone
  // (`1 * 1 === 1`) is enough to inherit it, and this view arriving straight
  // after `surface` would caption its single greyscale tile "RGB". The
  // corner marker is what distinguishes axes that came from a matrix.
  // 1チャンネルの scalar。layoutFor の「現在の軸を引き継ぐ」フォールバック
  // のために置く。rgb の配置は1行1列なので枚数だけなら一致してしまい、
  // `surface` の直後にこれが来ると1枚のグレースケールに「RGB」という見出しが
  // 付く。行列由来の軸かどうかを区別しているのが corner である。
  { key: "mono_scalar", label: "1chスカラー", kind: "scalar", channels: 1, params: [] },
];

// Keyed by view rather than by channel count: `surface` has 3 channels and
// would otherwise be handed the 4 categorical labels, which the backend's own
// channels/labels agreement check would never allow.
// チャンネル数ではなくキーで引く。3チャンネルの `surface` に4つのラベルを
// 渡さないため。
const LABELS_BY_KEY = {
  coef_tiles: TILE_LABELS,
  plain_tiles: TILE_LABELS,
  legacy_tiles: TILE_LABELS,
  gamut: ["色域外", "1超クリップ", "両方", "範囲内, 補正可"],
  surface: ["R", "G", "B"],
  // Deliberately *not* ["R","G","B"]: `tri_scalar` must reach its 1x3 grid
  // through fallbackAxes on its own merits, not by borrowing `surface`'s.
  tri_scalar: ["ch0", "ch1", "ch2"],
  // Only reached if the channel-count guard is removed; served so that the
  // mutation's failure is "it rendered", not "the fake server threw".
  bad_rgb: ["R", "G", "B", "A"],
  mangled_layout: TILE_LABELS,
  // Deliberately not "a -> b": `mono_scalar` must reach a 1x1 grid through
  // fallbackAxes, which is the path the corner gate protects.
  // 意図的に "a -> b" 形式にしない。fallbackAxes 経由で1行1列に至る経路が
  // corner のガードが守っているものだからである。
  mono_scalar: ["only"],
};

const viewLabels = (spec) => LABELS_BY_KEY[spec.key];

// `plain_tiles` and `legacy_tiles` are served the *old* way: interleaved
// (H, W, C) and with no X-Layout header at all.
//
// The shipped client keeps an interleaved path on purpose -- api.js defaults to
// "interleaved" so an endpoint that has not been converted still renders. Once
// every fixture became planar that path had no test execution anywhere and was
// free to rot. Serving one view the old way exercises it, and exercises
// api.js's default at the same time: nothing else here omits the header.
// `plain_tiles` だけは旧来通りインタリーブ (H, W, C) で、X-Layout も付けずに
// 返す。出荷クライアントは未変換のエンドポイントのために意図的にインタリーブ
// 経路を残しているが、全フィクスチャを planar にした時点でその経路はどこから
// も実行されなくなった。1つだけ旧形式で返して経路と api.js の既定値の両方を
// 実際に通す。
const INTERLEAVED_VIEWS = new Set(["plain_tiles", "legacy_tiles"]);

// Per-matrix delays, so responses can be made to resolve in a different order
// from the one they were issued in. `session` delays POST /session/matrix
// (which is what load() awaits) and `view` delays the binary view fetch (which
// is what reload() awaits) -- the two awaits have separate guards, so each
// needs to be reachable on its own.
// load() と reload() は別々の await を持ち、それぞれに別のガードがあるため、
// POSTとバイナリ取得のどちらも独立に遅延できるようにする。
const sessionDelayMs = {};
const viewDelayMs = {};
// Identifies which matrix a payload came from: every float in the body is the
// matrix's tag, so state.payload.data[0] names its origin unambiguously.
const tagForPath = { A: 1, B: 2, C: 3, D: 4 };
const requestLog = [];
// Which responses have actually been returned. `resolved` answers "has this
// landed at all"; `viewFetchCount` answers "how many times", which the button
// scenario needs because the same matrix and view get fetched twice there --
// once by the load, once by the reload that the 正規化 change issues.
// `resolved` は「届いたか」、`viewFetchCount` は「何回届いたか」。ボタンの
// シナリオでは同じ行列・同じビューが2回取得されるため後者が要る。
const resolved = new Set();
const viewFetchCount = new Map();

globalThis.fetch = async (url, options) => {
  requestLog.push(url);

  if (url === "/api/views") {
    return new Response(JSON.stringify({ views: VIEWS }), {
      headers: { "content-type": "application/json" },
    });
  }

  if (url === "/api/session/matrix") {
    const matrixPath = JSON.parse(options.body).path;
    await sleep(sessionDelayMs[matrixPath] ?? 0);
    resolved.add(`session:${matrixPath}`);
    return new Response(JSON.stringify({
      id: `matrix_${matrixPath}`,
      meta: {
        id: `matrix_${matrixPath}`, shape: [H, W, 4, 3], dtype: "float32",
        space: "camera", path: matrixPath, active_region: null,
        projector: { width: 64, height: 48 },
      },
    }), { headers: { "content-type": "application/json" } });
  }

  const m = url.match(/^\/api\/matrix\/(.+?)\/view\/([^?]+)/);
  if (m) {
    const [, id, key] = m;
    const matrixPath = id.replace("matrix_", "");
    await sleep(viewDelayMs[matrixPath] ?? 0);
    resolved.add(`view:${matrixPath}`);
    const counted = `${matrixPath}:${key}`;
    viewFetchCount.set(counted, (viewFetchCount.get(counted) ?? 0) + 1);
    const spec = VIEWS.find((v) => v.key === key);
    // Channel c carries (c + 1) * tag, not one constant for the whole array.
    //
    // The tag still identifies the matrix -- channel 0 is 1 * tag, and it is
    // the first element of the body in *both* layouts (planar: the first run;
    // interleaved: the first pixel's first channel), so state.payload.data[0]
    // reads it straight off either way and payloadTag() needs no layout
    // knowledge. But the channels differ from each other, which is what lets a
    // per-channel range be told apart from the shared one. Filled with a single
    // constant, every channel's 99th percentile came out at the header's own
    // vmax and "shared vs per-tile" was unobservable: the histogram path ran
    // and the assertion could not fail.
    // チャンネルcの値を (c+1)*tag にする。チャンネル0は 1*tag であり、planar
    // でもインタリーブでも本体の先頭要素なので data[0] は layout を問わず
    // 行列の識別子を返し、payloadTag() は layout を知らなくてよい。全体を
    // 同一定数で埋めるとチャンネル別分位点がヘッダのvmaxと一致してしまい、
    // shared と per_tile を区別できなかった。
    const tag = tagForPath[matrixPath] ?? 0;
    const interleaved = INTERLEAVED_VIEWS.has(key);
    const data = new Float32Array(H * W * spec.channels);
    if (interleaved) {
      for (let i = 0; i < data.length; i++) data[i] = ((i % spec.channels) + 1) * tag;
    } else {
      for (let c = 0; c < spec.channels; c++) {
        data.fill((c + 1) * tag, c * H * W, (c + 1) * H * W);
      }
    }
    const headers = {
      "X-Height": String(H), "X-Width": String(W),
      "X-Channels": String(spec.channels), "X-Kind": spec.kind,
      "X-Vmin": "-1.0", "X-Vmax": "1.0",
      "X-Labels-B64": Buffer.from(JSON.stringify(viewLabels(spec)), "utf-8").toString("base64"),
    };
    // Omitted, not set to "interleaved": the client must reach that layout via
    // api.js's default, which is what a pre-planar endpoint actually looks like.
    // "interleaved" と明示せず省く。旧エンドポイントの実際の見え方を再現する。
    if (key === "mangled_layout") headers["X-Layout"] = "Planar";
    else if (!interleaved) headers["X-Layout"] = "planar";
    return new Response(data.buffer, { headers });
  }
  throw new Error(`unexpected url ${url}`);
};

// ------------------------------------------------------------------- driver
await import(pathToFileURL(path.join(STATIC_DIR, "js", "main.js")).href);

async function waitFor(predicate, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return true;
    await sleep(5);
  }
  return false;
}

const started = await waitFor(() => globalThis.window.__viewer?.state?.spec);
if (!started) {
  check("frontend reached a ready state", false, "window.__viewer.state.spec never set");
  emit();
  process.exit(0);
}
const { state, views } = globalThis.window.__viewer;

// Click 読み込み for `matrixPath` the way the user does.
function clickLoad(matrixPath) {
  el("path-input").value = matrixPath;
  el("btn-load").__fire("click");
}

const payloadTag = () => (state.payload ? state.payload.data[0] : null);

// What each tile most recently uploaded: the value filling the plane GLView
// pulled out of the body. Tile c must get channel c, i.e. (c + 1) * tag,
// whichever layout the body arrived in.
// 各タイルが最後に載せた平面の値。layout に関わらずタイルcはチャンネルc、
// すなわち (c + 1) * tag を受け取らねばならない。
const uploadedPlanes = () => views.map((v) => {
  const u = v.gl.__uploads;
  return u.length ? u[u.length - 1] : null;
});

// Expected per-tile plane for matrix `tag`: full size, filled with (c+1)*tag.
const expectedPlanes = (tag, channels) => Array.from({ length: channels }, (_, c) => ({
  length: H * W, first: (c + 1) * tag, last: (c + 1) * tag,
  head: Array.from({ length: 6 }, () => (c + 1) * tag),
}));

// Expected upload for a 3-channel rgb view: one texture holding all three
// channels interleaved, so the first six elements cycle 1*tag, 2*tag, 3*tag.
// rgbビューの期待値。3チャンネルを1枚に詰めるので先頭6要素は tag の1,2,3倍を
// 2周する。
const expectedPackedRgb = (tag) => [{
  length: H * W * 3, first: tag, last: 3 * tag,
  head: [tag, 2 * tag, 3 * tag, tag, 2 * tag, 3 * tag],
}];

// Hover one pixel and read the 12 readout cells back off the DOM, which is the
// only place showProbe's layout arithmetic is observable.
// 1画素をホバーして読み取りセルをDOMから読む。showProbe の添字計算が外から
// 見えるのはここだけ。
const probeTexts = (col = 2, row = 1) => {
  views[0].onHover({ col, row });
  return el("probe-grid").children
    .filter((e) => e.className === "probe-cell")
    .map((e) => e.textContent);
};

const expectedProbe = (tag, channels) =>
  Array.from({ length: channels }, (_, c) => ((c + 1) * tag).toFixed(4));

// Expected per-tile ranges under `normalize=per_tile`.
//
// Channel c holds one value, (c + 1) * tag, so its 99th percentile of |value|
// is that value exactly -- the histogram puts every sample in the top bin and
// the result is clamped back to the channel's own maximum. Reading the planar
// body with the interleaved walk (or the reverse) mixes all twelve channels
// into each bucket and every range comes out the same, which is the failure
// this pins: twelve plausible numbers that describe the wrong channels.
// チャンネルcの値は (c+1)*tag ただ一種なので、|値| の99パーセンタイルはその
// 値そのものになる。layout を取り違えると全チャンネルが混ざり、12枚とも同じ
// レンジになる（「もっともらしいが違う」12個の数値）。
const expectedRanges = (tag, channels) => Array.from({ length: channels }, (_, c) => ({
  vmin: -(c + 1) * tag, vmax: (c + 1) * tag,
}));

// The tile grid's column headers, which is where an inherited axis label shows.
// タイルグリッドの列見出し。継承された軸ラベルが表に出るのはここである。
const colHeads = () => el("tile-grid").children
  .filter((e) => e.className === "col-head mono")
  .map((e) => e.textContent);

// How many binary view fetches have been *issued* so far. Server-side params
// are staged and submitted as a group, so "did changing a control fetch?" is a
// question about this counter, not about what eventually rendered.
// これまでに発行されたビュー取得の回数。サーバ側パラメータはまとめて送るので
// 「コントロールを変えたらフェッチしたか」はこの数で見る。
const viewRequestCount = () => requestLog.filter((u) => u.includes("/view/")).length;

// Set a staged server parameter without submitting it.
const stageParam = (name, value) => {
  const control = el(`sp-${name}`);
  if (!control) return false;
  control.value = value;
  control.__fire("change");
  return true;
};

const applyParams = () => el("sp-apply").__fire("click");

const statusText = () => el("status").textContent;
const settled = () => waitFor(() => !el("btn-load").disabled && !/取得中|読み込み中/.test(statusText()));

// Wait until the named responses have all been *returned* by the fake server,
// then give the continuations that follow each await a few turns to run. This
// is what makes "the late response lands last" deterministic instead of a
// race against a fixed sleep.
// 応答が返り切ったことを確認してから継続処理に数ターン与える。固定待ちに
// 頼らず「遅い応答が最後に届く」状況を確定させる。
async function afterResponses(...keys) {
  await waitFor(() => keys.every((k) => resolved.has(k)));
  for (let i = 0; i < 12; i++) await sleep(5);
}

// Same, counting repeats: `{ "A:coef_tiles": 2 }` waits for the second one.
async function afterViewFetches(expected) {
  const entries = Object.entries(expected);
  await waitFor(() => entries.every(([k, n]) => (viewFetchCount.get(k) ?? 0) >= n));
  for (let i = 0; i < 12; i++) await sleep(5);
}

// ---------------------------------------------------------------- scenarios
if (SCENARIO === "ordering") {
  // Phase 1 -- the slow await is reload()'s binary fetch.
  //
  // A's view transfer is slow, B's is immediate: B is issued second and
  // resolves first, so A's response lands last. A's continuation must then
  // decline to publish, or it overwrites the 12 tiles and the hover readout
  // with matrix A while the header still says B.
  // A の転送を遅くし B を即時にすると、後発の B が先に解決し A の応答が最後に
  // 届く。ここで A が書き込んでしまうと、ヘッダは B・中身は A になる。
  viewDelayMs.A = 150;
  viewDelayMs.B = 0;

  clickLoad("A");
  await sleep(25);
  clickLoad("B");
  // The race has to actually stage, or the rest of this scenario is theatre.
  // Counting issued requests is not enough: if A's response had already landed
  // by now there would be no out-of-order arrival left to guard against, and
  // every assertion below would pass for the wrong reason. Fail loudly instead
  // -- a test that quietly stops testing is worse than an intermittent one.
  // ここで A が既に着地していたら追い越しは起きず、以下の検証は別の理由で
  // 通ってしまう。発行数を数えるだけでは検出できないので明示的に落とす。
  check("the race actually staged (A had not landed when B was issued)",
    !resolved.has("view:A"), `resolvedA=${resolved.has("view:A")}`);
  await afterResponses("view:A", "view:B");

  let metaPath = state.meta ? state.meta.path : null;
  check("both view fetches were really in flight together",
    requestLog.filter((u) => u.includes("/view/")).length === 2, requestLog.join(" | "));
  check("displayed metadata describes the newest matrix (B)",
    metaPath === "B", `meta.path=${metaPath}`);
  check("state.payload is the newest matrix's data (B), not the late A",
    payloadTag() === 2, `payload tag=${payloadTag()} (A=1, B=2)`);
  check("metadata and payload describe the same matrix (slow view fetch)",
    metaPath === "B" && payloadTag() === 2,
    `meta.path=${metaPath} payload tag=${payloadTag()}`);

  // Phase 2 -- the slow await is load()'s POST /session/matrix.
  //
  // Same shape, one await earlier: C's registration is slow and D's is
  // immediate, so C's continuation is the last to run and would otherwise
  // publish C's *metadata* over a screen showing D's pixels. This is the half
  // of the ordering guard that phase 1 cannot reach.
  // 同じ形を1つ手前の await で。C のメタが後から D の画面に上書きされる。
  sessionDelayMs.C = 150;
  viewDelayMs.C = 0;

  clickLoad("C");
  await sleep(25);
  clickLoad("D");
  check("the race actually staged (C's registration had not landed when D was issued)",
    !resolved.has("session:C"), `resolvedC=${resolved.has("session:C")}`);
  await afterResponses("session:C", "session:D", "view:D");

  metaPath = state.meta ? state.meta.path : null;
  check("displayed metadata describes the newest matrix (D)",
    metaPath === "D", `meta.path=${metaPath}`);
  check("metadata and payload describe the same matrix (slow session POST)",
    metaPath === "D" && payloadTag() === 4,
    `meta.path=${metaPath} payload tag=${payloadTag()} (C=3, D=4)`);
}

if (SCENARIO === "views") {
  const options = el("view-select").children.map((o) => o.value);
  check("view selector is populated from /api/views",
    options.join(",") === "coef_tiles,plain_tiles,legacy_tiles,gamut,surface,"
      + "tri_scalar,bad_rgb,mangled_layout,mono_scalar",
    options.join(","));
  check("a second registered view is reachable from the selector",
    options.includes("plain_tiles"), options.join(","));

  clickLoad("A");
  await settled();
  check("baseline load rendered the first view", payloadTag() === 1, `tag=${payloadTag()}`);

  // --- planar body (X-Layout: planar): the channel is a contiguous run
  //
  // Both halves of the layout handling are checked here, because each fails
  // silently on its own: a wrong plane still draws a picture, and a wrong probe
  // index still prints numbers. The picture and the numbers just stop being of
  // the channel their tile is labelled with.
  // どちらも単独では静かに壊れる（誤った平面でも絵は出るし、誤った添字でも
  // 数値は出る）。ラベルと中身の対応だけが失われる。
  check("planar body is declared and read as planar",
    state.payload.layout === "planar", `layout=${state.payload.layout}`);
  check("planar body: tile c uploads channel c's plane",
    JSON.stringify(uploadedPlanes()) === JSON.stringify(expectedPlanes(1, 12)),
    JSON.stringify(uploadedPlanes().map((u) => u && u.first)));
  check("planar body: the hover readout reads channel c for cell c",
    JSON.stringify(probeTexts()) === JSON.stringify(expectedProbe(1, 12)),
    JSON.stringify(probeTexts()));

  // --- per-tile ranges, the third consumer of the layout
  //
  // `normalize=per_tile` is the only route into percentileAbsPerChannel, and
  // its planar branch was reached by the button scenario but never looked at.
  // A branch that *throws* wedges the reload and gets noticed; one that runs
  // and attributes every sample to the wrong channel does not. It returns
  // twelve finite, plausible display ranges -- of the wrong channels -- and
  // the only visible consequence is that the tiles are scaled by each other's
  // extremes. Nothing anywhere would have said so.
  // per_tile は percentileAbsPerChannel への唯一の入口で、ボタンのシナリオが
  // 通ってはいたが結果を誰も見ていなかった。例外を投げれば気付けるが、動いて
  // しまう場合は「もっともらしい12個のレンジ」が出るだけで誰も気付かない。
  check("the 正規化 control exists on the coefficient view", el("sp-normalize") !== null);
  // Staged, then submitted. Changing a server-side control must not fetch on
  // its own -- see the 適用 checks in the rgb scenario for why.
  // 値を置いてから送る。コントロールの変更単体ではフェッチしない。
  const beforeStage = viewRequestCount();
  stageParam("normalize", "per_tile");
  await sleep(60);
  check("changing 正規化 stages the value without fetching",
    viewRequestCount() === beforeStage, `${beforeStage} -> ${viewRequestCount()}`);
  applyParams();
  await afterViewFetches({ "A:coef_tiles": 2 });
  await settled();
  check("planar body: per-tile ranges follow each channel's own values",
    JSON.stringify(state.ranges) === JSON.stringify(expectedRanges(1, 12)),
    JSON.stringify(state.ranges && state.ranges.map((r) => r.vmax)));

  // --- an unsupported kind must fail loudly, not render as if it were scalar
  const before = state.payload;
  el("view-select").value = "gamut";
  el("view-select").__fire("change");
  await sleep(300);
  check("unsupported kind is refused with a message naming it",
    /未対応/.test(statusText()) && /categorical/.test(statusText()), statusText());
  check("unsupported kind leaves state.payload identity unchanged",
    state.payload === before, `same object: ${state.payload === before}`);

  // --- the `normalize` special-case must not fire for a view without it
  el("view-select").value = "plain_tiles";
  el("view-select").__fire("change");
  await settled();
  const plainRequests = requestLog.filter((u) => u.includes("/view/plain_tiles"));
  check("a view without a `normalize` param is fetched without one",
    plainRequests.length > 0 && plainRequests.every((u) => !u.includes("normalize")),
    plainRequests.join(" | "));
  check("its per-tile control is gone from the DOM", el("sp-normalize") === null);
  check("its ranges are shared, not per-tile",
    state.ranges.every((r) => r.vmax === state.ranges[0].vmax)
      && el("range-info").textContent !== "タイルごと",
    el("range-info").textContent);
  check("switching back to a scalar view still renders",
    /表示中/.test(statusText()), statusText());

  // --- interleaved body (no X-Layout at all): the pre-planar wire format
  //
  // `plain_tiles` is served interleaved and header-less on purpose; see
  // INTERLEAVED_VIEWS. Without this the fallback the client deliberately keeps
  // has no test execution anywhere and is free to rot unnoticed.
  // `plain_tiles` だけ旧形式で返している。これが無いと、意図的に残している
  // フォールバックはどこからも実行されず、黙って腐る。
  check("a response without X-Layout falls back to interleaved",
    state.payload.layout === "interleaved", `layout=${state.payload.layout}`);
  check("interleaved body: tile c uploads channel c's plane",
    JSON.stringify(uploadedPlanes()) === JSON.stringify(expectedPlanes(1, 12)),
    JSON.stringify(uploadedPlanes().map((u) => u && u.first)));
  check("interleaved body: the hover readout reads channel c for cell c",
    JSON.stringify(probeTexts()) === JSON.stringify(expectedProbe(1, 12)),
    JSON.stringify(probeTexts()));

  // --- per-tile ranges over an interleaved body: the branch nothing reached
  //
  // percentileAbsPerChannel has two walks and, until this fixture existed, only
  // the planar one was ever executed -- reaching the other needs a body that is
  // interleaved *and* a view that declares `normalize`, and no registration was
  // both. Replacing the interleaved walk with the planar one broke no check at
  // all (verified before adding this). See `legacy_tiles` in VIEWS.
  //
  // The layout is asserted alongside the ranges on purpose: if the view switch
  // silently did not happen, the ranges left over from the planar step would
  // still match and this would pass without having tested anything.
  // 分岐の片方はどこからも実行されていなかった。到達にはインタリーブかつ
  // `normalize` を持つビューが要る。layout も一緒に検証しているのは、ビュー
  // 切替が起きなかった場合に planar 側の結果が残って通ってしまうため。
  el("view-select").value = "legacy_tiles";
  el("view-select").__fire("change");
  await settled();
  check("the 正規化 control exists on the legacy view", el("sp-normalize") !== null);
  stageParam("normalize", "per_tile");
  applyParams();
  await afterViewFetches({ "A:legacy_tiles": 2 });
  await settled();
  check("interleaved body: per-tile ranges follow each channel's own values",
    state.payload.layout === "interleaved"
      && JSON.stringify(state.ranges) === JSON.stringify(expectedRanges(1, 12)),
    `layout=${state.payload.layout} `
      + JSON.stringify(state.ranges && state.ranges.map((r) => r.vmax)));

  // --- an X-Layout the client does not know must fail loudly
  //
  // `mangled_layout` is served with `X-Layout: "Planar"`: present, capitalised,
  // and in neither known layout. The header is not decoration -- four separate
  // index calculations branch on it -- so an unrecognised value has to be an
  // error rather than a silent vote for interleaved. Asserted the same way the
  // unsupported `kind` is: a message that names the offending value, and
  // `state.payload` untouched.
  // 既知でない X-Layout は黙って通してはならない。ヘッダは飾りではなく4つの
  // 添字計算が分岐する対象であり、未知の値は「インタリーブに一票」ではなく
  // 誤りである。未対応 kind と同じく、値を名指しするメッセージと
  // state.payload が変わらないことで確認する。
  const beforeMangled = state.payload;
  el("view-select").value = "mangled_layout";
  el("view-select").__fire("change");
  await sleep(300);
  check("an unknown X-Layout is refused with a message naming it",
    /X-Layout/.test(statusText()) && /Planar/.test(statusText()), statusText());
  check("an unknown X-Layout leaves state.payload identity unchanged",
    state.payload === beforeMangled, `same object: ${state.payload === beforeMangled}`);
}

if (SCENARIO === "button") {
  // The exact sequence that used to wedge the button: a reload() -- from the
  // 正規化 dropdown -- issued while a load() is still transferring. reload()
  // bumps the shared request counter, so a button keyed off that counter would
  // never be released by the load that disabled it.
  // 読み込み中に正規化を変更する。reload() が共通のリクエストIDを進めるため、
  // それを流用したボタン制御では二度と解放されない。
  viewDelayMs.A = 150;

  clickLoad("A");
  check("button is disabled while the load runs", el("btn-load").disabled === true);

  await sleep(25);
  // The load must still be transferring, or the reload below is not "during"
  // anything and the lockout it reproduces cannot occur.
  // ここで読み込みが終わっていれば「読み込み中の reload()」ではなくなり、
  // 再現しようとしているロックアウト自体が起きない。
  check("the load was still in flight when 正規化 was changed",
    !resolved.has("view:A"), `resolvedA=${resolved.has("view:A")}`);

  check("the 正規化 control exists to be changed", el("sp-normalize") !== null);
  stageParam("normalize", "per_tile");
  applyParams();

  // Both the load's own fetch and the one the 正規化 change issued have to
  // have landed before the button state means anything -- they are the same
  // matrix and the same view, so it is the second arrival that matters.
  // ボタンの状態を見る前に、読み込み自身の取得と正規化変更が出した取得の
  // 両方が着地している必要がある。同じ行列・同じビューなので2回目を待つ。
  await afterViewFetches({ "A:coef_tiles": 2 });
  check("button is released after a reload during an in-flight load",
    el("btn-load").disabled === false,
    `disabled=${el("btn-load").disabled}`);

  // The same hazard through the other entry point: the view selector.
  // もう一方の入口（ビュー切替）でも同じことが起きないか確認する。
  viewDelayMs.B = 150;
  clickLoad("B");
  await sleep(25);
  check("the load was still in flight when the view was switched",
    !resolved.has("view:B"), `resolvedB=${resolved.has("view:B")}`);
  el("view-select").value = "plain_tiles";
  el("view-select").__fire("change");
  await afterViewFetches({ "B:coef_tiles": 1, "B:plain_tiles": 1 });
  check("button is released after a view switch during an in-flight load",
    el("btn-load").disabled === false,
    `disabled=${el("btn-load").disabled}`);
}

if (SCENARIO === "rgb") {
  // An rgb view is three channels that are *one picture*, and every assertion
  // here is about that difference surviving the round trip.
  //
  // The failure mode this guards is not a crash. `surface`'s labels are
  // ["R","G","B"], which `axesFromLabels` cannot parse, so the label-derived
  // fallback produces a 1x3 grid: three tiles, each drawing one component
  // through the scalar path. That renders. It just renders three greyscale
  // pictures of a colour, which is precisely the thing the view exists not to
  // do -- and it looks like a working viewer while doing it.
  // rgbビューの3チャンネルは1枚の絵である。ラベル ["R","G","B"] は
  // `axesFromLabels` では解釈できず、フォールバックだと1行3列＝3枚の
  // グレースケールになる。落ちはしないが、このビューが避けたいものそのもの。
  clickLoad("A");
  await settled();
  check("baseline scalar load drew one tile per channel",
    views.length === 12, `tiles=${views.length}`);
  // The colormap turns a *number* into a colour, so it belongs to scalar views
  // and to no other kind. Checked on both sides of the switch below, because
  // "always enabled" and "always disabled" each pass one half.
  // カラーマップは数値を色にする仕組みなので scalar 専用である。切替の前後
  // 両方で見る（常時有効・常時無効のどちらも片側だけなら通ってしまう）。
  check("the colormap control is enabled for a scalar view",
    el("cmap-select").disabled === false, `disabled=${el("cmap-select").disabled}`);

  // Registering an rgb view in Python has to be enough to reach it from the
  // GUI -- the same promise the selector was added for, restated for the kind
  // that arrived after it.
  // Pythonで登録するだけでGUIから選べること。セレクタ導入時の約束を、後から
  // 増えた kind についても言い直している。
  const options = el("view-select").children.map((o) => o.value);
  check("the rgb view is reachable from the selector",
    options.includes("surface"), options.join(","));

  el("view-select").value = "surface";
  el("view-select").__fire("change");
  await settled();

  check("an rgb view renders instead of being refused as an unknown kind",
    /表示中/.test(statusText()), statusText());
  check("an rgb view draws one tile, not one greyscale tile per channel",
    views.length === 1, `tiles=${views.length}`);
  check("the rgb payload is uploaded as one packed interleaved RGB texture",
    JSON.stringify(uploadedPlanes()) === JSON.stringify(expectedPackedRgb(1)),
    JSON.stringify(uploadedPlanes()));
  check("the single tile is told it is in rgb mode",
    views[0].image && views[0].image.kind === "rgb",
    `kind=${views[0].image && views[0].image.kind}`);
  // And so is the shader. Uploading RGB32F while leaving u_mode at 0 draws the
  // red component alone through the scalar colormap -- a picture, of one third
  // of the data, with nothing to say so.
  // シェーダ側にも伝わっていること。RGB32Fを載せて u_mode を0のままにすると
  // 赤成分だけがカラーマップを通り、3分の1のデータの絵が黙って出る。
  check("the shader is told u_mode = 1 for the rgb tile",
    views[0].gl.__uniforms.u_mode === 1, `u_mode=${views[0].gl.__uniforms.u_mode}`);
  // One tile, but still three numbers to report: the readout grid keeps one
  // cell per channel rather than mirroring the tile grid.
  // タイルは1枚でも読み取り値は3つある。読み取りグリッドはチャンネル数で組む。
  check("the hover readout still reports all three channels",
    JSON.stringify(probeTexts()) === JSON.stringify(expectedProbe(1, 3)),
    JSON.stringify(probeTexts()));
  // rgb data is already a colour and the shader's rgb branch never reaches the
  // colormap. Leaving the dropdown live would offer a control that silently
  // does nothing -- and, worse, suggests the picture is a false-colour render
  // of something rather than the surface's own colour.
  // rgb は既に色であり、シェーダの rgb 経路はカラーマップを通らない。有効の
  // まま残すと何も起きないコントロールを見せることになり、しかも「この絵は
  // 何かの疑似カラーだ」という誤った印象まで与える。
  check("the colormap control is disabled for an rgb view",
    el("cmap-select").disabled === true, `disabled=${el("cmap-select").disabled}`);

  // --- the whole parameter group is one submit, not one fetch per control
  //
  // Moving the projection command from (1,1,1) to (0,0,0) touches three
  // controls. Fetching on each `change` made that three full server renders --
  // measured at ~740 ms and ~390 MB peak *each* on the 1080p matrix, two thirds
  // of it discarded by reqId after the fact, and computed concurrently because
  // render_view is a sync def whose NumPy kernels release the GIL. The staging
  // half is asserted separately from the submit half: a build that fetched
  // eagerly *and* on 適用 would pass a "one fetch happened" check on its own.
  // 投影コマンドを (1,1,1) から (0,0,0) にするとコントロールは3つ動く。
  // change ごとに取得すると 1080p で約740ms・ピーク約390MB のレンダが3回走り、
  // うち2回分は reqId が捨てる（しかも render_view は同期関数なので3回は並行に
  // 走る）。「置くだけ」と「送る」を別々に検証するのは、両方やる実装でも
  // 「1回取得された」だけなら通ってしまうためである。
  const beforeStaging = viewRequestCount();
  check("the rgb view's three command controls exist",
    ["p_r", "p_g", "p_b"].every((n) => el(`sp-${n}`) !== null));
  for (const name of ["p_r", "p_g", "p_b"]) stageParam(name, "0");
  await sleep(80);
  check("changing three server params issues no fetch on its own",
    viewRequestCount() === beforeStaging,
    `${beforeStaging} -> ${viewRequestCount()}`);

  applyParams();
  await settled();
  check("適用 submits the whole group as exactly one fetch",
    viewRequestCount() === beforeStaging + 1,
    `${beforeStaging} -> ${viewRequestCount()}`);
  const applied = requestLog[requestLog.length - 1];
  check("適用 sends every staged value, not just the last one edited",
    /p_r=0/.test(applied) && /p_g=0/.test(applied) && /p_b=0/.test(applied), applied);

  // --- a 1-channel scalar view must not inherit the rgb view's axis label
  //
  // Run here, directly after `surface`, because that is the only state in which
  // the bug exists: layoutFor may keep the current axes when their count
  // matches, and an rgb layout is 1 row x 1 col, so `1 * 1 === 1` lets a
  // 1-channel scalar view walk off with `cols = ["RGB"]` and caption a
  // greyscale tile as a colour. It renders either way; only the heading is
  // wrong, which is exactly why nothing else would catch it.
  // `surface` の直後でしか起きないので、ここで確認する。layoutFor は枚数が
  // 一致すれば現在の軸を引き継ぐが、rgb の配置は1行1列なので 1*1===1 が
  // 通ってしまい、1チャンネルの scalar が cols=["RGB"] を持ち去って
  // グレースケール1枚に色の見出しを付ける。どちらでも描けてしまい、
  // 違うのは見出しだけなので、他のどのチェックにも引っかからない。
  el("view-select").value = "mono_scalar";
  el("view-select").__fire("change");
  await settled();
  check("a 1-channel scalar view after an rgb one draws its own tile",
    views.length === 1, `tiles=${views.length}`);
  check("a 1-channel scalar view does not inherit the rgb view's column label",
    JSON.stringify(colHeads()) === JSON.stringify(["only"]),
    JSON.stringify(colHeads()));

  // Going back has to rebuild the 12-tile grid. A one-way trip into rgb would
  // leave a single tile drawing coefficient 0 and eleven channels invisible.
  // 戻れることまで確認する。片道だと係数タイルが1枚だけになる。
  el("view-select").value = "coef_tiles";
  el("view-select").__fire("change");
  await settled();
  check("switching back to a scalar view restores one tile per channel",
    views.length === 12, `tiles=${views.length}`);
  check("switching back to a scalar view uploads single-channel planes again",
    JSON.stringify(uploadedPlanes()) === JSON.stringify(expectedPlanes(1, 12)),
    JSON.stringify(uploadedPlanes().map((u) => u && u.length)));
  check("switching back to a scalar view puts the shader back in u_mode = 0",
    views.every((v) => v.gl.__uniforms.u_mode === 0),
    JSON.stringify(views.map((v) => v.gl.__uniforms.u_mode)));

  // The other half of layoutFor's decision: three channels that are *not* one
  // colour. `tri_scalar` differs from `surface` only in `kind`, so a layout
  // that keyed on the channel count alone would draw it as a single colour
  // tile -- two of its three pictures gone, the third recoloured. That renders
  // too, which is why the count is asserted rather than the absence of a throw.
  // タイル数の判断のもう半分。`tri_scalar` は kind だけが `surface` と違う。
  // チャンネル数だけで決める実装だと1枚の色タイルになり、3枚のうち2枚が消える
  // ——これも描画自体は成功するので、例外の有無ではなく枚数を検証する。
  el("view-select").value = "tri_scalar";
  el("view-select").__fire("change");
  await settled();
  check("a 3-channel view whose kind is scalar still draws one tile per channel",
    views.length === 3, `tiles=${views.length}`);

  // An rgb view that is not three channels is not an rgb view. Refused from the
  // registry, before the transfer: reaching packRgb with a short payload reads
  // past the end of the array, and JavaScript answers `undefined` rather than
  // throwing -- NaN in the texture, the whole tile in the mask colour, and no
  // route from "everything is masked" back to "the view declared 4 channels".
  // rgb を名乗る非3チャンネルは rgb ではない。転送前に登録内容の段階で断る。
  // packRgb まで届くと終端越えの読み出しが undefined → NaN となり、タイル全面
  // がマスク色になるが、そこから「4チャンネルで登録されている」へは辿れない。
  el("view-select").value = "bad_rgb";
  el("view-select").__fire("change");
  await sleep(300);
  const badRgbFetches = requestLog.filter((u) => u.includes("/view/bad_rgb"));
  check("an rgb view that is not 3 channels is refused before the transfer, naming the count",
    /チャンネル数が 4/.test(statusText()) && badRgbFetches.length === 0,
    `${statusText()} | fetches=${badRgbFetches.length}`);
}

if (!checks.length) check(`unknown scenario ${SCENARIO}`, false);
emit();
process.exit(0);
