// Drives the shipped viewer frontend under a stub DOM/WebGL, for pytest.
//
// Usage: node frontend_harness.mjs <static-dir> <scenario>
//   <static-dir>  directory holding index.html and js/{main,glview,api}.js
//   <scenario>    "ordering" | "views" | "button" | "rgb" | "probe"
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
    children: [], parentElement: null,
    style: { setProperty(k, v) { el.style[k] = v; } },
    // Backed by `className`, the way the DOM backs it, rather than swallowed.
    //
    // It used to be three empty functions, which made one thing invisible from
    // out here: `status(msg, true)` marks the line by toggling `error`, and
    // with the calls dropped a status bar left red over a failure that has
    // since been recovered from was indistinguishable from a clean one. The
    // stub must not be the reason a behaviour cannot be checked.
    // `className` を裏付けにする（DOMと同じ）。空関数のままだと
    // `status(msg, true)` の付ける `error` が外から見えず、「復旧済みなのに
    // 赤いままのステータス行」を検出できない。検証できない理由がスタブ側に
    // あってはならない。
    classList: {
      _tokens: () => String(el.className).split(/\s+/).filter(Boolean),
      _set(tokens) { el.className = [...new Set(tokens)].join(" "); },
      contains(name) { return this._tokens().includes(name); },
      add(...names) { this._set([...this._tokens(), ...names]); },
      remove(...names) { this._set(this._tokens().filter((t) => !names.includes(t))); },
      toggle(name, force) {
        const on = force === undefined ? !this.contains(name) : !!force;
        if (on) this.add(name); else this.remove(name);
      },
    },
    // Parentage is recorded, not just membership: a check has to be able to
    // ask *where* an element sits, and `children` alone only answers downwards.
    // 親も記録する。要素が「どこに居るか」を問うチェックには下向きの
    // `children` だけでは足りない。
    appendChild(child) { child.parentElement = el; el.children.push(child); return child; },
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
      // Detached, not merely forgotten: a removed child that kept its
      // `parentElement` would still walk up into a tree it is no longer in.
      // 取り外した子は親への参照も切る。残すと、既に木から外れた要素が
      // 祖先を辿れてしまう。
      for (const child of el.children) { unregister(child); child.parentElement = null; }
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

// Build the element *tree* index.html declares, not merely its ids.
//
// The registry alone answers "does this element exist", which was all the
// earlier checks asked. It cannot answer "*where* does it sit", and that is a
// real property of this page: the display range and the timings have to be in
// the bottom dashboard, beside the matrix, rather than at the foot of a sidebar
// that scrolls -- and an id registry is perfectly indifferent to which of the
// two contains them. Rather than weaken that check to "the ids still exist"
// (which is the shape of a check that cannot fail), the stub grew a parser.
//
// Deliberately minimal: it records the tag, the `class` and the `id`, and the
// nesting. Not `value`, not boolean attributes -- every one of those would
// change what the *existing* scenarios see at startup, and a stub that starts
// simulating a browser more broadly than the checks need is a new source of
// false answers rather than a better fixture.
//
// index.html が宣言している要素の木を組む。id の登録だけでは「その要素が
// 在るか」しか答えられず、「どこに在るか」——表示レンジと計測が、縦に溢れる
// サイドバーの末尾ではなく下部ダッシュボードに在ること——を検証できない。
// 「idが在ること」にチェックを弱めるのは「落ちないテスト」を作ることなので、
// 代わりにスタブ側を直す。記録するのはタグ・class・id・入れ子だけに留める
// （value や真偽属性まで再現すると既存シナリオの初期状態が変わり、
// フィクスチャの改善ではなく新しい誤答の発生源になる）。
const VOID_ELEMENTS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img",
  "input", "link", "meta", "param", "source", "track", "wbr",
]);
// Elements whose content is text, not markup. `if (a < b)` inside a <script>
// otherwise opens a phantom <b>, and while a closing </script> happens to pop it
// again (a close unwinds to the innermost match), that is luck rather than a
// rule: `a < b && c > d` would close the phantom itself and leave the following
// elements nested one level too deep. Skipping to the matching close is what
// makes the tree independent of what the page's script says.
// 中身がテキストであってマークアップではない要素。`if (a < b)` は幻の <b> を
// 開いてしまう。今日は `</script>` が偶然それを閉じているだけで規則ではなく、
// `a < b && c > d` なら幻を自分で閉じ、以降の入れ子が1段ずれる。閉じタグまで
// 読み飛ばすことで、木の形をページのスクリプトの中身から切り離す。
const RAW_TEXT_ELEMENTS = new Set(["script", "style", "textarea", "title"]);
// Comments first, so a commented-out tag cannot open an element. Attribute
// values are matched as quoted runs so a ">" inside one does not end the tag.
// コメントを先に食う。属性値は引用符ごと拾い、値の中の ">" でタグが
// 終わらないようにする。
const TAG_RE = /<!--[\s\S]*?-->|<(\/?)([a-zA-Z][\w-]*)((?:"[^"]*"|'[^']*'|[^>])*?)(\/?)>/g;
const ATTR_RE = /([a-zA-Z_:][-\w:.]*)\s*=\s*"([^"]*)"/g;

function parseHtml(source) {
  const root = makeEl("#document");
  const open = [root];
  let rawText = null;   // the raw-text element currently being skipped over
  for (const m of source.matchAll(TAG_RE)) {
    if (m[0].startsWith("<!--")) continue;
    const [, closing, rawTag, rawAttrs, selfClosed] = m;
    const tag = rawTag.toLowerCase();
    if (rawText) {
      // Everything until this element's own close tag is its text.
      if (!(closing && tag === rawText)) continue;
      rawText = null;   // and the close itself is handled below, as any other
    }
    if (closing) {
      // Close the innermost element of that name. An unbalanced close is
      // ignored rather than unwinding the stack to the root.
      // 同名の最も内側の要素を閉じる。対応が無ければ無視する。
      for (let i = open.length - 1; i > 0; i--) {
        if (open[i].tagName === tag.toUpperCase()) { open.length = i; break; }
      }
      continue;
    }
    const node = makeEl(tag);
    const attrs = {};
    for (const a of rawAttrs.matchAll(ATTR_RE)) attrs[a[1].toLowerCase()] = a[2];
    if (attrs.class) node.className = attrs.class;
    if (attrs.id) node.id = attrs.id;          // the setter is what registers it
    // An inline `style` is recorded because an inline declaration is the one
    // thing that beats the stylesheet: `<section class="readout"
    // style="display:block">` restores the stacked layout this task replaced
    // while every class-based check stays green. No element in index.html
    // carries one today, so recording it changes nothing that runs -- it is
    // there so that the check on `.readout`'s layout cannot be sidestepped.
    // インラインの `style` だけは記録する。シートに勝つのはこれだけであり、
    // `style="display:block"` を書けば縦積みに戻せてしまう。現在の index.html
    // にこの属性を持つ要素は無いので、記録しても既存の挙動は変わらない。
    if (attrs.style) {
      for (const decl of attrs.style.split(";")) {
        const at = decl.indexOf(":");
        if (at > 0) node.style[decl.slice(0, at).trim()] = decl.slice(at + 1).trim();
      }
    }
    open[open.length - 1].appendChild(node);
    if (!selfClosed && !VOID_ELEMENTS.has(tag)) {
      open.push(node);
      if (RAW_TEXT_ELEMENTS.has(tag)) rawText = tag;
    }
  }
  return root;
}

// Exactly the ids index.html declares are registered, and nothing else, so a
// reference to an id the page does not have throws rather than silently
// finding a blank stub.
// 宣言されているidだけが登録される。存在しないidの参照は黙って空要素を
// 拾わず例外になる。
const INDEX_HTML = readFileSync(path.join(STATIC_DIR, "index.html"), "utf-8");
const documentTree = parseHtml(INDEX_HTML);

// The parser's real hazard is not a crash, it is a *different tree* built in
// silence: a placement check would then be answering about a page nobody wrote.
// Comparing the ids it registered against a plain scan of the same file turns
// any such divergence into an immediate, loud harness error -- the ids are the
// one part of the tree an independent reading can check, and every check here
// reaches its element through one.
//
// Comments are stripped from both sides, so a commented-out element is absent
// from each for the same reason rather than by coincidence.
//
// パーサの本当の危険は例外ではなく、黙って別の木ができることである。その木の
// 上で置き場所を検証しても、誰も書いていないページについて答えることになる。
// 登録された id を素朴な走査と突き合わせれば、食い違いはその場で harness の
// エラーになる。id は木のうち独立に読み直せる唯一の部分であり、どのチェックも
// id を経由して要素に到達している。コメントは両方から落とす。
const naiveIds = new Set(
  [...INDEX_HTML.replace(/<!--[\s\S]*?-->/g, "").matchAll(/\sid="([^"]+)"/g)]
    .map((m) => m[1]),
);
const idDiff = [
  ...[...naiveIds].filter((id) => !registry.has(id)).map((id) => `only in the file: ${id}`),
  ...[...registry.keys()].filter((id) => !naiveIds.has(id)).map((id) => `only in the tree: ${id}`),
];
if (idDiff.length) {
  throw new Error(
    `the element tree parsed out of index.html disagrees with a plain scan of the `
    + `same file (${idDiff.join(", ")}). The tree is what every placement check `
    + `reads, so it must not be a different page from the one on disk.`,
  );
}

// The stylesheet, as text, with comments removed.
//
// There is no CSS engine here and there is not going to be one, so a layout
// cannot be measured. What can be pinned is the declaration itself: `.readout`
// is a row of panels, and `display: block` there is the stacked layout this
// whole task exists to replace -- a change that produces no error, no wrong
// number, and no failing check anywhere else in this file.
//
// スタイルシートを本文として読む。ここに CSS エンジンは無く、今後も置かない
// ので、レイアウトを測ることはできない。固定できるのは宣言そのものである。
// `.readout` は板を横に並べる行であり、そこが `display: block` になれば
// このタスクが置き換えた縦積みに戻る——例外も、誤った数値も、他のどの
// チェックの失敗も伴わずに。
const CSS_TEXT = readFileSync(path.join(STATIC_DIR, "style.css"), "utf-8")
  .replace(/\/\*[\s\S]*?\*\//g, "");
const cssDeclarations = (selector) => {
  const found = [];
  for (const rule of CSS_TEXT.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (rule[1].split(",").map((s) => s.trim()).includes(selector)) found.push(rule[2]);
  }
  return found.join(" ");
};

globalThis.document = {
  documentElement: documentTree.children.find((e) => e.tagName === "HTML") ?? makeEl("html"),
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

// `values_are_matrix` is the registry's answer to "is the picture the matrix
// itself?", and the frontend drops its second readout panel where it is true.
// The three kinds of answer all have a fixture: `true` (the coefficient views),
// an explicit `false` (`surface`), and *absent* (everything else) -- which has
// to keep the panel, because a view that does not answer is not a view whose
// values are known to be a duplicate. A default that hid it would silently
// blank the only numbers a newly registered view produces.
// `values_are_matrix` は「描かれているものが行列そのものか」という宣言で、
// 真ならフロントは読み取り欄の2枚目の板を落とす。3通りの答え全てに
// フィクスチャを置く: 真（係数系のビュー）・明示的な偽（`surface`）・無宣言
// （その他）。無宣言は板を残す側でなければならない。
const VIEWS = [
  {
    key: "coef_tiles", label: "12係数タイル", kind: "scalar", channels: 12,
    values_are_matrix: true,
    params: [{
      name: "normalize", label: "正規化", type: "select",
      default: "shared", choices: ["shared", "per_tile"],
    }],
  },
  // A second registration, to check it is reachable at all (F4).
  {
    key: "plain_tiles", label: "パラメータなしタイル", kind: "scalar", channels: 12,
    values_are_matrix: true, params: [],
  },
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
    values_are_matrix: true,
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
    values_are_matrix: false,
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
  // Served with no `X-Layout` header at all. The client used to read that as
  // "interleaved"; it now refuses, because nothing in this repository omits the
  // header and a stripped one is otherwise indistinguishable from a declared
  // interleaved body -- which is precisely the silent misattribution the guard
  // exists for. Registered alongside `mangled_layout` so both halves of "the
  // header must be present *and* known" have a fixture.
  // `X-Layout` を一切付けずに返す。以前はインタリーブとして読んでいたが、
  // このリポジトリにヘッダを省く経路は無く、欠落は「インタリーブと申告された
  // 本体」と区別が付かない——ガードが防ぎたい静かな取り違えそのものである。
  // 「ヘッダが有り、かつ既知であること」の両方にフィクスチャを持たせる。
  { key: "headerless_tiles", label: "配置未申告タイル", kind: "scalar", channels: 12, params: [] },
  // Answered with a real FastAPI-shaped refusal: 400 and `{"detail": …}`. This
  // is what `surface` returns for a camera-space matrix, and the only fixture
  // here that exercises the *server's* error path rather than a client-side
  // guard. `fetchView` used to `throw new Error(await res.text())`, which put
  // the whole JSON envelope on screen -- braces, quotes and backslashes -- in a
  // status bar that is otherwise Japanese prose.
  // FastAPI と同じ形の拒否（400 + `{"detail": …}`）を返す。カメラ座標の行列に
  // 対して `surface` が返すものであり、クライアント側のガードではなく
  // *サーバ側*のエラー経路を通す唯一のフィクスチャである。以前は本文をそのまま
  // 投げていたため、JSONの外側ごと（波括弧も引用符も）ステータス行に出ていた。
  { key: "refused_view", label: "サーバが断るビュー", kind: "scalar", channels: 12, params: [] },
];

// The `detail` the fake server refuses `refused_view` with. Contains a quote,
// so a raw-body render is unmistakable: it would show the backslash escape.
// 引用符を含めてある。生のまま出せばバックスラッシュのエスケープが見えるので、
// 取り違えようがない。
const REFUSAL_DETAIL =
  'view "refused_view" needs a projector-space matrix; this one is in camera space.';

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
  headerless_tiles: TILE_LABELS,
};

const viewLabels = (spec) => LABELS_BY_KEY[spec.key];

// `plain_tiles` and `legacy_tiles` are served interleaved (H, W, C) -- and say
// so, with `X-Layout: interleaved`.
//
// The shipped client keeps an interleaved decode path on purpose, and once
// every fixture became planar that path had no test execution anywhere and was
// free to rot. These two exercise it. They used to omit the header entirely,
// relying on api.js's "absent means interleaved" default; that default is gone
// (a stripped header is now refused, see `headerless_tiles`), so the layout
// they are actually using is now declared rather than inferred.
// `plain_tiles` と `legacy_tiles` はインタリーブ (H, W, C) で返し、
// `X-Layout: interleaved` と明示する。出荷クライアントは意図的にインタリーブ
// の復号経路を残しているが、全フィクスチャを planar にした時点でその経路は
// どこからも実行されなくなった。この2つがそれを通す。以前はヘッダを省いて
// api.js の既定値に頼っていたが、その既定値は廃止した（欠落は
// `headerless_tiles` のとおり拒否される）ので、使っている配置を明示する。
const INTERLEAVED_VIEWS = new Set(["plain_tiles", "legacy_tiles"]);

// ------------------------------------------------------------- point query
//
// One pixel's 4x3 coefficients, which is the whole of what the readout table
// now gets from the server. Every value depends on *both* row and col and no
// two positions overlap, so a transposed request is visible rather than
// harmless; within a block the twelve values are distinct and ascending, so a
// column-major render is visible too. A fixture symmetric enough to pass
// either way is how this repository has repeatedly shipped tests that could
// not fail.
// 1画素分の4x3。読み取り表がサーバから受け取るのはこれだけである。値は row と
// col の両方に依存し、位置ごとに範囲が重ならないので row/col の取り違えが
// 見える。ブロック内の12個も相異なるため列優先で描いた場合も見える。どちら向き
// でも通ってしまう対称なフィクスチャこそ、このリポジトリが繰り返し作ってきた
// 「落ちないテスト」の作り方である。
const PROBE_OVERRIDES = new Map();   // "row,col" -> a 4x3 (entries may be null)
const probeKey = (row, col) => `${row},${col}`;

// Pixels the fake server answers with `inside_active_region: false`.
//
// The values it returns for them are ordinary finite numbers, because that is
// what the real file holds: `coef_tiles.py` applies the NaN mask *at render
// time*, so the tile is painted grey while the stored array still has a
// plausible-looking coefficient underneath. On
// data/linear_pipeline_1080/matrices_linear_gamma.npy (active region
// [613, 201, 722, 722]) row 5, col 5 reads 13.440 / 1.197 / 0.017 -- numbers
// that mean nothing, under a grey tile, in a table that used to present them
// exactly as it presents real ones.
//
// Not on the diagonal: (row 0, col 2) transposes to (row 2, col 0), which is
// inside, so a client that swapped the two coordinates loses the marker instead
// of coincidentally keeping it.
//
// この画素に対しても値は有限で返す。実ファイルがそうだからである——NaNマスクは
// `coef_tiles.py` が描画時に掛けるので、タイルは灰色でも配列にはもっともらしい
// 係数が残っている（実測: row 5, col 5 で 13.440 / 1.197 / 0.017）。読み取り表は
// それを本物と同じ顔で出していた。対角線上に置かないのは、row と col を
// 取り違えた実装が (row 2, col 0)＝領域内を訊くことになり、印が消えるように
// するためである。
const OUTSIDE_ACTIVE_REGION = new Set([probeKey(0, 2)]);
const probeBlock = (row, col) =>
  PROBE_OVERRIDES.get(probeKey(row, col))
  ?? [0, 1, 2, 3].map((i) =>
    [0, 1, 2].map((j) => (row + 1) * 1000 + (col + 1) * 100 + i * 3 + j));

// Point-query responses can be held open, so "a request is in flight" becomes
// a state a scenario can actually put the client into.
//
// Without this the coalescing and the staleness guard are both untestable, and
// untestable in the quiet way: every fetch would settle within a microtask, so
// five hovers would look identical whether they cost one round trip or five
// (each having finished before the next began), and no response could ever
// come back for a pixel the pointer had already left. The checks would pass
// against an implementation with neither guard in it.
// 点クエリの応答を保留できるようにする。これが無いと合流もスタール判定も
// 検証できない——しかも静かに検証できない。すべての fetch がマイクロタスク内で
// 解決してしまうため、5回のホバーが1往復でも5往復でも同じに見え（各回が次の前に
// 完了する）、「ポインタが去った画素の応答が後から届く」状況自体が作れない。
// どちらのガードも無い実装に対してチェックが通ってしまう。
let holdProbes = false;
const heldProbes = [];   // [{ key, release }] -- responses waiting to be let go
const probeLog = [];     // every probe URL, recorded when it is *issued*

const heldIndex = (row, col) => heldProbes.findIndex((h) => h.key === probeKey(row, col));
function releaseProbe(row, col) {
  const i = heldIndex(row, col);
  if (i < 0) return false;
  heldProbes.splice(i, 1)[0].release();
  return true;
}
function releaseHeldProbes() {
  while (heldProbes.length) heldProbes.shift().release();
}
const waitForHeld = (row, col) => waitFor(() => heldIndex(row, col) >= 0);

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

  // The point query, answered the way routers/session.py::probe_matrix does:
  // a row-major 4x3, `null` for every non-finite coefficient, and a 400 for a
  // pixel outside the matrix.
  // 点クエリ。routers/session.py::probe_matrix と同じ形で返す。
  const probe = url.match(/^\/api\/matrix\/([^/?]+)\/probe\?(.*)$/);
  if (probe) {
    const [, , qs] = probe;
    probeLog.push(url);
    const query = new URLSearchParams(qs);
    const row = Number(query.get("row")), col = Number(query.get("col"));
    if (holdProbes) {
      await new Promise((release) => heldProbes.push({ key: probeKey(row, col), release }));
    }
    if (!(row >= 0 && row < H && col >= 0 && col < W)) {
      return new Response(JSON.stringify({
        detail: `pixel (row=${row}, col=${col}) is outside the matrix`,
      }), { status: 400, headers: { "content-type": "application/json" } });
    }
    return new Response(JSON.stringify({
      row, col, values: probeBlock(row, col),
      inside_active_region: !OUTSIDE_ACTIVE_REGION.has(probeKey(row, col)),
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
    if (key === "refused_view") {
      return new Response(JSON.stringify({ detail: REFUSAL_DETAIL }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
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
    // One masked sample: `surface`'s green channel at (row 3, col 0).
    //
    // Every other value in every fixture is finite, so the view-value panel's
    // own mask class had no execution anywhere and could have been deleted
    // with nothing failing. Masked pixels are not exotic in this viewer --
    // coef_tiles and surface both write NaN outside the active projection
    // region, which on the real matrix is most of the frame -- so the panel
    // has to render one, and be seen to.
    //
    // Placed where nothing else looks: the rgb upload assertions read the
    // first six elements, the first and the last, and this is none of them.
    // 1つだけ非有限の標本を置く（`surface` の緑チャンネル、row 3 / col 0）。
    // 他のフィクスチャは全て有限なので、ビュー値パネルのマスク用クラスは
    // どこからも実行されず、消しても何も落ちない状態だった。有効投影領域の
    // 外は coef_tiles も surface も NaN を書くため、マスクされた画素は珍しく
    // ない。rgbの転送検査が見る先頭6要素・先頭・末尾のいずれでもない位置に置く。
    if (key === "surface") data[1 * H * W + 3 * W + 0] = NaN;
    const headers = {
      "X-Height": String(H), "X-Width": String(W),
      "X-Channels": String(spec.channels), "X-Kind": spec.kind,
      "X-Vmin": "-1.0", "X-Vmax": "1.0",
      "X-Labels-B64": Buffer.from(JSON.stringify(viewLabels(spec)), "utf-8").toString("base64"),
    };
    // Every view declares its layout, except `headerless_tiles`, whose whole
    // purpose is to omit the header and be refused for it.
    // すべてのビューが配置を申告する。唯一の例外が `headerless_tiles` で、
    // ヘッダを省いて拒否されることそのものが役目である。
    if (key === "mangled_layout") headers["X-Layout"] = "Planar";
    else if (key === "headerless_tiles") { /* deliberately omitted */ }
    else headers["X-Layout"] = interleaved ? "interleaved" : "planar";
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

// The readout is two panels with different sources, so it takes two readers.
//
// `#view-values` holds the *displayed view*'s channels at that pixel, read out
// of the payload already in hand -- the only place showViewValues' layout
// arithmetic is observable. `#probe-grid` holds the *matrix*'s own 4x3, which
// arrives from the point query and therefore lands one round trip later.
// 読み取り欄は出所の違う2枚の板でできているので、読み手も2つ要る。
// `#view-values` は表示中のビューの値（手元のペイロードから同期的に作られる）で、
// showViewValues の添字計算が外から見えるのはここだけ。`#probe-grid` は行列
// そのものの4x3で、点クエリの1往復ぶん遅れて届く。
const viewValueItems = () => el("view-values").children;
const viewValueLabels = () => viewValueItems().map((item) => item.children[0].textContent);
const viewValueNumbers = () => viewValueItems().map((item) => item.children[1].textContent);
const viewValueClasses = () => viewValueItems().map((item) => item.children[1].className);

// Hover one pixel and read the view-value panel back off the DOM.
const probeTexts = (col = 2, row = 1) => {
  views[0].onHover({ col, row });
  return viewValueNumbers();
};

const expectedProbe = (tag, channels) =>
  Array.from({ length: channels }, (_, c) => ((c + 1) * tag).toFixed(4));

// The 12 matrix cells: their text, and their class (which is what carries
// "this coefficient is masked" -- see .probe-cell.nan in style.css).
// 行列の12セル。文字と、マスクを表すクラスの両方を見る。
const matrixCells = () => el("probe-grid").children.filter((e) => /^probe-cell/.test(e.className));
const matrixTexts = () => matrixCells().map((e) => e.textContent);
const matrixClasses = () => matrixCells().map((e) => e.className);
const expectedMatrixTexts = (row, col) =>
  probeBlock(row, col).flat().map((v) => (v === null ? "NaN" : v.toFixed(4)));

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

// Where an element sits in the page, from the tree parsed out of index.html.
// `panelOf` is the chain of class names above it, which is what a placement
// check reports when it fails -- "sidebar > group readouts" says what happened
// far better than "false".
// index.html から組んだ木の上での位置。失敗時は上位のクラス名の連なりを出す
// （「false」より「sidebar > group readouts」のほうが原因に近い）。
const ancestors = (node) => {
  const out = [];
  for (let p = node && node.parentElement; p; p = p.parentElement) out.push(p);
  return out;
};
const sitsIn = (id, className) =>
  ancestors(el(id)).some((a) => a.classList.contains(className));
const panelOf = (id) =>
  `${id}: ${ancestors(el(id)).map((a) => a.className || a.tagName).reverse().join(" > ")}`;

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

// Type a value and press Enter, in the order a browser produces the events.
//
// This order is the whole point. The value property updates as the user types,
// so it is already current when `keydown` fires; the browser only fires
// `change` *afterwards*, to say the edit was committed. A frontend that armed
// its "not applied" indicator from the `change` event therefore re-armed it
// immediately after Enter had applied the value, and nothing came along to
// clear it again.
// ブラウザが出す順序どおりに再現する。この順序こそが要点で、`value` は入力中に
// 更新されるため `keydown` の時点で既に新しく、`change` は「確定した」ことを
// 告げるためにその*後*で来る。「未適用」表示を `change` で立てる実装は、
// Enter が適用し終えた直後に立て直してしまい、以後それを降ろす者がいない。
const pressEnter = (name, value) => {
  const control = el(`sp-${name}`);
  if (!control) return false;
  control.value = value;
  control.__fire("keydown", { key: "Enter" });
  control.__fire("change");
  return true;
};

const applyParams = () => el("sp-apply").__fire("click");

// Does the 適用 button claim there are unapplied changes?
const applyPending = () => /\bpending\b/.test(el("sp-apply").className);

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
      + "tri_scalar,bad_rgb,mangled_layout,mono_scalar,headerless_tiles,refused_view",
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
  check("a body declared interleaved is read as interleaved",
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

  // --- the readout's heading is this view's name, not one inferred from its shape
  //
  // Asserted on a *third* view for a reason. `coef_tiles` and `legacy_tiles`
  // are both 12-channel scalar views, so a heading derived from the payload --
  // `channels === 3 ? "投影面の色" : "12係数タイル"`, say -- names the right
  // view on the first load and on `surface`, and only here is it visibly the
  // wrong one. The name has to come from the spec that rendered.
  // 3つ目のビューで見る理由がある。`coef_tiles` と `legacy_tiles` はどちらも
  // 12チャンネルの scalar なので、ペイロードの形から見出しを決める実装は
  // 初回でも `surface` でも正しい名前を出し、ここでだけ誤りが見える。
  check("the readout's heading names the view that rendered, not its shape",
    el("view-title").textContent === "旧形式タイル", el("view-title").textContent);

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

  // --- and a *missing* X-Layout must be refused too, not assumed
  //
  // Stripping is at least as ordinary as mangling, and it used to land in the
  // silent class: absent read as "interleaved", so a planar body was walked
  // with the wrong stride and the tiles, the readout and the ranges all agreed
  // about the wrong channels. `headerless_tiles` is served planar-shaped with
  // no header, which is exactly what a proxy that drops X-* headers produces.
  // ヘッダの欠落も断ること。欠落は書き換えと同程度にありふれており、以前は
  // 静かな側に落ちていた（欠落＝インタリーブとみなし、planar の本体を誤った
  // ストライドで走査して、タイル・読み取り値・レンジがそろって誤る）。
  // `headerless_tiles` は X-* を落とすプロキシが作るものそのものである。
  const beforeHeaderless = state.payload;
  el("view-select").value = "headerless_tiles";
  el("view-select").__fire("change");
  await sleep(300);
  check("a missing X-Layout is refused rather than assumed to be interleaved",
    /X-Layout/.test(statusText()), statusText());
  check("a missing X-Layout leaves state.payload identity unchanged",
    state.payload === beforeHeaderless,
    `same object: ${state.payload === beforeHeaderless}`);

  // --- a server-side refusal is shown as its message, not as its envelope
  //
  // The one fixture that fails on the *server* rather than at a client guard.
  // The refusal that motivated this -- `surface` on a camera-space matrix --
  // exists precisely so the user is told which matrix to load instead, and
  // that only works if the sentence is legible. Both directions are asserted:
  // the prose appears, and the JSON envelope does not.
  // クライアント側のガードではなくサーバ側で失敗する唯一のフィクスチャ。
  // この拒否（カメラ座標に対する `surface`）は「代わりにどの行列を読めばよいか」
  // を伝えるために存在するので、文が読めなければ意味がない。文が出ることと、
  // JSONの外側が出ないことの両方を確認する。
  const beforeRefused = state.payload;
  // What the readout says about the view on screen, captured before a switch
  // that will not render. Compared rather than named: which view is showing
  // here depends on how many of the switches above were refused, and a literal
  // would make this check fail for every mutation that changes that -- which is
  // a property of those mutations, not of the heading.
  // 拒否される切替の直前に、読み取り欄が名乗っている内容を控える。名前を直接
  // 書かず前後比較にするのは、ここで表示されているビューが「上の切替のうち
  // 幾つが拒否されたか」で変わるためで、直値にすると、その数を変える変異の
  // すべてでこのチェックが落ちる（それは見出しの性質ではない）。
  const titleBeforeRefusal = el("view-title").textContent;
  const panelBeforeRefusal = el("view-panel").hidden;
  el("view-select").value = "refused_view";
  el("view-select").__fire("change");
  await sleep(300);
  check("a server refusal is shown as its detail message",
    statusText().includes("needs a projector-space matrix"), statusText());
  check("a server refusal is not shown as a raw JSON envelope",
    !/\{"detail"|\\"/.test(statusText()), statusText());
  check("a server refusal leaves state.payload identity unchanged",
    state.payload === beforeRefused, `same object: ${state.payload === beforeRefused}`);

  // --- and the readout still describes the view that is actually on screen
  //
  // The heading over `#view-values`, and whether that panel is there at all,
  // are properties of the *rendered* view, written where `state.payload` is
  // set. Nothing asserted that until this check: writing them in the selector's
  // `change` handler instead -- the implementation both the code comment and
  // the harness comment exist to forbid -- passed every check in every
  // scenario, and so did hardcoding the two labels by channel count.
  //
  // The server refused this switch, so nothing about it reached the screen and
  // nothing about it may reach the readout: both must still describe whatever
  // view is still drawn. A selector-time write would be captioning that view's
  // numbers with the name of one that never rendered -- and, since
  // `refused_view` declares no `values_are_matrix`, would also have put the
  // duplicate panel back over them.
  //
  // 見出しと、その板を出すかどうかは、*描かれた*ビューの性質であり、
  // `state.payload` を確定する場所で書く。これを検証するチェックが無かった:
  // セレクタの `change` で書く実装（コード側とハーネス側のコメントがともに
  // 禁じている当のもの）でも、ラベルをチャンネル数で決め打ちする実装でも、
  // 全シナリオが通っていた。この切替はサーバが断ったので画面には何も届いて
  // おらず、読み取り欄にも届いてはならない。セレクタ時に書いていれば、今も
  // 描かれているビューの数値に、描かれなかったビューの名前が付く。
  check("a refused view leaves the readout naming the view still on screen",
    el("view-title").textContent === titleBeforeRefusal
      && el("view-panel").hidden === panelBeforeRefusal
      && titleBeforeRefusal !== "—" && titleBeforeRefusal !== "サーバが断るビュー",
    `title=${titleBeforeRefusal} -> ${el("view-title").textContent} | `
      + `panel hidden=${panelBeforeRefusal} -> ${el("view-panel").hidden}`);
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

  check("staged-but-unsent changes are shown as pending on 適用",
    applyPending(), el("sp-apply").className);

  applyParams();
  await settled();
  check("適用 submits the whole group as exactly one fetch",
    viewRequestCount() === beforeStaging + 1,
    `${beforeStaging} -> ${viewRequestCount()}`);
  const applied = requestLog[requestLog.length - 1];
  check("適用 sends every staged value, not just the last one edited",
    /p_r=0/.test(applied) && /p_g=0/.test(applied) && /p_b=0/.test(applied), applied);
  check("適用 clears the pending marker once the values are sent",
    !applyPending(), el("sp-apply").className);

  // --- Enter applies, and must not leave the UI claiming it did not
  //
  // The value goes out correctly either way; what broke was the *indicator*.
  // The browser fires `keydown` before committing the edit and firing `change`,
  // so Enter ran apply -> clear -> (change) re-arm, and the ring stayed on over
  // a value that had just been applied, with the status line still asking the
  // user to press 適用. Asserted through both: the fetch happened exactly once,
  // and the button and status line agree that it did.
  // Enter でも値は正しく送られる。壊れていたのは表示のほうで、ブラウザは編集を
  // 確定して `change` を出す前に `keydown` を出すため、適用→解除→（change で）
  // 再点灯となり、適用済みの値の上にリングが残って「適用を押せ」と言い続けた。
  // フェッチがちょうど1回であることと、ボタンとステータス行がそれに同意して
  // いることの両方で確認する。
  const beforeEnter = viewRequestCount();
  check("the p_r control is there to type into", pressEnter("p_r", "1"));
  await settled();
  check("Enter in a server-param field submits exactly one fetch",
    viewRequestCount() === beforeEnter + 1,
    `${beforeEnter} -> ${viewRequestCount()}`);
  check("Enter does not leave 適用 claiming the value was never applied",
    !applyPending(), el("sp-apply").className);
  check("Enter does not leave the status line asking for 適用",
    !/適用 を押す/.test(statusText()), statusText());

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

if (SCENARIO === "probe") {
  // The hover readout, once it stopped reading `state.payload`.
  //
  // What it used to show was the *displayed view*'s values under headings that
  // named the matrix: hovering `surface` printed three surface colours beneath
  // `in_R -> out_R`, which is a readout naming one thing and showing another.
  // Every check here is run on `surface` for that reason -- on `coef_tiles` the
  // old behaviour and the new one produce the same twelve numbers, so the whole
  // point of the change is invisible there.
  // `state.payload` を読むのをやめた後のホバー読み取り。以前は「行列の見出しの
  // 下に表示中のビューの値」が出ていた（`surface` をホバーすると
  // `in_R -> out_R` の下に投影面のRGBが並ぶ）。以下を `surface` で回すのは
  // そのためで、`coef_tiles` では新旧どちらも同じ12個を出すため差が見えない。

  // --- the readout has to be compact, and share the strip with everything else
  //
  // Both of these are about space rather than arithmetic, and both were real:
  // twelve cells stretched across a 1600px window put `-0.1234` in the middle
  // of a 500px column, while the display range and the three timings sat at the
  // bottom of a sidebar that had to be scrolled to reach them. The point of the
  // readout is to be read *while* looking at the tiles, so anything that has to
  // be scrolled to, or that pushes the tiles up to make room for whitespace,
  // fails at the only job it has.
  //
  // Asserted about the page as loaded, before any matrix is: neither fact
  // depends on data, and putting them first means a failure here is not
  // confusable with a failure in the point query below.
  //
  // どちらも「場所」の検証であり、どちらも実際に起きていた。12個のセルを
  // 1600px の窓幅いっぱいに引き伸ばすと、`-0.1234` の7文字が500pxの列の
  // 真ん中に浮く。一方で表示レンジと計測はサイドバーの下端にあり、見るには
  // スクロールが要った。読み取り欄はタイルを見ながら読むためのものなので、
  // スクロールしないと届かない・余白のためにタイルを押し上げる、のどちらも
  // 唯一の役目を果たしていない。行列の読み込み前に検証する（どちらもデータに
  // 依存しないし、下の点クエリの失敗と混ざらない）。
  // Every id the strip is made of, not only the four that moved. Listing just
  // the migrated ones left `#probe-grid` and `#view-values` free to be put back
  // in the sidebar with every check green -- and those two are the readout.
  // 移動した4つだけでなく、帯を構成する id を全て見る。移動した分だけを並べる
  // と `#probe-grid` と `#view-values` はサイドバーへ戻せてしまい、しかもその
  // 2つこそが読み取り欄である。
  const DASHBOARD_IDS = [
    "probe-pos", "probe-grid", "view-values",
    "range-info", "perf-fetch", "perf-ranges", "perf-upload", "perf-heap",
  ];
  check("the readout's panels all sit in the dashboard, not the sidebar",
    DASHBOARD_IDS.every((id) => sitsIn(id, "readout") && !sitsIn(id, "sidebar")),
    DASHBOARD_IDS.map(panelOf).join(" | "));

  // The 4x3 is built at startup (buildGrid -> buildProbeGrid), so its column
  // template is already on the element. Fixed lengths only -- `minmax(0, 1fr)`
  // is what spread twelve seven-character cells across the whole window -- and
  // fixed is not the same as compact: `38px repeat(3, 600px)` has no `fr` in it
  // and is 1838px wide, which is the same defect wearing the fix's clothes. The
  // bound is what the panel is for, not a measurement of the current value: the
  // table holds twelve numbers of seven characters and has no business past
  // 300px.
  // 4x3 は起動時に組まれるので列の指定は既に載っている。固定長のみを許すが、
  // 固定であることとコンパクトであることは別で、`38px repeat(3, 600px)` は
  // `fr` を含まないまま1838pxになる（直したふりをした同じ欠陥）。上限は
  // 現在値の写しではなく、この板が何のためにあるかから決める。
  const probeCols = String(el("probe-grid").style.gridTemplateColumns);
  const trackWidthPx = (template) => {
    const flat = template.replace(
      /repeat\(\s*(\d+)\s*,\s*([^)]*)\)/g,
      (_, n, body) => new Array(Number(n)).fill(body).join(" "));
    let total = 0;
    for (const m of flat.matchAll(/([\d.]+)px/g)) total += Number(m[1]);
    return total;
  };
  const probeWidth = trackWidthPx(probeCols);
  check("the readout table's columns are fixed width, not stretched to the window",
    /px/.test(probeCols) && !/fr\b/.test(probeCols) && probeWidth > 0 && probeWidth <= 300,
    `${probeCols} -> ${probeWidth}px`);

  // --- and the strip itself is a row, which is the whole point of the task
  //
  // There is no CSS engine here, so this is asserted about the declaration
  // rather than about a measured box: the stylesheet says `display: flex` for
  // `.readout`, and nothing on the element overrides it inline. Weak on
  // purpose, and stated as such -- what it rules out is the one edit that
  // silently restores the pre-task stacked layout, which is `display: block`
  // in either of the two places it can be written. The row *fitting* on one
  // line at a given width is not checkable from here and is verified by the
  // screenshots instead.
  // CSS エンジンが無いので、測った箱ではなく宣言について検証する。意図的に
  // 弱いチェックであり、排除できるのは「黙って縦積みに戻す1つの編集」だけで
  // ある（`display: block` を書ける2箇所の両方を見る）。実際に1行に収まるか
  // どうかはここからは測れず、スクリーンショットで確認する。
  const readoutRule = cssDeclarations(".readout");
  const readoutBox = ancestors(el("probe-grid")).find((a) => a.classList.contains("readout"));
  check("the readout lays its panels out in a row rather than stacking them",
    /display:\s*flex\b/.test(readoutRule) && !(readoutBox && readoutBox.style.display),
    `sheet: ${readoutRule.trim().replace(/\s+/g, " ")} | `
      + `inline: ${readoutBox ? readoutBox.style.display ?? "(none)" : "no .readout ancestor"}`);

  // --- nothing loaded yet: there is no matrix to ask about
  //
  // The 12 empty tiles exist from startup, so onHover is reachable long before
  // any matrix is. A point query issued now would carry `null` as the matrix
  // id, come back 404, and put that on the status line as the first thing the
  // user ever sees.
  // 起動時点で空のタイルは12枚あるので、行列より先にホバーが起きうる。ここで
  // 点クエリを投げると行列IDが `null` のまま404になり、利用者が最初に見る
  // ものがそのエラーになる。
  views[0].onHover({ col: 1, row: 1 });
  await sleep(40);
  check("hovering with no matrix loaded issues no point query",
    probeLog.length === 0, `probes=${probeLog.length} ${probeLog.join(" | ")}`);

  clickLoad("A");
  await settled();

  // --- the status line says what is on screen and nothing that is already on it
  //
  // It used to end with 「転送 570.8 ms / レンジ 0.3 ms / GPU転送 51.2+0.7 ms」,
  // twenty pixels under a 計測 panel printing two of those three timings
  // verbatim, and to open with dimensions the toolbar also states. Both halves
  // are asserted: the line is the summary alone, and レンジ -- the one figure
  // that lived nowhere else -- is in the panel rather than deleted with the
  // duplicates.
  // ステータス行は画面上の他所にあるものを繰り返さない。両方向を見る（行が
  // 要約だけになったことと、ここにしか無かったレンジがパネルに移った——
  // 重複と一緒に消えたのではない——こと）。
  const renderedStatus = statusText();
  const rangeTiming = el("perf-ranges").textContent;
  check("the status line after a render is the dimensions alone, with レンジ in the 計測 panel",
    renderedStatus === `${W}×${H}×12 を表示中` && /^[\d.]+ ms$/.test(rangeTiming),
    `status="${renderedStatus}" レンジ="${rangeTiming}"`);

  // What the coefficient view does with the second panel, captured before the
  // switch below; the assertion needs both views and is made once, after it.
  // 係数ビューでの2枚目の板の状態。切替の前に控え、両方そろってから見る。
  const panelOnCoefTiles = el("view-panel").hidden;

  el("view-select").value = "surface";
  el("view-select").__fire("change");
  await settled();
  check("the readout is exercised on a view that is not the coefficient tiles",
    state.spec.key === "surface" && views.length === 1,
    `spec=${state.spec.key} tiles=${views.length}`);

  // --- the second panel exists only where the picture is not the matrix
  //
  // The readout table shows the matrix on every view. On `coef_tiles` the panel
  // beside it therefore held the same twelve numbers in a second format --
  // 974px of a 1600px strip, the widest element in a bar built to stop wasting
  // width, and 113px off the tiles at 900px. On `surface` the two panels answer
  // different questions and both are needed.
  //
  // Both directions in one check, deliberately. "Always hidden" is as wrong as
  // "always shown" and would blank the only numbers `surface` produces, and a
  // check that looked at one view alone would pass one of the two.
  // 読み取り表はどのビューでも行列を出す。`coef_tiles` では隣の板が同じ12個の
  // 別書式になっていた（1600pxの帯で974px、900pxではタイルから113px）。
  // `surface` では2枚は別の問いに答えており、両方要る。「常に隠す」は
  // 「常に出す」と同じく誤りで、`surface` の唯一の数値が消える。片方のビュー
  // だけを見るチェックはそのどちらかを通してしまう。
  check("the view-value panel is dropped where it repeats the matrix and kept where it does not",
    panelOnCoefTiles === true && el("view-panel").hidden === false,
    `coef_tiles hidden=${panelOnCoefTiles} surface hidden=${el("view-panel").hidden}`);

  // --- the hover asks the server for that pixel of the matrix
  //
  // (col 3, row 2) on a 5x4 fixture: both orderings are in range, so swapping
  // them does not 400 -- it silently answers for a different pixel. The query
  // string is read back by name rather than by pattern for the same reason.
  // 5x4 のフィクスチャ上の (col 3, row 2)。入れ替えても範囲内なので400にはならず、
  // 黙って別の画素を答える。だからクエリは名前で読む。
  PROBE_OVERRIDES.set(probeKey(2, 3), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
  views[0].onHover({ col: 3, row: 2 });
  await waitFor(() => probeLog.length === 1);
  const asked = new URLSearchParams((probeLog[0] ?? "").split("?")[1] ?? "");
  check("hovering issues a point query for the hovered pixel of the matrix",
    probeLog.length === 1
      && probeLog[0].startsWith("/api/matrix/matrix_A/probe?")
      && asked.get("row") === "2" && asked.get("col") === "3",
    probeLog.join(" | "));

  // --- and all twelve cells are filled from it, row-major
  //
  // 1..12 in the response, so the order the cells are written in is pinned:
  // a column-major render would read 1,4,7,10,2,... and is not a permutation
  // this can miss.
  // 応答が 1..12 なので書き込み順が固定される。列優先なら 1,4,7,10,2,… となり
  // 見逃しようがない。
  await waitFor(() => matrixTexts()[0] !== "—");
  check("the readout fills all twelve cells from the point query, row-major",
    JSON.stringify(matrixTexts())
      === JSON.stringify(Array.from({ length: 12 }, (_, k) => (k + 1).toFixed(4))),
    JSON.stringify(matrixTexts()));

  // --- the view's own values keep a panel of their own
  //
  // Label and value are asserted together: `surface`'s three channels hold
  // 1, 2 and 3, so a panel that paired them the wrong way round -- or listed
  // the right numbers under the matrix's headings again -- is visible here.
  // ラベルと値を同時に見る。`surface` の3チャンネルは 1, 2, 3 なので、対応が
  // ずれた場合も、行列の見出しの下に並べ直した場合も、ここに出る。
  check("the view-value panel lists the payload's channels with their labels",
    JSON.stringify(viewValueLabels()) === JSON.stringify(["R", "G", "B"])
      && JSON.stringify(viewValueNumbers()) === JSON.stringify(expectedProbe(1, 3)),
    `${JSON.stringify(viewValueLabels())} ${JSON.stringify(viewValueNumbers())}`);

  // --- and it says which view those values belong to
  //
  // Two panels of numbers now stand side by side, and only one of them is the
  // matrix. Without a heading that names the view, the right-hand three numbers
  // under `surface` are just three unlabelled numbers next to twelve
  // coefficients -- and the reading a user takes away is that they are part of
  // the same table. The label comes from the spec that was actually rendered,
  // so a refused view switch must not move it.
  // 数値の板が2枚並ぶが、行列はそのうち1枚だけである。ビュー名の見出しが
  // 無ければ、`surface` の3つの数値は12個の係数の隣に並んだ無名の3つでしかなく、
  // 同じ表の一部として読まれる。見出しは実際に描かれたビューの名前であり、
  // 拒否されたビュー切替で動いてはならない。
  // Read defensively only because a *missing* heading is one of the outcomes
  // this is checking for, and a thrown TypeError here would take the rest of
  // the scenario's results down with it.
  // 見出しが無いことも検査対象の結果の1つなので、ここだけは null を許して読む
  // （例外にすると以降のチェック結果ごと失われる）。
  const viewTitle = el("view-title");
  check("the view-value panel names the view whose values it is showing",
    !!viewTitle && viewTitle.textContent === "投影面の色",
    viewTitle ? viewTitle.textContent : "#view-title does not exist");

  // --- a null coefficient is a masked value, not a missing one
  //
  // `null` is what routers/session.py sends for every non-finite coefficient,
  // and on the real matrix roughly three quarters of the frame is outside the
  // active projection region and holds NaN. So this is the ordinary answer for
  // most of the image, not an error path: it has to read as "masked", in the
  // same colour the tiles already paint those pixels, rather than as the "—"
  // that means "nothing has been read yet".
  // 非有限係数は `null` で届き、実機の行列では投影領域外＝全画素の約75%が該当
  // する。つまりこれは例外ではなく大半の画素の通常の答えであり、「まだ何も
  // 読んでいない」を意味する「—」ではなく、タイルと同じマスク色で出なければ
  // ならない。
  //
  // Pinned at (row 1, col 3) rather than on the diagonal: a fixture whose two
  // coordinates are interchangeable is the defect this repository keeps
  // shipping, and (0, 0) would have made this check indifferent to a
  // transposed request.
  // (0,0) ではなく (row 1, col 3) に置く。2つの座標を入れ替えても同じになる
  // フィクスチャこそ、このリポジトリが繰り返してきた欠陥である。
  PROBE_OVERRIDES.set(probeKey(1, 3), [[null, 1, 2], [3, null, 4], [5, 6, null], [7, 8, 9]]);
  views[0].onHover({ col: 3, row: 1 });
  await waitFor(() => matrixTexts()[1] === "1.0000");
  check("a null coefficient renders as a masked NaN cell, not as an empty one",
    JSON.stringify(matrixTexts()) === JSON.stringify(expectedMatrixTexts(1, 3))
      && JSON.stringify(matrixClasses()) === JSON.stringify(
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0].map((n) => (n ? "probe-cell nan" : "probe-cell"))),
    `${JSON.stringify(matrixTexts())} ${JSON.stringify(matrixClasses())}`);

  // --- a pixel outside the active projection region must say so
  //
  // This is the readout's one remaining way of lying. The mask that
  // `coef_tiles.py` and `surface.py` paint is applied *at render time*; the
  // stored array holds finite, meaningless numbers outside the active region.
  // So the tile is grey and the table -- which reads the array through the
  // point query -- prints 13.440 / 1.197 / 0.017 in exactly the type, colour
  // and precision it prints a real coefficient in. Nothing on screen
  // distinguishes them, and the numbers are plausible enough to be written down.
  //
  // Both directions are checked in one place on purpose. A readout that always
  // warns is as useless as one that never does, and "always" passes the half of
  // this that only looks at the masked pixel. The two pixels differ in `row`
  // alone, so the marker cannot be following the column instead.
  //
  // 読み取り欄に残っていた最後の嘘。マスクは描画時に掛かるものであり、配列
  // 自身は有効領域の外にも有限で無意味な値を持つ。したがってタイルは灰色でも、
  // 点クエリ経由で配列を読む表は本物の係数と同じ字体・同じ色・同じ桁数で
  // 13.440 / 1.197 / 0.017 と出す。画面上に区別は無く、しかも書き留めるに
  // 足るほどもっともらしい。常に警告する実装は一度も警告しない実装と同じく
  // 役に立たないので、両方向を1つのチェックで見る。2つの画素は row だけが
  // 違うので、印が列に付いているだけということも起こらない。
  views[0].onHover({ col: 2, row: 0 });                    // outside
  await waitFor(() => matrixTexts()[0] === expectedMatrixTexts(0, 2)[0]);
  const outsideSaid = el("probe-pos").textContent;
  const outsideFlagged = el("probe-pos").classList.contains("outside");
  const outsideCells = matrixTexts();
  views[0].onHover({ col: 2, row: 1 });                    // inside, same column
  await waitFor(() => matrixTexts()[0] === expectedMatrixTexts(1, 2)[0]);
  const insideSaid = el("probe-pos").textContent;
  const insideFlagged = el("probe-pos").classList.contains("outside");
  check("a pixel outside the active region is marked as such, and one inside is not",
    /有効投影領域の外/.test(outsideSaid) && outsideFlagged
      && /col 2, row 0/.test(outsideSaid)
      && JSON.stringify(outsideCells) === JSON.stringify(expectedMatrixTexts(0, 2))
      && !/有効投影領域の外/.test(insideSaid) && !insideFlagged
      && /col 2, row 1/.test(insideSaid),
    `outside=${outsideSaid} flagged=${outsideFlagged} `
      + `cells=${JSON.stringify(outsideCells)} `
      + `inside=${insideSaid} flagged=${insideFlagged}`);

  // --- and the view-value panel marks a masked channel the same way
  //
  // `surface`'s green channel is NaN at (row 3, col 0); see the fake server.
  // Without this the panel's mask class has no execution anywhere and could be
  // deleted with every check still green, leaving a masked channel rendered as
  // an ordinary number in the one panel that reads the payload directly.
  // ビュー値パネル側のマスクも見る。これが無いとパネルのマスク用クラスは
  // どこからも実行されず、消しても全チェックが緑のまま——マスクされた
  // チャンネルが普通の数値として出るようになる。
  views[0].onHover({ col: 0, row: 3 });
  check("the view-value panel marks a masked channel instead of printing it as a number",
    JSON.stringify(viewValueNumbers()) === JSON.stringify(["1.0000", "NaN", "3.0000"])
      && JSON.stringify(viewValueClasses())
        === JSON.stringify(["vv-num", "vv-num nan", "vv-num"]),
    `${JSON.stringify(viewValueNumbers())} ${JSON.stringify(viewValueClasses())}`);
  // That hover also issued a point query, and the coalescing means anything
  // still in flight owns the next request. Let it land before the drag below
  // starts counting, or the request it counts is the one this hover's answer
  // fired -- which is a real property of the client, just not the one being
  // measured there.
  // 直前のホバーも点クエリを出しており、合流の性質上それが着地するまで次の
  // 要求はそれが握る。ドラッグの計測を始める前に着地させる。
  await waitFor(() => matrixTexts()[0] === expectedMatrixTexts(3, 0)[0]);

  // --- a fast drag costs two round trips, not one per mousemove
  //
  // A mousemove fires dozens of times a second and each one would otherwise be
  // a request. Held open, the five hovers below have exactly one request in
  // flight; releasing it must produce exactly one more, for the position the
  // pointer actually ended on -- not for the position it was at when the first
  // request went out, and not one per hover.
  // mousemove は毎秒何十回も飛ぶので、1回ずつ投げれば全部が往復になる。保留した
  // 状態での5回のホバーで飛ぶ要求はちょうど1つ、解いたあとに飛ぶのもちょうど1つ、
  // しかも最終位置のものでなければならない。
  holdProbes = true;
  const base = probeLog.length;
  for (let col = 0; col < 5; col++) views[0].onHover({ col, row: 1 });
  await sleep(60);
  const duringDrag = probeLog.length - base;
  releaseProbe(1, 0);            // the first position -- the only one in flight
  await waitForHeld(1, 4);       // the last position is what fires next
  await sleep(40);
  const afterFirst = probeLog.length - base;
  releaseProbe(1, 4);            // the last position, which waited for it
  await waitFor(() => matrixTexts()[0] === expectedMatrixTexts(1, 4)[0]);
  await sleep(40);
  const afterSecond = probeLog.length - base;
  check("a fast drag costs one point query in flight plus one for the final position",
    duringDrag === 1 && afterFirst === 2 && afterSecond === 2
      && JSON.stringify(matrixTexts()) === JSON.stringify(expectedMatrixTexts(1, 4)),
    `during=${duringDrag} afterFirst=${afterFirst} afterSecond=${afterSecond} `
      + JSON.stringify(matrixTexts()));
  releaseHeldProbes();
  await sleep(40);

  // --- an answer that outlived its claim must not reach the cells
  //
  // A claim can be retired two ways while its answer is still in the air, and
  // one comparison retires both.
  //
  // The pointer moving on is the transient case: `#probe-pos` is written
  // synchronously, so painting a superseded answer puts one pixel's
  // coefficients under another pixel's heading until the newer answer lands.
  //
  // The pointer *leaving* is the permanent one, and it is the worse of the
  // two: the readout is cleared, and an answer arriving afterwards refills all
  // twelve cells beneath a position that reads "—". Nothing follows it, so the
  // table sits there stating twelve coefficients of no stated pixel until
  // something else is hovered.
  //
  // 権利が失効したあとに戻ってくる応答は2通りあり、同じ比較がどちらも捨てる。
  // ポインタが先へ進んだ場合は一時的な誤り（位置の表示は同期的に変わるので、
  // 「ある画素の係数が別の画素の見出しの下に」出る）。ポインタが画像の外へ出た
  // 場合は恒久的な誤りで、こちらのほうが悪い——読み取りは消えているのに、後から
  // 届いた応答が「—」という位置表示の下に12個の数値を並べ直し、次にホバー
  // されるまでそのまま残る。
  const beforeStale = matrixTexts();
  holdProbes = true;
  views[0].onHover({ col: 1, row: 0 });
  await waitForHeld(0, 1);
  views[0].onHover({ col: 4, row: 3 });
  await sleep(40);
  releaseProbe(0, 1);            // answers for a pixel the pointer has left
  await waitForHeld(3, 4);
  await sleep(40);
  const midStale = matrixTexts();
  releaseProbe(3, 4);
  await waitFor(() => matrixTexts()[0] === expectedMatrixTexts(3, 4)[0]);
  const afterStale = matrixTexts();

  // ...and the same again, retired by the pointer leaving the image (GLView
  // calls onHover(null) on pointer-out) rather than by a newer position.
  views[0].onHover({ col: 2, row: 2 });
  await waitForHeld(2, 2);
  views[0].onHover(null);
  const clearedPos = el("probe-pos").textContent;
  const cleared = matrixTexts();
  releaseProbe(2, 2);
  await sleep(80);
  const afterLeave = matrixTexts();
  holdProbes = false;
  releaseHeldProbes();

  const dashes = JSON.stringify(new Array(12).fill("—"));
  check("a point query answered after its claim was retired never reaches the cells",
    JSON.stringify(midStale) === JSON.stringify(beforeStale)
      && JSON.stringify(afterStale) === JSON.stringify(expectedMatrixTexts(3, 4))
      && clearedPos === "—" && JSON.stringify(cleared) === dashes
      && JSON.stringify(afterLeave) === dashes,
    `before=${JSON.stringify(beforeStale)} mid=${JSON.stringify(midStale)} `
      + `after=${JSON.stringify(afterStale)} pos=${clearedPos} `
      + `cleared=${JSON.stringify(cleared)} afterLeave=${JSON.stringify(afterLeave)}`);

  // --- a refused point query blanks the cells, says why, and hands the line back
  //
  // The endpoint refuses an out-of-range pixel with a 400 and an unknown id
  // with a 404. Neither is reachable by dragging a mouse today -- GLView's
  // screenToImage rejects coordinates outside the image, and both registered
  // views render at the matrix's own size -- which is exactly why the path
  // needs a test rather than exempts it: it is new code that nothing else
  // executes, and "the table wedged on the last good values" looks identical
  // to "the table is working" from any screenshot.
  //
  // The recovery half matters as much as the failure half. The status bar is
  // shared with the loader, so a readout that stops writing to it but never
  // puts it back leaves the whole viewer marked as failed over a hover that
  // has since succeeded -- and once a matrix is loaded and the user is only
  // hovering, nothing else ever writes there.
  //
  // 範囲外の画素は400、未知のIDは404で断られる。今日どちらもマウス操作では
  // 到達できない（screenToImage が画像外を弾き、両ビューとも行列と同じ寸法で
  // 描かれる）——だからこそ検証が要る。誰も実行しない新しい経路であり、しかも
  // 「直前の値のまま固まった表」は画面写真では「動いている表」と見分けが付かない。
  // 復旧側も同じくらい重要で、ステータス行は読み込みと共用のため、書くのを
  // やめるだけで戻さないと、既に成功しているホバーについてビューア全体が
  // 失敗表示のまま残る（ホバーしているだけなら他に誰も書かない）。
  views[0].onHover({ col: 3, row: 2 });
  await waitFor(() => matrixTexts()[0] !== "—");
  const beforeRefusal = matrixTexts();
  views[0].onHover({ col: 99, row: 99 });         // outside the matrix -> 400
  await waitFor(() => /読み取りに失敗/.test(statusText()));
  const refusedTexts = matrixTexts();
  const refusedFlagged = el("status").classList.contains("error");
  const refusedSaid = statusText();
  views[0].onHover({ col: 3, row: 2 });
  await waitFor(() => matrixTexts()[0] !== "—");
  check("a refused point query blanks the cells, says why, and gives the line back on recovery",
    JSON.stringify(beforeRefusal) !== dashes
      && JSON.stringify(refusedTexts) === dashes
      && refusedFlagged && /outside the matrix/.test(refusedSaid)
      && !/読み取りに失敗/.test(statusText())
      && !el("status").classList.contains("error"),
    `before=${JSON.stringify(beforeRefusal)} refused=${JSON.stringify(refusedTexts)} `
      + `flagged=${refusedFlagged} said=${refusedSaid} `
      + `recovered=${statusText()} class=${el("status").className}`);

  // --- the hover series must not be the view series
  //
  // This is the one property the brief states in bold, and until this check
  // existed nothing asserted it: merging `probeSeq` into `state.reqId` passed
  // all 76 checks in all five scenarios, because every other hover in this
  // harness happens after `settled()` and therefore never overlaps a transfer.
  //
  // The damage is silent and total. `reload()` drops its payload when
  // `isStale(req)`, so one mousemove during a transfer retires the transfer
  // that is still arriving: the tiles stay on the previous matrix, the status
  // line stays on 「ビュー取得中…」 forever, and nothing on screen says why.
  // On the real matrix that transfer takes seconds and the pointer is over
  // the image the whole time, so this is not a corner -- it is every load.
  //
  // ブリーフが太字で述べている唯一の性質であり、このチェックができるまで誰も
  // 検証していなかった。`probeSeq` を `state.reqId` に統合しても5シナリオ76件が
  // すべて通ってしまう（他のホバーはすべて `settled()` のあとで、転送と重ならない
  // ため）。壊れ方は静かで、しかも致命的である。`reload()` は `isStale(req)` で
  // ペイロードを捨てるので、転送中の mousemove 1回がその転送を失効させる——
  // タイルは前の行列のまま、ステータス行は「ビュー取得中…」のまま、理由は
  // どこにも出ない。実機では転送に数秒かかり、その間ポインタは画像の上にある。
  viewDelayMs.B = 200;
  clickLoad("B");
  await sleep(25);
  check("the view transfer was still in flight when the hover happened",
    !resolved.has("view:B"), `resolvedB=${resolved.has("view:B")}`);
  for (let col = 0; col < 4; col++) views[0].onHover({ col, row: 2 });
  await settled();
  check("hovering during a view transfer does not stall the tiles",
    payloadTag() === 2 && /表示中/.test(statusText()),
    `payload tag=${payloadTag()} (A=1, B=2) status=${statusText()}`);
}

if (!checks.length) check(`unknown scenario ${SCENARIO}`, false);
emit();
process.exit(0);
