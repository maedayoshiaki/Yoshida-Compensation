// Drives the shipped viewer frontend under a stub DOM/WebGL, for pytest.
//
// Usage: node frontend_harness.mjs <static-dir> <scenario>
//   <static-dir>  directory holding index.html and js/{main,glview,api,plot}.js
//   <scenario>    "ordering" | "views" | "button" | "rgb" | "gamut" | "probe"
//                 | "target" | "lut"
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
//
// The GL *constants* are the third thing not swallowed, and they had to stop
// being swallowed for the texture format to be checkable at all. Every
// SCREAMING_CASE property used to answer the same `1`, so `gl.R32F` and
// `gl.RGB32F` were the same value and `texImage2D` could not tell which had
// been asked for: a categorical view uploading its class indices as RGB32F --
// which puts the index in the red component alone and leaves green and blue
// undefined -- was indistinguishable from the R32F upload it needs. Each name
// now gets its own id, and the id maps back to the name so a check can say
// "RGB32F" rather than "4098".
// GL定数も飲み込まない。以前はどの定数も同じ `1` だったため `gl.R32F` と
// `gl.RGB32F` が区別できず、クラス番号を RGB32F で載せる実装（赤成分だけに
// 番号が入り、緑と青は未定義）が正しい R32F の転送と同じに見えていた。
const GL_CONSTANT_IDS = new Map();    // name -> id
const GL_CONSTANT_NAMES = new Map();  // id -> name
let nextGlConstant = 0x4000;

function glConstant(name) {
  if (!GL_CONSTANT_IDS.has(name)) {
    const id = nextGlConstant++;
    GL_CONSTANT_IDS.set(name, id);
    GL_CONSTANT_NAMES.set(id, name);
  }
  return GL_CONSTANT_IDS.get(name);
}

const glConstantName = (id) => GL_CONSTANT_NAMES.get(id) ?? id;

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
        // (target, level, internalformat, w, h, border, format, type, pixels)
        return (...args) => {
          const px = args[args.length - 1];
          uploads.push({
            length: px.length,
            first: px[0],
            last: px[px.length - 1],
            head: Array.from(px.slice(0, 6)),
            internal: glConstantName(args[2]),
            format: glConstantName(args[6]),
          });
        };
      }
      if (typeof prop === "string" && /^[A-Z0-9_]+$/.test(prop)) return glConstant(prop);
      return glStub(uploads, uniforms);
    },
    apply() { return glStub(uploads, uniforms); },
  });
}

// -------------------------------------------------------------- 2D ctx stub
// Records every call and every property assignment, in order.
//
// The GL stub above swallows all but three calls, because what a WebGL upload
// *means* is visible in `texImage2D`'s last argument alone. A 2D canvas is the
// opposite: the whole output of js/plot.js is the sequence of `moveTo`/`lineTo`
// coordinates and the `strokeStyle` in force when `stroke()` is called. A stub
// that dropped them would leave "three lines were drawn" as the only checkable
// property -- and three lines at identical coordinates, or drawn for the wrong
// channels, satisfy that just as well as the right ones do. The brief for this
// task names that failure explicitly: counting lines cannot detect that they
// were drawn in the wrong place, so the stub records where.
//
// Property *assignments* are recorded too, and interleaved with the calls in
// one list rather than kept as final values, because `strokeStyle` is set once
// per curve: only the order tells which colour belonged to which polyline.
//
// GL側は3つを除いて飲み込むが、2Dは逆で、js/plot.js の出力そのものが
// `moveTo`/`lineTo` の座標列と `stroke()` 時点の `strokeStyle` である。落とすと
// 検証できるのは「3本描かれた」だけになり、同じ座標に3本でも、チャンネルを
// 取り違えた3本でも通ってしまう。プロパティ代入も同じ列に順序どおり記録する
// （`strokeStyle` は曲線ごとに設定されるので、順序だけがどの色がどの折れ線の
// ものかを決める）。
function ctx2dStub(calls) {
  const properties = {};
  return new Proxy(properties, {
    get(target, prop) {
      if (prop === "__calls") return calls;
      if (Object.prototype.hasOwnProperty.call(target, prop)) return target[prop];
      return (...args) => { calls.push([String(prop), ...args]); };
    },
    set(target, prop, value) {
      target[prop] = value;
      calls.push([`set:${String(prop)}`, value]);
      return true;
    },
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
    // One context per element *per type*, kept, so each tile's uploads
    // accumulate on its own GLView rather than on a fresh proxy per call --
    // and so the LUT canvas's 2D calls accumulate across the redraws a scenario
    // triggers (load, overlay toggle, DPR change) instead of being lost.
    //
    // Branching on the type is what makes the 2D canvas testable at all: the
    // GL proxy answers every property with another proxy, so `getContext("2d")`
    // used to hand js/plot.js something that accepted `moveTo`/`stroke` and
    // recorded nothing. Nothing would have thrown; the plot would simply have
    // been invisible from here.
    // 要素ごと・種類ごとに1つのコンテキストを保持する。種類で分けないと
    // `getContext("2d")` がGLのプロキシを返し、`moveTo`/`stroke` を受け取って
    // 何も記録しない（例外も出ないので、図はただ外から見えなくなる）。
    getContext(type) {
      if (String(type) === "2d") {
        el.__ctx2dCalls ||= [];
        return (el.__ctx2d ||= ctx2dStub(el.__ctx2dCalls));
      }
      return (el.__gl ||= glStub([], {}));
    },
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
// Deliberately minimal: it records the tag, the `class`, the `id`, `hidden`,
// and the nesting. Not `value`, and no other boolean attribute -- every one of
// those would change what the *existing* scenarios see at startup, and a stub
// that starts simulating a browser more broadly than the checks need is a new
// source of false answers rather than a better fixture.
//
// `hidden` is the exception, and it was added because leaving it out made a
// check unable to fail. Every element starts `hidden: false` in makeEl, so
// `class-legend is on screen for a categorical view` passed against a page
// whose legend does not exist and against one that declares `hidden` -- the
// only way it could fail was for code to actively set it true. Three elements
// in index.html carry the attribute (`#space-note`, `#target-images`,
// `#class-legend`) and all three are hidden-until-something-happens, which is
// a real initial state and not a detail of presentation.
// `hidden` だけは例外として記録する。これが無いと全要素が `hidden: false` で
// 始まるため、「凡例が出ている」の検査はコードが明示的に true を書いたときに
// しか落ちなかった。index.html で `hidden` を持つ3要素はいずれも「何かが
// 起きるまで隠れている」状態であり、体裁ではなく初期状態である。
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
    // Presence is the value, as in HTML: bare `hidden` and `hidden=""` both
    // mean hidden. Quoted runs are removed before looking for the bare token,
    // so a *value* containing the word (`placeholder="a hidden thing"`) is not
    // read as the attribute. See the note above for why this one alone is
    // modelled.
    // 属性値を先に取り除いてから裸のトークンを探す（値の中の "hidden" を属性と
    // 読まないため）。
    if ("hidden" in attrs
      || /(?:^|\s)hidden(?=[\s/=>]|$)/.test(rawAttrs.replace(/"[^"]*"|'[^']*'/g, ""))) {
      node.hidden = true;
    }
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
  // One channel of class indices, four class labels: the `categorical`
  // contract routers/view.py pins (channels == 1, len(labels) >= 2,
  // vmax == len(labels) - 1). It is registered with its real shape rather
  // than a convenient one because every categorical check here reads
  // `state.payload.labels`, and a fixture whose label count matched its
  // channel count would let the two axes be confused with no visible effect.
  // クラス番号1チャンネルとクラス名4つ。ラベル数とチャンネル数が一致する
  // フィクスチャにすると、2つの軸を取り違えても見た目に差が出なくなる。
  //
  // Its parameters mirror the shipped `views/gamut.py`: `target`,
  // `target_decode` and `target_gamma`, and **`linearizer` omitted** -- which
  // is the only difference between it and `compensation` that matters to the
  // 生成 decision. With `params: []` the fixture was symmetric in the dimension
  // under test: a refusal keyed on how many parameters a view declares, or one
  // that special-cased "any view that is not `compensation`", would pass here
  // and diverge against the real registry, where `gamut` declares three of the
  // four fields the request body needs. The refusal check reads the sentence,
  // which names the missing parameters, so it now has to name `linearizer`
  // alone -- against `params: []` it said `target, linearizer`.
  //
  // パラメータは出荷側の `views/gamut.py` を写す（`linearizer` だけが無い）。
  // `params: []` では「宣言数で断る実装」や「compensation 以外は断る実装」と
  // 区別が付かず、検証したい次元についてフィクスチャが対称になっていた。
  {
    key: "gamut", label: "色域外判定", kind: "categorical", channels: 1,
    params: [
      { name: "target", label: "ターゲットID", type: "text", default: "", choices: [] },
      {
        name: "target_decode", label: "ターゲット復号", type: "select",
        default: "srgb", choices: ["none", "srgb", "gamma"],
      },
      { name: "target_gamma", label: "ターゲットγ", type: "number", default: 2.2 },
    ],
  },
  // A second `categorical` view, and its only job is to have a class count that
  // is **not** the palette's length.
  //
  // `gamut` declares four classes and `CLASS_COLORS` has four entries, so on
  // that fixture alone `legendRows().length === payload.labels.length` is
  // `4 === 4` whichever array `renderClassLegend` loops over. Driving the legend
  // off `CLASS_COLORS.length` -- the palette instead of the server's answer --
  // left the whole `gamut` scenario green (measured). That is the same symmetry
  // the `gamut` fixture's own note warns about one axis over (labels vs
  // channels), and this closes the other axis.
  //
  // Three classes, not four, and not two: `routers/view.py` admits any
  // categorical view at `len(labels) >= 2` and says in its own comment that the
  // consumer which will violate the four-class assumption is "the second
  // categorical view". This is that view, and it is the first time the client
  // side of that clause is exercised at all. A legend built from the palette
  // draws a fourth row here whose label is the string `undefined` beside a
  // colour no pixel carries.
  //
  // 2つ目の `categorical` ビュー。役目は「クラス数がパレットの長さと違うこと」
  // だけである。`gamut` は4クラスで `CLASS_COLORS` も4要素なので、凡例の行数を
  // どちらから取っても `4 === 4` になり、パレット側から作る実装でも通ってしまう
  // （実測で確認）。`routers/view.py` は `len(labels) >= 2` を認めており、その
  // コメント自身が「破るのは2つ目の categorical ビューだ」と書いている。
  {
    key: "gamut_lite", label: "3クラスの色域外", kind: "categorical", channels: 1,
    values_are_matrix: false, params: [],
  },
  // A kind no frontend implements, and the only reason it exists.
  //
  // `gamut` used to be this fixture -- `categorical` was the unimplemented
  // kind -- and the check that the viewer refuses what it does not understand
  // rode on it. Implementing `categorical` would have taken that check with
  // it: the scenario would have gone on asserting a refusal that no longer
  // happens, or (worse) been quietly repointed at a kind the viewer now draws
  // and stopped asserting anything. The property is about *unknown* kinds and
  // it outlives any particular one, so it gets a fixture that will never be
  // implemented rather than borrowing whichever kind is next in line.
  //
  // 未対応 kind のフィクスチャ。以前はこれが `gamut`（= `categorical`）で
  // あり、実装した時点でこの検査は行き場を失う。「未知の kind を断る」性質は
  // 特定の kind より長生きするので、実装される予定の無いものを充てる。
  { key: "vector_field", label: "未対応kindのビュー", kind: "flow", channels: 2, params: [] },
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
  // The only view with a `text` parameter, and the reason it is registered.
  //
  // `buildServerParams`' `else` branch -- the one that builds
  // `<input type="text">` -- had no execution anywhere until `compensation`
  // (plan B3 Task 3) declared the first `type: "text"` ParamSpec, for a target
  // id that only exists at runtime and so cannot be a static `choices`. Every
  // earlier param was `number` (`surface`'s three) or `select` (`normalize`),
  // both of which take other branches.
  //
  // That branch is now the *only* way to reach the view at all: without a
  // usable text input there is no way to name a target, and the server refuses
  // every request with "needs a target image". Narrowing the branch -- to
  // `"hidden"`, or to an `else if` that leaves unknown types with no element --
  // still passes every Python test and every other check here, while the
  // ターゲットID box silently disappears from the page.
  //
  // Same standard as `legacy_tiles` above: a fourth kind of parameter gets its
  // own registration rather than being bolted onto `surface`, whose three
  // number controls are what its own staging checks count.
  //
  // `text` パラメータを持つ唯一のビュー。`buildServerParams` の
  // `<input type="text">` を作る `else` 枝は、`compensation`（計画B3 Task 3）が
  // 最初の `type: "text"` を宣言するまでどこからも実行されていなかった
  // （それ以前は number か select のみ）。しかもこの枝は今やビューへ到達する
  // 唯一の道である——テキスト入力が使えなければターゲットidを指定できず、
  // サーバは全要求を「ターゲットが要る」と断る。枝を狭めても Python 側も
  // 他のチェックも全て通り、ターゲットID欄だけが黙って消える。
  {
    key: "compensation", label: "補償画像", kind: "rgb", channels: 3,
    values_are_matrix: false,
    // Mirrors the shipped `views/compensation.py`'s six parameters, and the
    // *defaults* are the shipped ones verbatim (`""`, `"none"`, `""`,
    // `"linear"`, `"srgb"`, `2.2`) rather than convenient ones.
    //
    // That matters for one scenario only, and it matters a great deal there.
    // `generate` asserts the *contents* of the body the 生成 button sends, and
    // an expected value that coincides with a declared default cannot tell a
    // body built from the controls from one that hard-coded the default -- the
    // third of the three shapes of unfailable test this repository keeps
    // shipping. So the defaults here have to be the real ones, and that
    // scenario has to move every field off them before it presses the button.
    //
    // 出荷側の6つのパラメータを写す。既定値は「都合のよい値」ではなく出荷側
    // そのもの（`""`/`"none"`/`""`/`"linear"`/`"srgb"`/`2.2`）にする。
    // `generate` シナリオは送られる body の中身を検証するが、期待値が既定値と
    // 一致していると「コントロールから組み立てた実装」と「既定値を直書きした
    // 実装」を区別できない。だから既定値は本物でなければならず、あちらは
    // 押す前に全項目を既定値から動かす。
    params: [
      { name: "target", label: "ターゲットID", type: "text", default: "", choices: [] },
      {
        name: "linearizer", label: "線形化", type: "select",
        default: "none", choices: ["none", "gamma", "inverse_lut"],
      },
      // The second runtime id, and the reason it is declared here as well as
      // `target`: the two sidebar loaders write into different boxes through
      // the same `setServerParam` path, and a build that filled one and not the
      // other would still pass every check that only looks at `target`.
      // 2つ目の実行時ID。読み込み欄は2つあり、同じ経路で別々の欄へ書く。
      // 片方だけ埋める実装でも `target` しか見ないチェックは通ってしまう。
      { name: "lut", label: "LUT ID", type: "text", default: "", choices: [] },
      {
        name: "output", label: "出力", type: "select",
        default: "linear", choices: ["linear", "code"],
      },
      {
        name: "target_decode", label: "ターゲット復号", type: "select",
        default: "srgb", choices: ["none", "srgb", "gamma"],
      },
      { name: "target_gamma", label: "ターゲットγ", type: "number", default: 2.2 },
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
  // Deliberately *not* views/gamut.py's own LABELS
  // (["範囲内", "負クリップ", "1超クリップ", "両方"]), and deliberately not in
  // its order. A frontend that hard-coded the shipped class names -- the
  // obvious way to build a legend, and the one the brief for this task names
  // as a mistake -- would produce a legend indistinguishable from a correct
  // one against a fixture that echoed them. The comma inside the last entry is
  // there so a legend that joined the labels into one string, or split them on
  // one, is visibly wrong rather than coincidentally right.
  // 実装が本物のラベルを埋め込んでも気付けるよう、意図的に別の語・別の順に
  // する。最後の要素にカンマを入れてあるのは、連結や分割で作った凡例が偶然
  // 正しく見えないようにするためである。
  gamut: ["色域外", "1超クリップ", "両方", "範囲内, 補正可"],
  // Three, against a four-entry palette. See the registration for why the count
  // itself is the fixture.
  gamut_lite: ["範囲内", "クリップ", "答えなし"],
  vector_field: ["u", "v"],
  surface: ["R", "G", "B"],
  compensation: ["R", "G", "B"],
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
// ------------------------------------------------------- targets and LUTs
//
// The two runtime-id endpoints the sidebar loaders drive.
//
// Ids are derived from *what was asked for*, never fixed strings: a client that
// invented an id, echoed the path, or reused the previous response would
// produce something that does not match `target_<key>_<index>`, and a client
// that re-posted with the wrong index would produce a real id for the wrong
// image. A constant id -- "target1" for everything -- would make all of that
// indistinguishable.
//
// 発行するIDは「何を要求されたか」から作る。固定文字列にすると、IDを勝手に
// 作る実装・パスを流用する実装・前の応答を使い回す実装が全て同じに見える。
const TARGETS = {
  "data/one.png": {
    key: "one", is_directory: false, images: ["one.png"], shapes: [[3, 5, 3]],
  },
  // Three images with three different sizes, so "which one is loaded" is
  // visible in `meta.shape` and not only in an opaque id.
  // 3枚とも大きさを変える（どれを読んだかが shape にも出る）。
  "data/dir": {
    key: "dir", is_directory: true,
    images: ["a.png", "b.png", "c.png"],
    shapes: [[3, 5, 3], [7, 4, 3], [2, 9, 3]],
  },
};

// The `detail` a missing target comes back with. Carries a quote, so a client
// that rendered the raw JSON envelope instead of the sentence would show the
// backslash escape -- the same trick REFUSAL_DETAIL uses above.
// 404 の説明文。引用符を含めてあるので、JSONの外側ごと出す実装は
// バックスラッシュのエスケープが見えて区別できる。
const TARGET_404_DETAIL = 'no image at "data/missing.png"';
const LUT_404_DETAIL = 'no inverse LUT at "data/missing.npz"';
// The failure of the *second* request of a load, after the first has already
// succeeded. Also quoted, for the same reason as the two above.
const LUT_CURVE_FAILURE_DETAIL = 'could not read the curve for "lut_curveless"';

// The LUT fixture: three curves with three different shapes and three
// different endpoints, none of them 1.0.
//
// Three curves of the same shape cannot detect a channel swap -- the failure
// the brief for this task names first -- so R is a straight line, G bulges up,
// and B bulges down and stops at 0.688, the largest of the three. That 0.688
// is what the plot's vertical axis has to come from: a LUT ending at 1.0 makes
// an axis pinned to 1.0 indistinguishable from one read off the data, and a
// measured LUT does *not* end at 1.0 (the ISO400 file in this repository stops
// at 0.813 / 0.876 / 0.751).
//
// 3本とも同じ形だとチャンネルの入れ替えが見えない。R は直線、G は上に凸、
// B は下に凸で 0.688（3本の最大値）で止まる。1.0 で終わるフィクスチャでは、
// 軸を 1.0 に固定した実装とデータ由来の実装が区別できない。実測LUTの終端は
// 1.0 ではない（このリポジトリのISO400ファイルは 0.813/0.876/0.751）。
const LUT_T_GRID = [0, 0.2, 0.5, 0.9, 1];
const LUT_ROWS = [
  [0, 0.11, 0.275, 0.495, 0.55],    // R: straight, ends at 0.55
  [0, 0.22, 0.311, 0.417, 0.44],    // G: concave,  ends at 0.44
  [0, 0.028, 0.172, 0.557, 0.688],  // B: convex,   ends at 0.688
];
const LUT_META = { source: "channel_ramp", trust_low: 5, trust_high: 245 };
// Not 2.2: the default would be indistinguishable from a frontend that
// hard-coded the usual exponent instead of reading the one the server sent.
// 2.2 にしない（既定値だと、サーバの値を読まず定番の指数を書いた実装と
// 区別が付かない）。
const LUT_GAMMA = 1.9;
const LUTS = {
  "data/inverse_lut.npz": { key: "measured", rows: LUT_ROWS },
  // A LUT whose POST succeeds and whose curve fetch does not.
  //
  // `loadInverseLut` publishes `state.lutId` and the LUT ID box only *after*
  // `GET /api/lut/{id}/curve` has returned, and said so in a comment that
  // nothing held it to: with only the entry above, a registered id always had
  // a curve and an unregistered path failed at the POST, so the two orders
  // were indistinguishable. Moving both assignments above the curve fetch left
  // all twelve checks in the `lut` scenario passing (measured).
  //
  // The state is not exotic -- the server restarted between the two calls, a
  // proxy dropped the second, validation was added to the curve endpoint
  // later. What the ordering prevents is the LUT ID box and `state.lutId`
  // naming LUT B while the canvas still shows LUT A's curves: press 適用 and
  // the compensation is computed from a LUT the plot beside it does not
  // describe.
  //
  // POST は成功するが curve の取得だけが失敗するLUT。`loadInverseLut` が
  // 「カーブが手元に来てからIDを公開する」ことを述べていたのはコメントだけで、
  // この状態を作れるフィクスチャが無かったため何も検証されていなかった
  // （順序を入れ替えても12件全部通ることを実測）。順序が逆だと、欄と
  // `state.lutId` は新しいLUTを名乗り canvas には前のLUTのカーブが残る。
  "data/curveless.npz": { key: "curveless", rows: LUT_ROWS, curveStatus: 500 },
};

// ------------------------------------------------------------ generation
//
// `POST /api/generate` is the one endpoint in this application that writes to
// the user's disk, and the only one whose *request body* is the thing under
// test: the pixels it saves are decided entirely by the ids and the settings
// the browser puts in it, so a body that does not describe the picture on
// screen saves a file the user was never shown. Every request is therefore
// recorded whole (`generatePostLog`) rather than counted, and the checks
// compare fields, not call counts.
//
// The three fixture directories are the three answers the real endpoint gives
// (see `routers/generate.py` and `generate.py`), because each one drives a
// different piece of the UI: a directory it can write (200), one that already
// holds files (409, and *nothing written* -- the frontend has to ask before
// resending), and one it cannot write at all (400, prose).
//
// このアプリで唯一ディスクへ書くエンドポイントであり、唯一「リクエストの本文
// そのもの」が検証対象であるもの。保存される画素は本文の設定だけで決まるので、
// 画面と違う本文は「見ていないものが保存される」ことを意味する。よって回数では
// なく本文を丸ごと記録し、フィールド単位で突き合わせる。3つのディレクトリは
// 実物が返す3種類の答え（200 / 409 / 400）に対応する。
const generatePostLog = [];   // every POST /api/generate body, in order

// Holds the response so a second press can happen while the first is in
// flight. The double-submit guard is unobservable against a fetch that
// settles immediately -- the button would be disabled and released again
// before any test could look.
// 生成中に2回目を押せるよう応答を保留する。即座に解決する fetch では
// 二重送信のガードは観測できない（押し直す前に解放されてしまう）。
let holdGenerate = false;
const heldGenerates = [];
const releaseGenerates = () => { while (heldGenerates.length) heldGenerates.shift()(); };

// A directory that already holds files. `generate.py` refuses these with 409
// and writes nothing at all -- not even the directory.
const GENERATE_NON_EMPTY = new Set(["data/existing"]);
// A path the OS will not accept. The real endpoint turns the `OSError` into a
// 400 whose detail names the path.
const GENERATE_UNWRITABLE = new Set(["Z:/nope"]);

// What the server resolves a relative `output_dir` to.
//
// Deliberately not the string that was sent. `write_projection_images` calls
// `.resolve()` before anything else and every path it reports -- the response,
// `files`, and the refusal message -- is absolute, precisely so that a refusal
// names a place the user can find. A frontend that displayed the box's own
// contents back would look identical against a fixture that echoed the input.
// 送られた文字列をそのまま返さない。実物は必ず絶対パスへ解決してから答える
// （断り文が指す場所が利用者に分かるようにするため）。入力を反響するだけの
// フィクスチャでは「入力欄を出しているだけの実装」と区別が付かない。
const generateAbsolute = (dir) => `C:/repo/${dir}`;
// The three files the endpoint reports, in the order it returns them: the
// linear compensation image, the projector image, and the metadata.
const generateFiles = (dir) => [
  `${generateAbsolute(dir)}/compensation_linear/compensation_image_02.png`,
  `${generateAbsolute(dir)}/compensation_projector/inv_lut_comp_image_02.png`,
  `${generateAbsolute(dir)}/generate_metadata.json`,
];
// Quoted, like the other refusal details here, so a client that rendered the
// JSON envelope instead of the sentence shows the backslash escape.
const generate409Detail = (dir) =>
  `${generateAbsolute(dir)} is not empty (contains "compensation_linear"). `
  + "Refusing to write into a directory that already holds files.";
const generate400Detail = (dir) =>
  `could not write to "${dir}": [WinError 3] the system cannot find the path`;

const targetPostLog = [];   // every POST /api/target body, in order
const lutPostLog = [];      // every POST /api/lut body, in order
const lutCurveLog = [];     // every GET /api/lut/{id}/curve url, in order

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

  // POST /api/target, answered the way routers/target.py does: an id plus the
  // meta the sidebar's image list and info line are built from.
  if (url === "/api/target") {
    const body = JSON.parse(options.body);
    targetPostLog.push(body);
    const entry = TARGETS[body.path];
    if (!entry) {
      return new Response(JSON.stringify({ detail: TARGET_404_DETAIL }), {
        status: 404, headers: { "content-type": "application/json" },
      });
    }
    const index = Number(body.index ?? 0);
    if (!(index >= 0 && index < entry.images.length)) {
      return new Response(JSON.stringify({
        detail: `index ${index} is out of range for ${body.path}`,
      }), { status: 400, headers: { "content-type": "application/json" } });
    }
    const id = `target_${entry.key}_${index}`;
    return new Response(JSON.stringify({
      id,
      meta: {
        id, shape: entry.shapes[index], dtype: "uint8", path: body.path,
        is_directory: entry.is_directory, images: entry.images, index,
      },
    }), { headers: { "content-type": "application/json" } });
  }

  // POST /api/lut, answered the way routers/lut.py does.
  if (url === "/api/lut") {
    const body = JSON.parse(options.body);
    lutPostLog.push(body);
    const entry = LUTS[body.path];
    if (!entry) {
      return new Response(JSON.stringify({ detail: LUT_404_DETAIL }), {
        status: 404, headers: { "content-type": "application/json" },
      });
    }
    const id = `lut_${entry.key}`;
    return new Response(JSON.stringify({
      id,
      meta: {
        id, path: body.path, dtype: "float32",
        shape: [entry.rows.length, LUT_T_GRID.length], meta: LUT_META,
      },
    }), { headers: { "content-type": "application/json" } });
  }

  // POST /api/generate, answered the way routers/generate.py does.
  //
  // The body is recorded *before* the hold, so a test can see that a request
  // was issued while it is still in flight -- which is the whole of what the
  // double-submit check needs to observe.
  // 保留の前に本文を記録する。「飛んでいる最中に発行済みである」ことが
  // 二重送信の検査が見たい唯一の状態だからである。
  if (url === "/api/generate") {
    const body = JSON.parse(options.body);
    generatePostLog.push(body);
    if (holdGenerate) await new Promise((release) => heldGenerates.push(release));
    const dir = String(body.output_dir ?? "");
    if (GENERATE_UNWRITABLE.has(dir)) {
      return new Response(JSON.stringify({ detail: generate400Detail(dir) }), {
        status: 400, headers: { "content-type": "application/json" },
      });
    }
    // Only an explicit `true` proceeds, exactly as `generate.py` has it: the
    // 409 is not "try again", it is "nothing has been written and someone has
    // to say so on purpose".
    if (GENERATE_NON_EMPTY.has(dir) && body.overwrite !== true) {
      return new Response(JSON.stringify({ detail: generate409Detail(dir) }), {
        status: 409, headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({
      output_dir: generateAbsolute(dir),
      files: generateFiles(dir),
      meta: { linearizer: body.linearizer, target_decode: body.target_decode },
    }), { headers: { "content-type": "application/json" } });
  }

  const lutCurve = url.match(/^\/api\/lut\/([^/?]+)\/curve$/);
  if (lutCurve) {
    lutCurveLog.push(url);
    const entry = Object.values(LUTS).find((e) => `lut_${e.key}` === lutCurve[1]);
    if (!entry) {
      return new Response(JSON.stringify({ detail: `lut not found: ${lutCurve[1]}` }), {
        status: 404, headers: { "content-type": "application/json" },
      });
    }
    // A registered LUT whose curve cannot be produced -- see `LUTS`.
    if (entry.curveStatus) {
      return new Response(JSON.stringify({ detail: LUT_CURVE_FAILURE_DETAIL }), {
        status: entry.curveStatus, headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({
      t_grid: LUT_T_GRID, lut: entry.rows, meta: LUT_META, gamma: LUT_GAMMA,
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
    if (key === "gamut") {
      // Class indices, not `(c + 1) * tag`.
      //
      // Three properties are being bought here and each cost a revision, so
      // they are written out rather than left to be re-derived.
      //
      // 1. **More than one class.** The generic fill gives a one-channel view a
      //    frame of a single value, and that is symmetric in the dimension
      //    under test: a shader painting every class the same colour could not
      //    be told from a correct one, and neither could a legend row picked by
      //    a constant.
      //
      // 2. **Not symmetric under transposition.** `(row + col) % 4` gives four
      //    classes and still cannot see a readout that swaps the coordinates,
      //    because pixel (row 1, col 2) and pixel (row 2, col 1) hold the same
      //    class. Measured, against the transposed-readout mutation:
      //
      //      (row * 2 + col) % 3, NaN at (2,1)  -> 2 fail
      //      (row + col) % 4,     NaN at (2,1)  -> 2 fail
      //      (row + col) % 4,     NaN at (2,2)  -> 0 fail  (24/24 green)
      //
      //    The third row is the trap: the symmetric fill catches the swap only
      //    because the swapped index lands on the masked pixel, and moving the
      //    mask onto the diagonal -- where it is its own transpose -- takes
      //    that away. Here (row 1, col 2) is class 1 and (row 2, col 1) is the
      //    mask, and the classes either side of the swap differ anyway.
      //
      // 3. **Fewer classes in the frame than in `labels`.** `% 3` against four
      //    labels, so the frame holds classes {0, 1, 2} and never class 3.
      //    This is what makes the declared range observable: with all four
      //    classes present, a client that recomputed vmin/vmax from the pixels
      //    would get {0, 3} -- the header's own values -- and the check that
      //    the range comes from the payload could not fail. Measured: with a
      //    four-class frame, replacing the header range with min/max over the
      //    data left the scenario 24/24 green. It is also the more honest
      //    fixture, because a real frame need not contain every class: the
      //    legend's row count then comes from `labels` because it has to,
      //    rather than because the two happen to agree.
      //
      // The NaN sits at (row 2, col 1). Outside the active projection region a
      // pixel has no class, and 0.0 is a perfectly good class 0 -- "everything
      // here is representable", written across the part of the frame the
      // projector never lit. It is off the first six elements, off the first
      // and off the last, so it does not collide with the upload summary.
      //
      // ここで買っている性質は3つ。(1) 複数クラスがあること（共通の埋め方だと
      // 1チャンネルのビューは全画素が同じ値になり、全クラスを同色で塗る実装も
      // 定数で選んだ凡例も区別できない）。(2) 転置に対して非対称であること
      // ——`(row + col) % 4` は (row 1, col 2) と (row 2, col 1) が同じクラスに
      // なるため行と列の取り違えを検出できない（実測: 対称な埋め方でマスクを
      // 対角へ置くと24件すべて通る）。(3) フレーム内のクラス数がラベル数より
      // 少ないこと——4クラス揃ったフレームでは画素から再計算したレンジも
      // {0, 3} になり、「レンジはペイロード由来」という検査が落ちなくなる
      // （実測: 24/24 のまま通った）。実フレームが全クラスを含むとは限らない
      // 以上、こちらのほうが正直でもある。
      for (let row = 0; row < H; row++) {
        for (let col = 0; col < W; col++) data[row * W + col] = (row * 2 + col) % 3;
      }
      data[2 * W + 1] = NaN;
    } else if (interleaved) {
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
    // A categorical view's range is the class index range, which
    // routers/view.py pins to [0, len(labels) - 1] -- not the +/-1 every other
    // fixture here declares. Sent as itself so a client that ignored the
    // header and assumed a symmetric range would be visible: 0 and 3 are
    // neither of the defaults, and neither is derivable from the other view's.
    // categorical のレンジはクラス番号の範囲であり、他のフィクスチャの ±1 とは
    // 別物である。0/3 はどちらの既定値でもないので、ヘッダを読まない実装が
    // 見える。
    // Keyed on the *kind*, not on `gamut`'s key: `gamut_lite` is categorical
    // too, and a range that did not follow its label count would be a fixture
    // disagreeing with the contract `routers/view.py` enforces
    // (`vmax == len(labels) - 1`) rather than exercising it.
    // キーではなく kind で決める。`gamut_lite` も categorical であり、ラベル数に
    // 従わないレンジはルータが強制している契約と食い違うだけになる。
    const range = spec.kind === "categorical"
      ? { vmin: "0.0", vmax: String(viewLabels(spec).length - 1) }
      : { vmin: "-1.0", vmax: "1.0" };
    const headers = {
      "X-Height": String(H), "X-Width": String(W),
      "X-Channels": String(spec.channels), "X-Kind": spec.kind,
      "X-Vmin": range.vmin, "X-Vmax": range.vmax,
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

// The shipped shader module, imported for the one thing only it can answer:
// whether the GLSL palette the GPU compiles is the same set of colours the
// sidebar legend draws.
//
// `tests/test_web_golden.py` reads `shaders.js` as *text* and therefore sees
// `${CLASS_PALETTE_GLSL}`, not the numbers it expands to. Evaluating the module
// is the only way to compare the two consumers of `CLASS_COLORS` against each
// other, and node is where that can happen.
// シェーダのモジュールを評価して読み込む。ゴールデンテストはテキストとして
// 読むので `${CLASS_PALETTE_GLSL}` しか見えず、展開後の数値を凡例と突き合わせ
// られるのはここだけである。
const shaders = await import(pathToFileURL(path.join(STATIC_DIR, "js", "shaders.js")).href);

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
// The *contents* only. `internal`/`format` are recorded on the same entry but
// read by uploadedFormats() below, so the existing contents assertions keep
// comparing against the same four fields they always have.
// 中身だけを見る。形式は同じ記録に入っているが下の関数で別に読む。
const uploadedPlanes = () => views.map((v) => {
  const u = v.gl.__uploads;
  if (!u.length) return null;
  const { length, first, last, head } = u[u.length - 1];
  return { length, first, last, head };
});

// ...and the *format* only.
//
// Separate from the contents because a wrong format and wrong contents are
// different failures with different symptoms: RGB32F on a one-channel plane
// uploads exactly the same numbers and draws exactly the same first tile, with
// the class index in the red component and two undefined components beside it.
// The plane summary above cannot see it; nothing in the picture says so either.
// 形式だけを見る。1チャンネルの平面を RGB32F で載せても数値は同じで、赤成分に
// 番号が入り残り2成分が未定義になるだけなので、中身の要約では見えない。
const uploadedFormats = () => views.map((v) => {
  const u = v.gl.__uploads;
  if (!u.length) return null;
  const { internal, format } = u[u.length - 1];
  return { internal, format };
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

// The caption under each tile, which is where computeRanges writes what the
// tile is showing. Read separately from the column heading above it: the two
// are written by different functions and say different things.
// 各タイルの下の見出し。上の列見出しとは別の関数が別のことを書くので分けて読む。
const tileCaptions = () => el("tile-grid").children
  .filter((e) => e.className === "tile")
  .map((cell) => (cell.children.find((c) => c.className === "tile-cap mono") ?? {}).textContent);

// The class legend, read back the way the page presents it.
//
// Three separate readings, because three separate things can be wrong and
// each is invisible from the others: how many rows there are, what they say,
// and what colour each one claims. A legend built from a hard-coded list has
// the right number of rows; one whose swatches all resolve to the same colour
// has the right labels. Only reading all three separates them.
// 凡例を3つの側面から読む（行数・文字・色）。どれか1つでは、ハードコードした
// 一覧も、全行が同じ色のパレットも通ってしまう。
const legendRows = () => el("class-legend").children;
const legendLabels = () => legendRows().map((row) => row.children[1].textContent);
const legendSwatches = () => legendRows().map((row) => row.children[0].style.background);

// The palette the *shader* carries, read out of the expanded FRAG_SRC and
// quantised the way the framebuffer quantises it -- so it can be compared,
// byte for byte, with what `classColorCss` gives the legend.
//
// The two are supposed to be one array: `CLASS_PALETTE_GLSL` is generated from
// `CLASS_COLORS`, and the legend derives its swatches from the same. Typing the
// GLSL out by hand instead is the drift this reads back, and it is invisible
// from everywhere else: the legend still shows four distinct colours and the
// picture still shows four flat regions, they are just no longer the same four.
// シェーダ側のパレットを展開後のソースから読み、フレームバッファと同じ量子化を
// 掛けて凡例の色と突き合わせる。GLSLを手書きに戻すと両者はずれるが、凡例は
// 4色、絵も4領域のままなので他のどこからも見えない。
const shaderPalette = () =>
  [...shaders.FRAG_SRC.matchAll(/if \(i == (\d+)\) return vec3\(([^)]*)\);/g)]
    .map((m) => {
      const to8 = (v) => Math.round(Math.min(Math.max(Number(v), 0), 1) * 255);
      const [r, g, b] = m[2].split(",").map((s) => to8(s.trim()));
      return { index: Number(m[1]), css: `rgb(${r}, ${g}, ${b})` };
    });

// The tone-chain controls' enabled state, as one object so a failure names
// which control is wrong rather than just "false".
//
// A copy of main.js's TONE_CONTROL_IDS, and necessarily one: a test that read
// the list out of the code under test would agree with it by construction. The
// cost is that a control added to one and not the other goes unwatched, which
// is why both sides carry a pointer to the other. `el(id)` returning null is
// left to throw here for the same reason main.js throws: the lists having come
// apart is the finding.
// main.js の TONE_CONTROL_IDS の写しであり、写しであることに意味がある
// （コード側の一覧を読むテストは構造上必ず一致する）。片方だけ足すと見落とすので
// 双方に相互参照を置く。
const TONE_IDS = ["ev-slider", "gamma-input", "gamma-slider", "dither-check"];
const toneDisabled = () =>
  Object.fromEntries(TONE_IDS.map((id) => [id, el(id) ? el(id).disabled : "missing"]));
const allToneDisabled = (want) =>
  TONE_IDS.every((id) => el(id) && el(id).disabled === want);

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

// An object as a key-sorted JSON string, so two bodies can be compared for
// equality without depending on the order their fields happen to be written
// in -- and so that an *extra* field is a difference rather than something a
// field-by-field comparison would never look at.
// キー順に正規化したJSON。フィールドの記述順に依存せず比較でき、かつ余計な
// フィールドが増えたことも差として見える（項目ごとの比較では見落とす）。
const canonical = (o) =>
  JSON.stringify(Object.fromEntries(Object.keys(o).sort().map((k) => [k, o[k]])));

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

// ------------------------------------------------------------ the LUT plot
//
// Every `beginPath … stroke` on the curve canvas is one data polyline, and
// nothing else on that canvas is a stroked path: js/plot.js draws its frame
// with `strokeRect` and its gridlines with `fillRect` precisely so this reader
// does not have to tell a curve from a grid line first. Reading them back is
// what turns "three lines were drawn" -- which three identical lines, or three
// lines of the wrong channels, satisfy just as well -- into "line c is at the
// coordinates channel c's values map to".
//
// `style` is captured at `stroke()` rather than at `beginPath()`, since either
// order is legal in the canvas API and only the value in force when the path is
// painted is the one the user sees.
//
// カーブ用canvas上の `beginPath … stroke` は必ずデータの折れ線1本であり、
// 枠は `strokeRect`、目盛線は `fillRect` で描かれているので、ここで両者を
// 区別する必要は無い。座標まで読むことで「3本描かれた」（同じ座標の3本でも、
// チャンネルを取り違えた3本でも成り立つ）を「c番目の線はチャンネルcの値が
// 写る座標にある」へ変える。色は `stroke()` 時点の値を採る。
function strokedPaths(canvas) {
  const out = [];
  let points = null;
  let style = null;
  for (const [op, ...args] of canvas.__ctx2dCalls || []) {
    if (op === "set:strokeStyle") style = args[0];
    else if (op === "beginPath") points = [];
    else if (op === "moveTo" || op === "lineTo") {
      if (points) points.push([args[0], args[1]]);
    } else if (op === "stroke") {
      if (points && points.length) out.push({ style, points });
      points = null;
    }
  }
  return out;
}

// Only the paths drawn by the newest `drawLutCurve` call. The canvas keeps one
// context for the whole run, so its call log accumulates across every redraw a
// scenario triggers; a check that read the whole log would be describing the
// first draw and the newest one at once.
// 最新の描画分だけを取る。コンテキストは実行を通じて1つなので、記録は再描画の
// たびに積み上がる（全部を読むと初回と最新が混ざる）。
function lutPathsSinceLastClear(canvas) {
  const calls = canvas.__ctx2dCalls || [];
  let start = 0;
  for (let i = 0; i < calls.length; i++) if (calls[i][0] === "clearRect") start = i;
  return strokedPaths({ __ctx2dCalls: calls.slice(start) });
}

// What js/plot.js says it drew: the data rectangle and the mapping.
//
// On `__viewer` rather than on `__viewer.state`: the page never reads it, so
// it is not part of the viewer's state -- see the note on `lastLutPlot` in
// main.js. `__viewer` is a getter, so this reads the newest draw.
const lutPlot = () => (globalThis.window.__viewer.lutPlot ?? null);

// Where a value at `t` must have been drawn, from the fixture and the plot's
// own declared rectangle and range.
//
// The *layout* (margins, insets) is the module's business and is read back from
// the return value; the *mapping from data to that rectangle* is not, and is
// computed here from `LUT_T_GRID`/`LUT_ROWS` directly. That split is what keeps
// this from comparing the implementation with itself: if `vMax` were pinned to
// 1.0, the coordinates and the returned range would still agree with each
// other -- so the range is asserted separately, against the fixture's own
// maximum, by `the plot's vertical axis comes from the data`.
//
// 余白などの配置は実装の裁量なので戻り値から読み、データ→矩形の写像は
// フィクスチャから自前で計算する。この分け方が「実装同士の突き合わせ」を
// 避けている（上限を 1.0 に固定した実装は座標と戻り値が互いに整合してしまう
// ため、範囲そのものは別のチェックがフィクスチャの最大値と突き合わせる）。
function expectedPoint(plot, t, v) {
  return [
    plot.plot.x + (plot.plot.w * (t - plot.tMin)) / (plot.tMax - plot.tMin),
    plot.plot.y + plot.plot.h * (1 - (v - plot.vMin) / (plot.vMax - plot.vMin)),
  ];
}

const closeEnough = (a, b) => Math.abs(a - b) < 1e-6;

function pointsMatch(points, plot, values) {
  if (!plot || points.length !== values.length) return false;
  return points.every(([x, y], i) => {
    const [ex, ey] = expectedPoint(plot, LUT_T_GRID[i], values[i]);
    return closeEnough(x, ex) && closeEnough(y, ey);
  });
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
    options.join(",") === "coef_tiles,plain_tiles,legacy_tiles,gamut,gamut_lite,"
      + "vector_field,surface,compensation,tri_scalar,bad_rgb,mangled_layout,"
      + "mono_scalar,headerless_tiles,refused_view",
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
  //
  // `vector_field` declares `kind: "flow"`, which no row of KIND_TRAITS
  // answers. It replaced `gamut` here when `categorical` was implemented, and
  // the replacement is the point: this check is about *unknown* kinds, so
  // pointing it at whichever kind happens to be unimplemented today would
  // retire it the moment that kind is built. See the fixture's own note.
  // 未知の kind は `flow`。`categorical` を実装した時点で `gamut` はここの
  // 役目を果たせなくなるため、実装予定の無いものに差し替えてある。
  const before = state.payload;
  el("view-select").value = "vector_field";
  el("view-select").__fire("change");
  await sleep(300);
  check("unsupported kind is refused with a message naming it",
    /未対応/.test(statusText()) && /flow/.test(statusText()), statusText());
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

  // --- a `text` parameter must produce a control a user can actually type in
  //
  // The id of a stored target only exists at runtime, so `compensation`
  // declares it `type: "text"` with no `choices` -- see the fixture's comment
  // in VIEWS for why a `select` cannot express it. Three things have to be
  // true for that to be usable, and each fails silently on its own: the
  // control has to exist, it has to be a *text* input (a `hidden` or `number`
  // one is either invisible or refuses the characters an id is made of), and
  // what is typed has to reach the query string.
  // 実行時にしか存在しないidのための `text` パラメータ。存在すること・
  // テキスト入力であること・打った値がクエリ文字列に届くこと、の3つとも
  // 単独では静かに壊れる。
  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await settled();

  const targetControl = el("sp-target");
  check("the view with a text param builds a control for it",
    targetControl !== null);
  check("the ターゲットID control is a text input, not a select or a hidden field",
    targetControl && targetControl.tagName === "INPUT" && targetControl.type === "text",
    `tag=${targetControl && targetControl.tagName} type=${targetControl && targetControl.type}`);
  // The select beside it still has to be a real dropdown: one `else` branch
  // serves both, so a change that fixes one can break the other.
  // 隣の select が本物のドロップダウンのままであることも見る。
  check("a select param next to a text one is still a populated dropdown",
    el("sp-output") && el("sp-output").tagName === "SELECT"
      && el("sp-output").children.length === 2,
    `tag=${el("sp-output") && el("sp-output").tagName}`);

  const beforeTyping = viewRequestCount();
  stageParam("target", "target7");
  stageParam("output", "code");
  await sleep(60);
  check("typing a target id stages it without fetching",
    viewRequestCount() === beforeTyping, `${beforeTyping} -> ${viewRequestCount()}`);
  applyParams();
  await settled();
  const withTarget = requestLog[requestLog.length - 1];
  check("the typed target id reaches the query string",
    /target=target7/.test(withTarget), withTarget);
  check("the select next to it is submitted in the same group",
    /output=code/.test(withTarget), withTarget);
}

if (SCENARIO === "gamut") {
  // A categorical view is one picture of *class indices*, and nothing about
  // that survives being treated as a scalar view of one channel.
  //
  // The failure this scenario is written against is the quiet one. A class
  // index is a perfectly good float: the scalar path uploads it, the diverging
  // colormap gives it a colour, exposure and gamma move that colour, and the
  // result is a smooth-looking picture of four flat regions with a ± range
  // under it. Everything renders. What is lost is that the numbers are names,
  // not quantities -- and once the tone chain is live, the colours stop being
  // the ones the legend is standing next to.
  //
  // categorical は「クラス番号の絵」であり、1チャンネルのスカラービューとして
  // 扱うと何も残らない。クラス番号も float なので、転送もカラーマップも露出も
  // 通ってしまい、絵は出る。失われるのは「その数値は量ではなく名前である」
  // ことと、トーンチェーンが効いた瞬間に凡例と色が食い違うことである。
  clickLoad("A");
  await settled();

  // Both directions for every control this view turns off. "Always enabled"
  // and "always disabled" each pass one half, which is how a check about a
  // control's state stops being able to fail.
  // 無効化の検査は両方向で見る（常時有効・常時無効のどちらも片側だけなら通る）。
  check("the tone controls are live on a scalar view",
    allToneDisabled(false), JSON.stringify(toneDisabled()));
  check("the colormap control is enabled for a scalar view",
    el("cmap-select").disabled === false, `disabled=${el("cmap-select").disabled}`);

  const options = el("view-select").children.map((o) => o.value);
  check("the categorical view is reachable from the selector",
    options.includes("gamut"), options.join(","));

  el("view-select").value = "gamut";
  el("view-select").__fire("change");
  await settled();

  check("a categorical view renders instead of being refused as an unknown kind",
    /表示中/.test(statusText()), statusText());

  // --- one tile, and one tile that is not captioned with a class name
  //
  // `axesFromLabels` cannot parse these labels (no "->"), so the label-derived
  // fallback produces a 1x1 grid captioned `labels[0]` -- "色域外" here, the
  // name of class 0, written over a picture containing all four. The tile
  // count is right either way, which is exactly why the heading is asserted
  // separately from it.
  // ラベルは "->" 形式ではないのでフォールバックは1行1列を作り、見出しは
  // `labels[0]`（＝クラス0の名前）になる。枚数はどちらでも正しいので、
  // 見出しは別に検証する。
  check("a categorical payload draws one tile",
    views.length === 1, `tiles=${views.length}`);
  check("the tile is not headed with the name of class 0",
    JSON.stringify(colHeads()) === JSON.stringify(["クラス"]),
    JSON.stringify(colHeads()));

  // --- the upload: contents, and the format the contents went up in
  //
  // Two assertions because they fail differently and neither implies the
  // other. RGB32F on this plane uploads the same twenty numbers -- so the
  // contents check passes -- and puts each class index in a red component with
  // two undefined components beside it, which draws.
  // 中身と形式は別々に見る。RGB32F でも数値は同じなので中身の検査は通り、
  // 赤成分だけに番号が入った状態で描けてしまう。
  check("the single tile carries the whole class-index plane",
    JSON.stringify(uploadedPlanes())
      === JSON.stringify([{ length: H * W, first: 0, last: 1, head: [0, 1, 2, 0, 1, 2] }]),
    JSON.stringify(uploadedPlanes()));
  check("class indices are uploaded as R32F/RED, not as a colour",
    JSON.stringify(uploadedFormats())
      === JSON.stringify([{ internal: "R32F", format: "RED" }]),
    JSON.stringify(uploadedFormats()));

  // --- and the shader is told which branch to run
  //
  // The texture can be uploaded correctly and the shader left in u_mode = 0,
  // which runs the diverging colormap over 0..3: four flat regions in four
  // shades of one ramp, which looks like a working view of a classification.
  // テクスチャを正しく載せても u_mode が0のままなら、0〜3が発散カラーマップに
  // 通り、1本のランプ上の4段階として「それらしく」描かれてしまう。
  check("the shader is told u_mode = 2 for the categorical tile",
    views[0].gl.__uniforms.u_mode === 2, `u_mode=${views[0].gl.__uniforms.u_mode}`);

  // --- the display range is the class range the server declared
  //
  // 0 and 3, from the header -- neither of which is this harness's default
  // ±1, and neither derivable from it. views/gamut.py takes the range from
  // its LABELS rather than from the pixels precisely so an all-in-gamut frame
  // does not collapse to vmin == vmax == 0; a client that recomputed it from
  // the data would undo that from the other side, and **the fixture's frame
  // holds only classes 0-2 so that such a client is visible here** (with all
  // four classes present the data-derived range is {0, 3} too, and this check
  // measured 24/24 green against exactly that mutation).
  // レンジはヘッダの 0/3（このハーネスの既定 ±1 のどちらでもない）。フレームが
  // クラス0〜2しか持たないのは、画素から再計算する実装をここで見えるように
  // するためである（4クラス揃っていると再計算しても {0, 3} になり通ってしまう）。
  check("the categorical range is the class index range from the payload",
    JSON.stringify(state.ranges) === JSON.stringify([{ vmin: 0, vmax: 3 }]),
    JSON.stringify(state.ranges));

  // ...and it is *worded* as a class count rather than as an interval.
  //
  // The other half of `labelsAreClasses`, and the half nothing was watching.
  // `[0.0000, 3.0000]` is what the shared-range branch writes for an rgb view,
  // and it invites the reading that 1.5 is a value this view can hold. The
  // tile caption is the same decision one line down: `labels.join(" ")` there
  // -- again correct for rgb, where the labels are R/G/B -- puts all four class
  // names under the tile, restating the legend in the one place with no room
  // for it. Measured before these two existed: forcing the `classes` ternary
  // false, and reverting the caption, each left the scenario 24/24 green.
  // レンジの「言い方」も見る。`[0.0000, 3.0000]` は rgb 用の書き方で、1.5 という
  // 値がありうるかのように読める。タイル見出しの `labels.join(" ")` も同様に
  // rgb 用で、幅の無い場所に凡例を書き写すことになる。実測: この2件が無い状態
  // では、どちらへ戻しても24件すべて通っていた。
  check("the display range is worded as a class count, not as an interval",
    el("range-info").textContent === "クラス 0–3", el("range-info").textContent);
  check("the tile caption does not restate the legend under the tile",
    JSON.stringify(tileCaptions()) === JSON.stringify(["4 クラス（凡例参照）"]),
    JSON.stringify(tileCaptions()));

  // --- the legend
  //
  // The one place the class -> colour mapping is stated in words, and the one
  // thing a screenshot of this view is read against. Its labels have to come
  // from `payload.labels`: the fixture's four are deliberately neither
  // views/gamut.py's own class names nor in its order, so a legend built from
  // a hard-coded list of the shipped names looks identical to a correct one
  // against the real server and wrong only here.
  // クラスと色の対応を言葉で述べる唯一の場所。ラベルは `payload.labels` から
  // 取らねばならない。フィクスチャの4つは実装側の語とも順序とも違えてある。
  check("the legend is on screen for a categorical view",
    el("class-legend").hidden === false, `hidden=${el("class-legend").hidden}`);
  check("the legend has one row per class in the payload",
    legendRows().length === state.payload.labels.length,
    `rows=${legendRows().length} labels=${state.payload.labels.length}`);
  check("the legend's labels are the payload's, not a hard-coded list",
    JSON.stringify(legendLabels()) === JSON.stringify(state.payload.labels),
    `${JSON.stringify(legendLabels())} vs ${JSON.stringify(state.payload.labels)}`);
  // Four rows are useless if two of them are the same colour: the picture
  // would show three regions where the data has four, and the legend would
  // insist there are four. Distinctness is what makes the mapping invertible
  // by eye; how far apart the four are perceptually is measured on the GPU,
  // where the pixels actually are.
  // 4行あっても2つが同じ色なら意味がない（絵は3領域、凡例は4クラスと主張）。
  // 見た目の距離はGPU側で測る。
  check("each class in the legend has its own colour",
    new Set(legendSwatches()).size === legendRows().length
      && legendSwatches().every((c) => typeof c === "string" && c.length > 0),
    JSON.stringify(legendSwatches()));
  // ...and it is the *shader's* colour, not a second palette that agrees today.
  //
  // `CLASS_COLORS` is supposed to be one array with two consumers: the GLSL is
  // generated from it, and `classColorCss` quantises it for the swatch. Typing
  // the GLSL out beside the array instead is a change nothing else here can
  // see -- the legend still lists four distinct colours, the picture still has
  // four flat regions, and only a GPU readback would show that they are no
  // longer the same four. Compared after 8-bit quantisation, which is the
  // precision at which "the swatch is the pixel" either holds or does not.
  // 凡例の色がシェーダの色そのものであること。GLSLを手書きに戻しても、凡例は
  // 4色のまま・絵も4領域のままで、他のどこからも差が見えない。比較は8bit量子化
  // 後に行う（「スウォッチ＝画素」が成り立つかどうかはその精度で決まる）。
  check("the legend's colours are the palette the shader compiles",
    (() => {
      // A missing export is a failing check, not an exception. Everywhere else
      // this harness lets a missing thing throw, because throwing names it --
      // but this check sits in the middle of the scenario and an abort here
      // would take the twenty-odd results after it with it, which reports less
      // than "this one is false" does.
      // 輸出が無い場合は例外ではなく失敗として扱う。ここで中断すると後続の
      // 20件あまりの結果ごと失われ、かえって分からなくなる。
      const palette = shaders.CLASS_COLORS;
      if (!Array.isArray(palette) || typeof shaders.classColorCss !== "function") return false;
      const glsl = shaderPalette();
      if (glsl.length !== palette.length) return false;
      return glsl.every((entry, i) => entry.index === i
        && entry.css === shaders.classColorCss(i));
    })(),
    `${JSON.stringify(shaderPalette())} vs `
      + JSON.stringify((shaders.CLASS_COLORS ?? []).map((_, i) => shaders.classColorCss?.(i))));
  // ...and it has to be where the picture is being looked at.
  //
  // The sidebar is `overflow-y: auto`, and with a target and a LUT loaded its
  // foot is off the bottom of a 1600x1000 window -- measured, in the browser,
  // with the legend in its first position down among the two standing
  // legends. That is the same failure the display range and the timings had
  // before they moved into the readout strip, and it is worse here: those two
  // restated numbers that were on screen elsewhere, while this is the only
  // statement of what the four colours mean. Pinned by position rather than by
  // pixels because there is no CSS engine here; the first `<hr>` is what
  // separates the per-view controls at the top of the column from everything
  // that scrolls below.
  // サイドバーは縦に溢れる。ターゲットとLUTを読み込んだ1600x1000の窓では
  // 下端が画面外にあることを実測した（凡例を最初に置いた位置）。表示レンジ
  // と計測を移したのと同じ問題であり、しかもこちらは「4色が何を意味するか」
  // を述べる唯一の場所なので、より重い。CSSエンジンが無いので位置で固定する。
  check("the class legend sits above the sidebar's first divider",
    (() => {
      const kids = el("class-legend").parentElement.children;
      const at = kids.indexOf(el("class-legend"));
      const firstHr = kids.findIndex((c) => c.tagName === "HR");
      return at >= 0 && firstHr >= 0 && at < firstHr;
    })(),
    panelOf("class-legend"));

  // --- exposure, gamma and the colormap do not apply to a name
  check("the colormap control is disabled for a categorical view",
    el("cmap-select").disabled === true, `disabled=${el("cmap-select").disabled}`);
  check("the tone controls are disabled for a categorical view",
    allToneDisabled(true), JSON.stringify(toneDisabled()));

  // --- the readout reads `labels[value]`, not `labels[channel]`
  //
  // The axis the brief for this task names. There is one channel and four
  // labels: the per-channel reading every other kind uses prints `labels[0]`
  // -- "色域外" -- beside the correct number, under the correct heading, for
  // every pixel in the frame. The fixture's class at (col 2, row 1) is 1, so
  // the two readings disagree here (`labels[0]` is 色域外), and its transpose
  // (col 1, row 2) is the masked pixel, so a swapped index disagrees too.
  // チャンネルで引く読み方は全画素で `labels[0]` を出す。フィクスチャの
  // (col 2, row 1) はクラス1、その転置 (col 1, row 2) はマスク画素なので、
  // ラベルの軸と添字の順序の両方について食い違いが出る。
  views[0].onHover({ col: 2, row: 1 });
  const classItems = viewValueItems();
  check("the readout names the class the pixel is, not the first class",
    classItems.length === 1
      && classItems[0].children[1].textContent === "1 1超クリップ",
    `${classItems.length} item(s): ${JSON.stringify(viewValueNumbers())}`);
  check("the readout's caption is the class axis, not a channel name",
    JSON.stringify(viewValueLabels()) === JSON.stringify(["クラス"]),
    JSON.stringify(viewValueLabels()));

  // A pixel with no class must read as no class. Outside the active
  // projection region the payload is NaN, and 0 -- the class a rounded NaN
  // would land on -- is "範囲内": the most reassuring possible wrong answer,
  // over the part of the frame the projector never lit.
  // クラスの無い画素は「無い」と読めること。丸めた NaN が落ちる先はクラス0
  // ＝「範囲内」であり、最も安心できる誤答になる。
  views[0].onHover({ col: 1, row: 2 });
  check("a pixel with no class reads as NaN rather than as class 0",
    JSON.stringify(viewValueNumbers()) === JSON.stringify(["NaN"])
      && JSON.stringify(viewValueClasses()) === JSON.stringify(["vv-num nan"]),
    `${JSON.stringify(viewValueNumbers())} ${JSON.stringify(viewValueClasses())}`);

  // --- a categorical view whose class count is not the palette's length
  //
  // Every legend assertion above runs on `gamut`, which has four classes
  // against a four-entry `CLASS_COLORS`. On that fixture alone,
  // `rows === payload.labels.length` is `4 === 4` whichever array the loop
  // reads, so a legend driven off the *palette* -- the obvious wrong build,
  // since the swatches come from there -- passed every one of them (measured:
  // `payload.labels.length` -> `CLASS_COLORS.length` left the scenario green).
  //
  // The failure that hides there is not this view's: it is the next
  // categorical view's. `routers/view.py` admits any of them at
  // `len(labels) >= 2`, so a three-class view draws four rows -- three right,
  // and a fourth whose label is the string `undefined` beside a colour no pixel
  // in the picture carries. The legend is the only place the mapping is stated
  // in words, so it would be asserting a class that does not exist.
  //
  // The asymmetry is asserted here as well as arranged in the fixture,
  // otherwise a later edit could quietly restore the four-class symmetry and
  // take this check's discrimination with it while leaving it green.
  // 上の凡例の検査はすべて `gamut`（4クラス）で走っており、`CLASS_COLORS` も
  // 4要素なので、行数をどちらから取っても同じ数になる。ここでクラス数の違う
  // ビューに切り替えて、行数がペイロード由来であることを見る。非対称であること
  // 自体もチェックに書く（フィクスチャが対称に戻れば判別力が消えるため）。
  el("view-select").value = "gamut_lite";
  el("view-select").__fire("change");
  await settled();
  const paletteLength = Array.isArray(shaders.CLASS_COLORS)
    ? shaders.CLASS_COLORS.length : -1;
  check("the legend's row count is the payload's class count, not the palette's",
    state.payload.labels.length === 3
      && paletteLength !== state.payload.labels.length
      && legendRows().length === state.payload.labels.length
      && JSON.stringify(legendLabels()) === JSON.stringify(state.payload.labels)
      && new Set(legendSwatches()).size === legendRows().length,
    `labels=${JSON.stringify(state.payload.labels)} palette=${paletteLength} `
      + `rows=${legendRows().length} shown=${JSON.stringify(legendLabels())}`);
  // The two other places the class count is read, on the same view: a caption
  // and a range that came from the palette would be wrong in the same way and
  // are not covered by the `gamut` checks above for the same reason.
  // クラス数を読む残り2箇所も同じビューで見る。
  check("the caption and range of a three-class view count three classes",
    el("range-info").textContent === "クラス 0–2"
      && JSON.stringify(tileCaptions()) === JSON.stringify(["3 クラス（凡例参照）"]),
    `range=${el("range-info").textContent} captions=${JSON.stringify(tileCaptions())}`);

  // --- and every one of those has to be given back on the way out
  //
  // A one-way trip would leave the coefficient tiles with dead exposure and γ
  // sliders and a class legend explaining a picture that is no longer there.
  // 片道だと、係数タイルに戻っても露出とγが死んだままになり、消えた絵を
  // 説明する凡例が残る。
  el("view-select").value = "coef_tiles";
  el("view-select").__fire("change");
  await settled();
  check("switching back to a scalar view restores one tile per channel",
    views.length === 12, `tiles=${views.length}`);
  check("switching back to a scalar view puts the shader back in u_mode = 0",
    views.every((v) => v.gl.__uniforms.u_mode === 0),
    JSON.stringify(views.map((v) => v.gl.__uniforms.u_mode)));
  check("switching back to a scalar view re-enables the tone controls",
    allToneDisabled(false), JSON.stringify(toneDisabled()));
  check("switching back to a scalar view takes the class legend away",
    el("class-legend").hidden === true && legendRows().length === 0,
    `hidden=${el("class-legend").hidden} rows=${legendRows().length}`);
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
    "probe-pos", "probe-pin", "probe-grid", "view-title", "view-values",
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

  // --- and every band is placed in a row *by name*, so a hidden one collapses
  //     its own row and nothing else's
  //
  // `body` is a 5-row grid and `#space-note` is `hidden` for a projector-space
  // matrix -- which is the ordinary case. `display: none` removes it as a grid
  // item entirely, so under a positional template every band after it slid up
  // one row and the `1fr` landed on the *readout* instead of on the tiles: the
  // strip grew a blank band under its panels (measured ~116px), and the tile
  // height came out equal to the sidebar's content height, i.e. determined by
  // how many controls the current view happens to declare. `surface` declares
  // three and `coef_tiles` one, so switching view resized the pictures.
  //
  // Asserted the same way and for the same reason as `.readout`'s row: it is a
  // stylesheet declaration with no error, no wrong number and no other check
  // behind it -- deleting the five lines restored the defect with the whole JS
  // gate green.
  //
  // 各段を*名指し*で行に置く。`body` は5行のグリッドで、`#space-note` は投影
  // 座標の行列（＝通常の場合）では `hidden` になり、`display: none` はグリッド
  // 項目そのものを消す。位置で決まるテンプレートでは以降の段が1行ずつ繰り上がり、
  // `1fr` がタイルではなく読み取り欄に載っていた——帯の下に約116pxの空白が生じ、
  // タイルの高さは「そのビューが宣言しているコントロールの数」で決まっていた
  // （`surface` は3つ、`coef_tiles` は1つ）。5行を消しても他に落ちるものが無い
  // ので、`.readout` の行と同じやり方で固定する。
  const GRID_ROWS = [[".toolbar", 1], ["#space-note", 2], ["main", 3],
    [".readout", 4], ["footer#status", 5]];
  const rowMisses = GRID_ROWS.filter(
    ([sel, n]) => !new RegExp(`grid-row:\\s*${n}\\b`).test(cssDeclarations(sel)));
  check("each band of the page is placed in a named grid row, not by document order",
    rowMisses.length === 0,
    rowMisses.length
      ? `not placed: ${rowMisses.map(([sel, n]) => `${sel} (want row ${n})`).join(", ")}`
      : GRID_ROWS.map(([sel, n]) => `${sel}=${n}`).join(" "));

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

  // --- a view that declares nothing keeps its panel
  //
  // The default is the *showing* direction. `views/base.py` argues that at
  // length, `routers/view.py` says the answer is "not derivable from `kind`",
  // and `main.js` repeats both -- and none of it was asserted anywhere. The
  // two views the panel state was read on are `coef_tiles` (scalar, declares
  // `true`) and `surface` (rgb, declares `false`), so `kind === "scalar"` --
  // the implementation all three comments forbid -- is indistinguishable from
  // reading the declaration, and so is `values_are_matrix !== false`, which
  // inverts the default the docstring spends a paragraph on. Both passed every
  // check in every scenario.
  //
  // `tri_scalar` is what separates them: scalar, three channels, and no
  // declaration -- the shape of a residual-error view, whose three numbers are
  // *not* the matrix and are the only ones it produces. Under either wrong
  // implementation they vanish from the screen entirely.
  //
  // 既定は「出す」側である。`views/base.py` はそれを長々と論じ、
  // `routers/view.py` は「`kind` からは導けない」と書き、`main.js` は両方を
  // 繰り返している——そしてどれも検証されていなかった。板の状態を読んでいた
  // ビューは `coef_tiles`（scalar・`true`）と `surface`（rgb・`false`）だけで、
  // この2つでは 3つのコメントが禁じている `kind === "scalar"` が宣言を読むのと
  // 区別できず、既定を反転させる `!== false` も同様である（どちらも全シナリオ
  // 通過）。`tri_scalar` がそれを分ける: scalar・3チャンネル・宣言なし——
  // 残差ビューの形であり、その3つの数値は行列ではなく、かつそのビューが出す
  // 唯一の数値である。誤った実装ではそれが画面から丸ごと消える。
  el("view-select").value = "tri_scalar";
  el("view-select").__fire("change");
  await settled();
  check("a view that declares nothing about its values keeps its panel",
    state.spec.key === "tri_scalar" && el("view-panel").hidden === false,
    `spec=${state.spec.key} hidden=${el("view-panel").hidden} `
      + `declared=${JSON.stringify(state.spec.values_are_matrix)}`);

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
  // `null` is what routers/session.py sends for every non-finite coefficient.
  // It is uncommon rather than ordinary -- the matrices this viewer is pointed
  // at are finite everywhere, out-of-region pixels included, and the NaN on
  // screen is painted by the views at render time (which is why the readout
  // needs the separate marker checked below). But nothing forbids a matrix
  // from holding NaN, and when one does the cell has to read as "masked", in
  // the same colour the tiles paint those pixels, rather than as the "—" that
  // means "nothing has been read yet".
  // 非有限係数は `null` で届く。これは通常の答えではなく稀な答えである——
  // このビューアが読む行列は領域外も含めて全て有限で、画面上のNaNはビューが
  // 描画時に塗っている（だから下の「領域外」の印が別に要る）。とはいえNaNを
  // 持つ行列がありえないわけではなく、その場合セルは「まだ何も読んでいない」
  // を意味する「—」ではなく、タイルと同じマスク色で出なければならない。
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

  // --- and the marker only leaves when the numbers it disclaims do
  //
  // The heading is written synchronously on hover; the twelve cells are not
  // touched until the point query answers. So between the two there is a state
  // where the cells still hold the *previous* pixel, and whatever the marker
  // says during it is a statement about those numbers -- not about the pixel
  // named above them. Clearing it there (`setProbePos(col, row, false)`, which
  // is what shipped) put the out-of-region values on screen unmarked, in the
  // same type, colour and precision as real coefficients: the exact state the
  // marker exists to make impossible, for one round trip, every time the
  // pointer left a masked pixel -- which on the real matrix is most pixels.
  //
  // Held open so the intermediate state is a place the client can be put,
  // rather than a microsecond nobody can observe. Both halves are asserted:
  // the marker survives *while* the outside pixel's numbers are shown, and it
  // goes away once the inside pixel's numbers arrive -- an implementation that
  // simply never clears it would fail the second.
  //
  // 見出しは同期的に、12個のセルは応答が来てから書かれる。その間セルには前の
  // 画素の値が残っており、その間の印は「その数値についての主張」であって、
  // 上に出ている画素名についてのものではない。ここで印を消す実装（出荷されて
  // いたもの）は、領域外の値を本物と同じ字体・色・桁数のまま印なしで見せる
  // ——マスクされた画素からポインタを動かすたびに、1往復ぶん。実機ではそれが
  // 大半の画素である。応答を保留して中間状態を作り、両方向を見る（外れた値が
  // 出ている間は印が残ること、内側の値が届いたら消えること）。
  holdProbes = true;
  views[0].onHover({ col: 2, row: 0 });                    // outside, answered
  await waitForHeld(0, 2);
  releaseProbe(0, 2);
  await waitFor(() => el("probe-pos").classList.contains("outside"));
  views[0].onHover({ col: 2, row: 1 });                    // inside, held open
  await waitForHeld(1, 2);
  await sleep(40);
  const duringFlight = {
    said: el("probe-pos").textContent,
    flagged: el("probe-pos").classList.contains("outside"),
    cells: matrixTexts(),
  };
  releaseProbe(1, 2);
  await waitFor(() => matrixTexts()[0] === expectedMatrixTexts(1, 2)[0]);
  holdProbes = false;
  releaseHeldProbes();
  await sleep(40);
  check("the marker stays with the numbers it disclaims until they are replaced",
    JSON.stringify(duringFlight.cells) === JSON.stringify(expectedMatrixTexts(0, 2))
      && duringFlight.flagged && /有効投影領域の外/.test(duringFlight.said)
      && !el("probe-pos").classList.contains("outside")
      && JSON.stringify(matrixTexts()) === JSON.stringify(expectedMatrixTexts(1, 2)),
    `in flight: said=${duringFlight.said} flagged=${duringFlight.flagged} `
      + `cells=${JSON.stringify(duringFlight.cells)} | after: `
      + `flagged=${el("probe-pos").classList.contains("outside")} `
      + `cells=${JSON.stringify(matrixTexts())}`);

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
  const probesBeforeTransfer = probeLog.length;
  clickLoad("B");
  await sleep(25);
  check("the view transfer was still in flight when the hover happened",
    !resolved.has("view:B"), `resolvedB=${resolved.has("view:B")}`);
  const tagDuringTransfer = payloadTag();
  for (let col = 0; col < 4; col++) views[0].onHover({ col, row: 2 });
  await sleep(40);
  const probedDuringTransfer = probeLog.slice(probesBeforeTransfer);
  await settled();
  check("hovering during a view transfer does not stall the tiles",
    payloadTag() === 2 && /表示中/.test(statusText()),
    `payload tag=${payloadTag()} (A=1, B=2) status=${statusText()}`);

  // --- and it asks the matrix the tiles are drawn from, not the one arriving
  //
  // `state.matrixId` becomes B the moment `POST /session/matrix` answers --
  // seconds before B's pixels arrive on the 1080p matrix, with the pointer over
  // the image the whole time (the comment above argues that is every load, not
  // an unlucky one). A readout keyed on it answers about B while the tiles
  // still show A. That is not merely early: `inside_active_region` is a
  // property of a particular matrix, and two matrices captured either side of
  // a `config.toml` edit have different active regions -- routine here, where
  // the config is an experiment log. The tile under the cursor would be grey
  // (masked in A, the picture) while the readout showed finite numbers and no
  // marker (inside B, the arrival), or the reverse, for the whole transfer.
  //
  // Asserted on the *issued* URLs rather than on the cells, because the answer
  // is discarded by the staleness guard anyway -- the request itself is where
  // the wrong matrix is chosen.
  //
  // `state.matrixId` は `POST /session/matrix` が返った瞬間に B になる——1080p
  // では B の画素が届く数秒前であり、その間ポインタは画像の上にある（上の
  // コメントが言うとおり、それは「たまたま」ではなく毎回である）。それを見る
  // 読み取り欄はタイルが A を映している間 B について答える。単に早いのではない。
  // `inside_active_region` は行列ごとの性質であり、`config.toml` を挟んで撮った
  // 2つの行列は有効領域が違う（設定が実験ログであるこのリポジトリでは日常）。
  // カーソルの下のタイルは灰色（Aではマスク）なのに読み取り欄は有限値を印なしで
  // 出す、あるいはその逆が、転送のあいだずっと続く。応答はスタール判定で捨てら
  // れるので、検証するのは発行されたURLのほうである（誤った行列が選ばれるのは
  // 要求の時点だからである）。
  check("a hover during a load asks the matrix on screen, not the one still arriving",
    tagDuringTransfer === 1 && probedDuringTransfer.length > 0
      && probedDuringTransfer.every((u) => u.startsWith("/api/matrix/matrix_A/probe?")),
    `payload tag during transfer=${tagDuringTransfer} `
      + `probes=${JSON.stringify(probedDuringTransfer)}`);

  // --- and a load clears the readout, rather than leaving the old matrix in it
  //
  // `clearReadout()` at the end of `reload()` is what makes the readout stop
  // describing a matrix that is no longer on screen. Deleting the call is
  // invisible to every other check here, and the result is that after loading
  // matrix B the strip still shows *matrix A's* twelve coefficients -- and, if
  // that pixel was outside A's active region, A's red marker -- until the
  // pointer happens to move. A readout stating the wrong matrix's numbers
  // under the picture of another is the failure class this whole branch exists
  // to remove.
  //
  // Run last, and with an out-of-region pixel, so both the cells and the
  // marker have something to fail with.
  //
  // `reload()` 末尾の `clearReadout()` が、画面から消えた行列を読み取り欄が
  // 説明し続けないようにしている。この呼び出しを消しても他のチェックは一切
  // 落ちず、結果として行列Bを読み込んだあとも帯には*行列A*の12個の係数が——
  // その画素がAの有効領域の外だったならAの赤い印ごと——ポインタが動くまで
  // 残る。別の絵の下に別の行列の数値を出す読み取り欄は、このブランチが
  // 取り除こうとしてきた失敗そのものである。セルと印の両方が落ちうるよう、
  // 領域外の画素で、最後に確認する。
  views[0].onHover({ col: 2, row: 0 });                    // outside -> marked
  await waitFor(() => el("probe-pos").classList.contains("outside"));
  const beforeReload = matrixTexts();
  const markedBeforeReload = el("probe-pos").classList.contains("outside");
  viewDelayMs.B = 0;
  clickLoad("B");
  await settled();
  // The marker's state before the load is reported but not required: this
  // check is about `clearReadout()`, and demanding a *marked* starting state
  // would make it fail for every mutation that breaks the marker instead --
  // which those mutations' own checks already cover.
  // 直前に印が付いていたかは報告するだけで条件にしない。このチェックの対象は
  // `clearReadout()` であり、印が付いていることを要求すると、印を壊す変異の
  // すべてでここも落ちる（それは印側のチェックの仕事である）。
  check("a load clears the readout instead of leaving the previous matrix's numbers in it",
    JSON.stringify(beforeReload) !== dashes
      && JSON.stringify(matrixTexts()) === dashes
      && el("probe-pos").textContent === "—"
      && !el("probe-pos").classList.contains("outside"),
    `before=${JSON.stringify(beforeReload)} marked=${markedBeforeReload} | `
      + `after=${JSON.stringify(matrixTexts())} pos=${el("probe-pos").textContent} `
      + `marked=${el("probe-pos").classList.contains("outside")}`);
}

if (SCENARIO === "target") {
  // The control that makes the `compensation` view drivable at all.
  //
  // That view has shipped complete since plan B3 Task 3 and, until this task,
  // unreachable: its ターゲットID is free text naming an id that only exists
  // after a `POST /api/target`, and no page element issued one. Every request
  // from a browser therefore came back "needs a target image", forever, with
  // the only way out being curl. Nothing failed; the viewer simply had a view
  // in its dropdown that could not be made to draw.
  //
  // `compensation` ビューを駆動する唯一の手段。ターゲットIDは
  // `POST /api/target` の後にしか存在しないidを指す自由入力であり、それを
  // 発行する要素がページに無かったため、ブラウザからは永久に「ターゲットが
  // 要る」と返り続けていた。何も失敗はせず、ただ描けないビューが選択肢に
  // 並んでいた。
  check("the page has a target loader (a path box and a button)",
    el("target-path") !== null && el("btn-load-target") !== null,
    `path=${el("target-path") !== null} button=${el("btn-load-target") !== null}`);

  // A matrix is loaded first so that "the loader did not fetch the view" can
  // be false.
  //
  // `reload()` returns at its first line while `state.matrixId` is null, so in
  // a scenario that never loads one the view counter stays at 0 no matter what
  // the loaders do -- and `viewRequestCount() === 0` below would be restating
  // this scenario's starting conditions rather than checking any behaviour of
  // the loader. It was: adding `applyServerParams()` to `loadTargetImage` left
  // all twelve checks here passing (measured), i.e. the "loading stages, it
  // does not render" rule was unguarded on the one path this task created.
  //
  // 先に行列を1つ読み込む。行列が無いと `reload()` は先頭で打ち切るため、
  // 読み込み側が何をしても取得数は 0 のままで、下の検査はシナリオの初期条件を
  // 述べているだけになる（実際そうだった: 読み込みに `applyServerParams()` を
  // 足しても12件全部通ることを実測）。
  clickLoad("A");
  await settled();

  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await settled();
  check("the compensation view's ターゲットID box exists to be filled",
    el("sp-target") !== null && el("sp-target").value === "",
    `value=${el("sp-target") && el("sp-target").value}`);

  // The counter is live: it has already moved once, so it is capable of moving
  // again. Without this the fix above could rot back into the state it came
  // from -- a matrix that silently stopped loading would put the scenario back
  // to "0 forever" and the check below would go green for the old wrong reason.
  // 取得数が実際に動くことをここで固定する。行列の読み込みが静かに壊れると
  // 「常に0」へ戻り、下の検査がまた別の理由で通ってしまう。
  const viewFetchesBeforeLoading = viewRequestCount();
  check("a matrix is loaded, so a fetch from a loader would be counted",
    state.matrixId !== null && viewFetchesBeforeLoading > 0,
    `matrixId=${state.matrixId} view fetches=${viewFetchesBeforeLoading}`);

  // --- 1. loading writes the id the server returned into the parameter box
  //
  // The id is checked against `target_one_0`, which the fake server derives
  // from the path and the index. A client that invented an id, echoed the
  // path, or left the box alone and only remembered the id internally fails
  // this -- and the last of those is the one worth naming: it looks like it
  // works from the outside and leaves the user with an empty box.
  // IDはサーバがパスとindexから作った値と突き合わせる。IDを勝手に作る実装も、
  // 内部に覚えるだけで欄を埋めない実装も落ちる（後者は外からは動いて見えて、
  // 利用者の手元には空欄が残る）。
  el("target-path").value = "data/one.png";
  el("btn-load-target").__fire("click");
  await waitFor(() => el("sp-target").value !== "");
  check("loading a target fills the ターゲットID box with the id the server returned",
    el("sp-target").value === "target_one_0"
      && targetPostLog.length === 1
      && targetPostLog[0].path === "data/one.png",
    `box=${el("sp-target").value} posts=${JSON.stringify(targetPostLog)}`);

  // The info line beside the button: the only place the loaded image's *size*
  // appears anywhere on the page. The fixture is 5 wide and 3 high, so a line
  // built from `meta.shape` in the order it arrives (H, W) prints "3×5" --
  // plausible, and wrong about the one fact the line exists to state. Nothing
  // looked at this line at all until now.
  // ボタン横の情報行。読み込んだ画像の大きさが出るのはここだけである。
  // フィクスチャは横5・縦3なので、`meta.shape` の並び順のまま出す実装は
  // 「3×5」と書く——もっともらしく、この行が述べるべき唯一の事実について誤る。
  check("the target info line names the id and the image's width x height",
    el("target-info").textContent === "target_one_0 / 5×3",
    `info=${el("target-info").textContent}`);

  // A single file has one selectable entry, so the picker has nothing to offer.
  check("a single file shows no image picker",
    el("target-images").hidden === true, `hidden=${el("target-images").hidden}`);

  // --- 2. a directory lists its images, and choosing one re-fetches by index
  el("target-path").value = "data/dir";
  el("btn-load-target").__fire("click");
  await waitFor(() => el("sp-target").value === "target_dir_0");
  const listed = el("target-images").children.map((o) => o.textContent);
  check("a directory lists its images and selects the first",
    el("target-images").hidden === false
      && JSON.stringify(listed) === JSON.stringify(["a.png", "b.png", "c.png"])
      && el("target-images").value === "0",
    `hidden=${el("target-images").hidden} listed=${JSON.stringify(listed)} `
      + `value=${el("target-images").value}`);

  // Both halves asserted: a *new* POST carrying index 1 (not a locally derived
  // id, not a re-post of index 0), and the box following it. A client that
  // computed the next id itself would satisfy the box half and issue no
  // request; one that re-posted the path without the index would issue a
  // request and load image 0 while the dropdown claimed image 1.
  // 両方向を見る。index 1 の新しいPOSTが飛ぶことと、欄がその結果に従うこと。
  // IDを自前で計算する実装は前者を、indexを付けずに投げ直す実装は後者を落とす。
  const postsBefore = targetPostLog.length;
  el("target-images").value = "1";
  el("target-images").__fire("change");
  await waitFor(() => el("sp-target").value === "target_dir_1");
  const reposted = targetPostLog[targetPostLog.length - 1];
  check("choosing another image re-fetches that index and re-fills the id",
    targetPostLog.length === postsBefore + 1
      && reposted.path === "data/dir" && Number(reposted.index) === 1
      && el("sp-target").value === "target_dir_1",
    `posts=${postsBefore} -> ${targetPostLog.length} last=${JSON.stringify(reposted)} `
      + `box=${el("sp-target").value}`);

  // Loading stages the id; it does not render. Only 適用 reaches the view
  // endpoint -- see the note above `refreshParamsPending` in main.js and the
  // measurement of what a per-control fetch costs on the 1080p matrix (~740 ms
  // and ~390 MB peak per render, most of it discarded by reqId afterwards, so
  // the screen looks right and only the machine knows).
  // Measured against the count taken with a matrix already loaded, not against
  // zero -- see the note where `viewFetchesBeforeLoading` is taken.
  // 読み込みは値を置くだけで再描画しない（ビュー取得へ行ってよいのは 適用 だけ）。
  check("loading a target does not fetch the view on its own",
    viewRequestCount() === viewFetchesBeforeLoading,
    `view fetches=${viewFetchesBeforeLoading} -> ${viewRequestCount()}`);

  // --- 3. a refusal is prose, and does not take the loader down with it
  //
  // The status bar is shared with everything else on this page, so a 404 shown
  // as `{"detail":"no image at \"data/missing.png\""}` is not merely ugly: it
  // is the one sentence that says which path was wrong, rendered as a JSON
  // envelope in a bar that is otherwise Japanese prose. And a loader left
  // disabled by a failed load is a viewer that cannot be recovered without a
  // page reload, which is why the button state and a subsequent success are
  // both part of this check.
  // 404は「どのパスが違ったか」を述べる唯一の文である。JSONの外側ごと出れば
  // 読めず、失敗でボタンが無効のままなら再読み込みするまで復帰できない。
  const idBeforeFailure = el("sp-target").value;
  el("target-path").value = "data/missing.png";
  el("btn-load-target").__fire("click");
  await waitFor(() => /no image at/.test(statusText()));
  check("a 404 is shown as its sentence, not as a JSON envelope",
    statusText().includes('no image at "data/missing.png"')
      && !/\{"detail"|\\"/.test(statusText())
      && el("status").classList.contains("error"),
    `status=${statusText()} class=${el("status").className}`);
  check("a refused load leaves the loader usable and the previous id intact",
    el("btn-load-target").disabled === false
      && el("target-images").disabled === false
      && el("sp-target").value === idBeforeFailure,
    `button disabled=${el("btn-load-target").disabled} `
      + `picker disabled=${el("target-images").disabled} box=${el("sp-target").value}`);

  el("target-path").value = "data/one.png";
  el("btn-load-target").__fire("click");
  await waitFor(() => el("sp-target").value === "target_one_0");
  check("a load after a refused one still works",
    el("sp-target").value === "target_one_0"
      && !el("status").classList.contains("error"),
    `box=${el("sp-target").value} class=${el("status").className}`);

  // --- and the id survives a view switch, because the controls do not
  //
  // `buildServerParams` rebuilds every `sp-*` from the spec's declared
  // defaults, and for a runtime id that default is "". Without the state the
  // loader keeps, looking at `surface` and coming back would empty the box for
  // a target that is still loaded server-side, and the view would refuse a
  // request the user has no reason to think incomplete.
  // コントロールはビュー切替のたびに宣言側の既定（空文字）から作り直される。
  // 読み込み側が状態を保持していないと、サーバには残っているターゲットについて
  // 欄だけが空に戻り、利用者から見て理由の無い拒否になる。
  el("view-select").value = "surface";
  el("view-select").__fire("change");
  await sleep(30);
  check("switching away from compensation drops its controls", el("sp-target") === null);
  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await sleep(30);
  check("coming back re-fills the ターゲットID box with the loaded id",
    el("sp-target") !== null && el("sp-target").value === "target_one_0",
    `box=${el("sp-target") && el("sp-target").value}`);
}

if (SCENARIO === "lut") {
  check("the page has a LUT loader and a curve canvas",
    el("lut-path") !== null && el("btn-load-lut") !== null
      && el("lut-curve") !== null,
    `path=${el("lut-path") !== null} button=${el("btn-load-lut") !== null} `
      + `canvas=${el("lut-curve") !== null}`);

  // A matrix first, for the same reason the `target` scenario loads one: with
  // `state.matrixId` null, `reload()` returns at its first line and the view
  // counter cannot move, so "loading a LUT does not fetch the view" would be a
  // statement about this scenario's starting conditions. Adding
  // `applyServerParams()` to `loadInverseLut` left all twelve checks here
  // passing (measured).
  // 行列を先に読む（理由は `target` シナリオと同じ）。
  clickLoad("A");
  await settled();

  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await settled();
  check("the compensation view's LUT ID box exists to be filled",
    el("sp-lut") !== null && el("sp-lut").value === "",
    `value=${el("sp-lut") && el("sp-lut").value}`);

  const viewFetchesBeforeLoading = viewRequestCount();
  check("a matrix is loaded, so a fetch from a loader would be counted",
    state.matrixId !== null && viewFetchesBeforeLoading > 0,
    `matrixId=${state.matrixId} view fetches=${viewFetchesBeforeLoading}`);

  // The overlay checkbox is `checked` in index.html, but the harness's parser
  // deliberately records only tags, classes, ids and inline styles -- adding
  // boolean attributes would change what every existing scenario sees at
  // startup. So the state this scenario needs is set here, explicitly.
  // 重ね描きのチェックボックスは index.html では既定でオンだが、ハーネスの
  // パーサは真偽属性を記録しない（既存シナリオの初期状態が変わるため）。
  // このシナリオが必要とする状態はここで明示的に設定する。
  el("lut-gamma-overlay").checked = true;

  el("lut-path").value = "data/inverse_lut.npz";
  el("btn-load-lut").__fire("click");
  await waitFor(() => el("sp-lut").value !== "");
  check("loading a LUT fills the LUT ID box and fetches its curve",
    el("sp-lut").value === "lut_measured"
      && lutPostLog.length === 1 && lutPostLog[0].path === "data/inverse_lut.npz"
      && lutCurveLog.length === 1 && lutCurveLog[0] === "/api/lut/lut_measured/curve",
    `box=${el("sp-lut").value} posts=${JSON.stringify(lutPostLog)} `
      + `curves=${JSON.stringify(lutCurveLog)}`);

  // The two lines of text beside and under the plot, neither of which anything
  // read before. The info line is where "which LUT is this, and how was it
  // measured" is answered -- `source` decides whether the curve describes a
  // grey ramp or per-channel ramps, and those disagree by up to 1.7x on this
  // projector, so a blank or wrong line is a plot with no provenance.
  // プロットの脇と下にある2行。どちらもこれまで誰も読んでいなかった。
  check("the LUT info line names the id, the table's shape and its source",
    el("lut-info").textContent === "lut_measured / 3×5 / channel_ramp",
    `info=${el("lut-info").textContent}`);
  // The exponent the dashed overlay is drawn at, in words next to its
  // checkbox. Asserted against the fixture's 1.9 rather than a formatting
  // pattern, so a label that printed the usual 2.2 while the curve was drawn
  // at the server's exponent -- a legend that disagrees with the line it
  // labels -- fails here.
  // 重ね描きの指数を文字で示す行。既定の 2.2 ではなくフィクスチャの 1.9 と
  // 突き合わせるので、線と凡例が食い違う実装が落ちる。
  check("the analytic gamma's exponent is shown, and it is the server's",
    el("lut-gamma-value").textContent === `γ=${LUT_GAMMA}`,
    `label=${el("lut-gamma-value").textContent} expected γ=${LUT_GAMMA}`);

  // --- 1. three curves, at the coordinates the three channels map to
  //
  // Counting lines is not enough and the brief says so: three lines drawn at
  // identical coordinates, or drawn for the wrong channels, satisfy a count
  // just as well as the right ones. Each polyline is therefore compared point
  // for point against its own channel's values (see `expectedPoint` for which
  // half of that comparison comes from the implementation and which from the
  // fixture). The three fixture curves have three different shapes and three
  // different endpoints, so no permutation of them agrees with another.
  // 本数だけでは足りない。各折れ線をチャンネルごとの値と1点ずつ突き合わせる。
  // 3本は形も終端も全て違うので、どの入れ替えも一致しない。
  const withOverlay = lutPathsSinceLastClear(el("lut-curve"));
  const plot = lutPlot();
  const channelsOk = LUT_ROWS.every(
    (row, c) => withOverlay[c] && pointsMatch(withOverlay[c].points, plot, row));
  check("each of the three curves is drawn at its own channel's coordinates",
    withOverlay.length === 4 && channelsOk,
    `paths=${withOverlay.length} plot=${JSON.stringify(plot)} `
      + `points=${JSON.stringify(withOverlay.map((p) => p.points.slice(0, 2)))}`);

  // Three curves that cannot be told apart on screen are one curve as far as
  // the reader is concerned.
  // 画面上で区別が付かない3本は、読む側にとっては1本と同じである。
  const styles = withOverlay.map((p) => p.style);
  check("the curves are drawn in distinguishable colours",
    new Set(styles).size === withOverlay.length && styles.every(Boolean),
    JSON.stringify(styles));

  // --- 2. the vertical axis comes from the data, not from 1.0
  //
  // The fixture's largest value is 0.688 and none of its channels reaches 1.0,
  // which is the case a measured LUT actually presents (the ISO400 file in
  // this repository stops at 0.813 / 0.876 / 0.751). Asserted against the
  // fixture's own maximum rather than against what the plot returned, so an
  // axis pinned to [0, 1] cannot satisfy it by being internally consistent.
  // フィクスチャの最大値は 0.688 で、どのチャンネルも 1.0 に届かない（実測LUT
  // が実際にそうである）。戻り値同士ではなくフィクスチャの最大値と突き合わせる
  // ので、[0, 1] に固定した実装が「自分と整合している」だけで通ることはない。
  const fixtureMax = Math.max(...LUT_ROWS.flat());
  const fixtureMin = Math.min(...LUT_ROWS.flat());
  check("the plot's vertical axis comes from the data, not pinned to 1.0",
    !!plot && closeEnough(plot.vMax, fixtureMax) && plot.vMax !== 1
      && closeEnough(plot.vMin, Math.min(0, fixtureMin)),
    `vMin=${plot && plot.vMin} vMax=${plot && plot.vMax} `
      + `(fixture ${fixtureMin}..${fixtureMax})`);

  // --- 3. the analytic gamma is a fourth line, not a redrawn LUT
  //
  // Both directions. The overlay has to be its *own* line -- at t ** (1 /
  // gamma), with the exponent the server sent rather than a hard-coded 2.2 --
  // and turning it off has to remove exactly that one line and leave the three
  // measured ones untouched. A build that drew it into one of the channels, or
  // that drew nothing extra at all, fails one of the two.
  // 両方向。重ね描きは独立した1本（指数はサーバが送った値であり、直書きの
  // 2.2 ではない）であり、切ると**その1本だけ**が消えて実測3本は変わらない。
  const overlayPath = withOverlay[3];
  const analytic = LUT_T_GRID.map((t) => Math.pow(t, 1 / LUT_GAMMA));
  check("the analytic gamma is drawn as a separate line, at the server's exponent",
    !!overlayPath && pointsMatch(overlayPath.points, plot, analytic)
      && !LUT_ROWS.some((row) => pointsMatch(overlayPath.points, plot, row)),
    `overlay=${overlayPath && JSON.stringify(overlayPath.points.slice(0, 2))} `
      + `expected=${JSON.stringify(analytic.slice(0, 2))}`);

  // Compared against the three measured lines from the draw above rather than
  // against the fixture again: what this asserts is that switching the overlay
  // off removed the overlay and *only* the overlay. Re-deriving the channels
  // here would make this a second copy of the correspondence check above, and
  // every mutation of the channel order would then fail two checks that say
  // the same thing.
  // 直前の描画の3本と突き合わせる（フィクスチャから引き直さない）。ここで
  // 見たいのは「重ね描きだけが消えたこと」であり、チャンネルの対応を再導出
  // すると上のチェックの写しになる。
  const measuredBefore = JSON.stringify(withOverlay.slice(0, 3).map((p) => p.points));
  el("lut-gamma-overlay").checked = false;
  el("lut-gamma-overlay").__fire("change");
  await sleep(20);
  const withoutOverlay = lutPathsSinceLastClear(el("lut-curve"));
  check("turning the overlay off removes exactly that line",
    withoutOverlay.length === 3
      && JSON.stringify(withoutOverlay.map((p) => p.points)) === measuredBefore,
    `paths=${withoutOverlay.length}`);

  // --- 4. the backing store follows the device pixel ratio
  //
  // A canvas sized by CSS is drawn into a backing store of CSS size x DPR, and
  // without the term the whole plot goes into a quarter of the canvas on a
  // 200% display and is then stretched to fit -- three curves and a scale, at
  // half resolution, on the machine this is used on (Windows, 150%). Both
  // directions are checked, since "always 1x" and "always 2x" each pass one.
  // CSSで伸縮するcanvasのバッキングストアは CSSサイズ×DPR で取る。DPR項が無い
  // と 200% 表示で図がcanvasの1/4に描かれて引き伸ばされる。常に1倍・常に2倍の
  // どちらも片側だけなら通るので両方見る。
  const canvas = el("lut-curve");
  const at1 = { width: canvas.width, height: canvas.height };
  globalThis.window.devicePixelRatio = 2;
  el("lut-gamma-overlay").__fire("change");
  await sleep(20);
  check("the canvas backing store scales with devicePixelRatio",
    at1.width === canvas.clientWidth && at1.height === canvas.clientHeight
      && canvas.width === canvas.clientWidth * 2
      && canvas.height === canvas.clientHeight * 2,
    `dpr1=${at1.width}x${at1.height} dpr2=${canvas.width}x${canvas.height} `
      + `css=${canvas.clientWidth}x${canvas.clientHeight}`);
  globalThis.window.devicePixelRatio = 1;

  // --- and the plot is still in CSS pixels afterwards
  //
  // Scaling the backing store without scaling the context draws the same
  // figure at half size in the corner of a canvas twice as large. The
  // coordinates recorded here are the ones passed to `moveTo`/`lineTo`, i.e.
  // before the transform, so they must not change with DPR at all.
  // バッキングストアだけ拡げて座標系を拡大しないと、2倍のcanvasの隅に同じ図が
  // 半分の大きさで描かれる。記録される座標は変換前の値なので、DPRで変わっては
  // ならない。
  const afterDpr = lutPathsSinceLastClear(canvas);
  check("the curve coordinates stay in CSS pixels across a DPR change",
    afterDpr.length === 3
      && JSON.stringify(afterDpr.map((p) => p.points)) === measuredBefore,
    `paths=${afterDpr.length}`);

  // --- a refused LUT load says why and leaves the loader usable
  const idBeforeFailure = el("sp-lut").value;
  el("lut-path").value = "data/missing.npz";
  el("btn-load-lut").__fire("click");
  await waitFor(() => /no inverse LUT at/.test(statusText()));
  check("a refused LUT load is prose, keeps the previous id and leaves the button usable",
    statusText().includes('no inverse LUT at "data/missing.npz"')
      && !/\{"detail"|\\"/.test(statusText())
      && el("btn-load-lut").disabled === false
      && el("sp-lut").value === idBeforeFailure,
    `status=${statusText()} disabled=${el("btn-load-lut").disabled} `
      + `box=${el("sp-lut").value}`);

  // --- a LUT whose *curve* could not be fetched is not published either
  //
  // The other half of a two-request load, and the only state in which
  // `loadInverseLut`'s publication order is observable at all: the POST
  // registers `lut_curveless` and the GET for its curve comes back 500 (see
  // `LUTS` for why that is an ordinary thing to happen and what it costs).
  //
  // Three things are asserted together, because each is survivable alone and
  // the failure is the disagreement between them: the id box still names the
  // LUT that is plotted, `state.lutId` -- which is what 適用 sends -- agrees
  // with the box, and the canvas still holds the previous LUT's three curves.
  // Publishing the id before the curve satisfies none of the first two while
  // the third stays true, which is exactly the "the heading and the numbers
  // disagree" failure this branch has closed everywhere else.
  //
  // 2つの要求のうち後半だけが失敗する場合。`loadInverseLut` の公開順が外から
  // 見えるのはこの状態だけである。欄・`state.lutId`・canvas の3つを同時に見る
  // （食い違いこそが失敗であり、単独ではどれも生き延びる）。
  const idBeforeCurveFailure = el("sp-lut").value;
  const curveBeforeFailure =
    JSON.stringify(lutPathsSinceLastClear(el("lut-curve")).map((p) => p.points));
  const curveFetchesBefore = lutCurveLog.length;
  el("lut-path").value = "data/curveless.npz";
  el("btn-load-lut").__fire("click");
  await waitFor(() => /could not read the curve/.test(statusText()));

  // The fixture has to actually reach the state being checked. If the POST
  // ever started failing for this path, everything below would pass for the
  // reason the 404 case above already covers and this case would quietly stop
  // being about ordering at all.
  // フィクスチャがその状態に本当に到達していることを固定する。POSTが失敗する
  // ようになったら、以下は404の場合と同じ理由で通り、順序の検査ではなくなる。
  check("the LUT POST succeeded and only its curve fetch failed",
    lutPostLog[lutPostLog.length - 1].path === "data/curveless.npz"
      && lutCurveLog.length === curveFetchesBefore + 1
      && lutCurveLog[lutCurveLog.length - 1] === "/api/lut/lut_curveless/curve",
    `posts=${JSON.stringify(lutPostLog)} curves=${JSON.stringify(lutCurveLog)}`);

  const curveAfterFailure =
    JSON.stringify(lutPathsSinceLastClear(el("lut-curve")).map((p) => p.points));
  check("a LUT whose curve could not be fetched is not named as loaded",
    el("sp-lut").value === idBeforeCurveFailure
      && state.lutId === idBeforeCurveFailure
      && curveAfterFailure === curveBeforeFailure
      && el("btn-load-lut").disabled === false,
    `box=${el("sp-lut").value} state.lutId=${state.lutId} `
      + `(expected ${idBeforeCurveFailure}) curve unchanged=`
      + `${curveAfterFailure === curveBeforeFailure} `
      + `disabled=${el("btn-load-lut").disabled}`);

  check("the failed curve fetch is shown as its sentence, not as a JSON envelope",
    statusText().includes('could not read the curve for "lut_curveless"')
      && !/\{"detail"|\\"/.test(statusText())
      && el("status").classList.contains("error"),
    `status=${statusText()} class=${el("status").className}`);

  // Measured against the count taken with a matrix already loaded, not against
  // zero -- see `viewFetchesBeforeLoading` above.
  check("loading a LUT does not fetch the view on its own",
    viewRequestCount() === viewFetchesBeforeLoading,
    `view fetches=${viewFetchesBeforeLoading} -> ${viewRequestCount()}`);
}

if (SCENARIO === "generate") {
  // The button that writes to disk, and the one property everything here is
  // about: **what it sends is what is on screen.**
  //
  // Every other check in this file is about a picture. This one is about a
  // file, and that changes what a wrong answer costs. A tile drawn from the
  // wrong channel is wrong until the next render; a PNG written from a stale
  // target id, a defaulted linearizer or an unapplied decode mode is wrong on
  // the user's disk, under a name that says it is the picture they were
  // looking at, and there is nothing on screen afterwards to say otherwise.
  //
  // "Pressing the button issues a fetch" does not test this, and cannot: a
  // build that sent `linearizer: "none"` and the first image of the directory
  // would satisfy it exactly as well as a correct one. So the body is compared
  // field for field, and **every fixture value below is moved off its own
  // declared default first** -- the target is image 2 of a directory (default
  // `""`), the linearizer is `inverse_lut` (default `"none"`), the decode mode
  // is `gamma` (default `"srgb"`) and its exponent 1.7 (default 2.2). An
  // expected value that coincided with a default would pass against a body
  // built entirely of hard-coded defaults.
  //
  // 画面に出ている設定がそのまま送られること。他のチェックが相手にしている
  // のは絵だが、ここで間違えると誤りはディスク上のファイルとして残り、しかも
  // 「見ていた絵」の名前で残る。「押すと fetch が飛ぶ」ではこれは検証できない
  // ——既定値だけで組んだ本文でも同じように通る。よって本文をフィールド単位で
  // 突き合わせ、フィクスチャの値は**すべて既定値から動かしてから**押す。
  check("the page has a generate control (an output directory and a button)",
    el("gen-output-dir") !== null && el("btn-generate") !== null,
    `dir=${el("gen-output-dir") !== null} button=${el("btn-generate") !== null}`);

  // The line beside the button as the page first paints, read before anything
  // is loaded and asserted at the end of the scenario together with its two
  // other states.
  //
  // Nothing is displayed yet here, so the obstacle is "load a matrix" and the
  // remedy is the path box at the top of the sidebar. Printing the *view*
  // refusal instead is not a small inaccuracy -- it tells the user to switch
  // view, which does nothing, and it spends the one sentence that has to mean
  // something under `gamut` on a state where it is false. Measured before this
  // was fixed: 「このビューでは生成できません」 from the first paint onward, and
  // then on every view that *can* generate, beside an enabled button.
  // 起動直後の行。まだ何も表示されていないので障害は「行列を読み込むこと」で
  // あり、ここでビューの断り文を出すのは「ビューを切り替えろ」と言うに等しい。
  const genInfoAtFirstPaint = el("gen-info").textContent;

  clickLoad("A");
  await settled();
  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await settled();

  // The two runtime ids, obtained the way a user obtains them: through the
  // sidebar loaders, from the server. Not typed in -- an id this harness made
  // up would be indistinguishable from one the frontend made up.
  // 2つの実行時IDは利用者と同じ手順（読み込み欄経由でサーバから）取る。
  // ここで直接打ち込むと、フロントが勝手に作ったIDと区別が付かない。
  el("target-path").value = "data/dir";
  el("btn-load-target").__fire("click");
  await waitFor(() => el("sp-target").value === "target_dir_0");
  // The *third* image, not the first: the id a client that ignored the picker
  // would send is `target_dir_0`, which is a real id for the wrong image.
  // 3枚目を選ぶ。選択を無視した実装が送る `target_dir_0` は「別の画像の、
  // 実在するID」であり、最も気付きにくい誤りである。
  el("target-images").value = "2";
  el("target-images").__fire("change");
  await waitFor(() => el("sp-target").value === "target_dir_2");

  el("lut-path").value = "data/inverse_lut.npz";
  el("btn-load-lut").__fire("click");
  await waitFor(() => el("sp-lut").value === "lut_measured");

  stageParam("linearizer", "inverse_lut");
  stageParam("target_decode", "gamma");
  stageParam("target_gamma", "1.7");
  // Staged and applied like the rest, and deliberately *not* expected in the
  // body: `POST /api/generate` writes both the linear compensation image and
  // the projector image every time, so which of the two the viewer happens to
  // be showing does not select anything. A body carrying `output` would be a
  // field the endpoint does not declare -- which the exact-key comparison
  // below is what catches.
  // `output` は適用するが本文には現れない。生成は毎回2枚とも書くので選ぶ
  // ものが無い。余計なフィールドは下のキー集合の比較が捉える。
  stageParam("output", "code");
  applyParams();
  await settled();

  // --- 1. the body is the settings on screen
  el("gen-output-dir").value = "data/out_generated";
  el("btn-generate").__fire("click");
  await waitFor(() => generatePostLog.length === 1);
  await waitFor(() => el("btn-generate").disabled === false);

  const sent = generatePostLog[0];
  const expected = {
    matrix_id: "matrix_A",
    target_id: "target_dir_2",
    linearizer: "inverse_lut",
    lut_id: "lut_measured",
    target_decode: "gamma",
    target_gamma: 1.7,
    output_dir: "data/out_generated",
    // The endpoint's own default, restated here because it is the field whose
    // silent flip is unrecoverable: `overwrite: true` on a first press turns
    // the 409 guard off for a user who never asked for it.
    // 既定値であっても書く。ここが黙って true になると、409 のガードは
    // 誰も頼んでいないのに外れる。
    overwrite: false,
  };
  check("the generate call sends the settings on screen, field for field",
    canonical(sent) === canonical(expected),
    `sent=${canonical(sent)} expected=${canonical(expected)}`);

  // The paths that come back, which are the server's absolute ones and not the
  // relative string in the box. Reporting the box back would look right on
  // every machine where the two happen to agree, and this endpoint's whole
  // safety argument is that a path it names is a path the user can find.
  // 返ってきた絶対パスを出すこと（入力欄の文字列ではない）。
  const shownFiles = el("gen-files").children.map((c) => c.textContent);
  check("the saved paths shown are the absolute ones the server reported",
    JSON.stringify(shownFiles) === JSON.stringify(generateFiles("data/out_generated"))
      && statusText().includes(generateAbsolute("data/out_generated")),
    `files=${JSON.stringify(shownFiles)} status=${statusText()}`);

  // ...and the line beside the button says a save happened.
  //
  // The other half of §8's line, and the half that was unasserted after the
  // refusal half was fixed: `refreshGenerateButton` was corrected to write
  // `generateOutcome` on both branches, but *what that variable holds after a
  // success* was read by nothing. Deleting the success `setGenerateOutcome`
  // left all 17 checks green (measured), and what it produces is the exact
  // inverse of the defect that fix was for: three absolute paths appear under a
  // line still reading 未生成, because the `finally`'s
  // `refreshGenerateButton()` restores the stale outcome over 「生成中…」.
  //
  // Read again after a round trip through a view that cannot generate, because
  // that is the path that overwrites the line -- a success asserted only
  // immediately after the press would not see the restore at all.
  // 断りの側を直したときに、成功の側は誰も読んでいなかった。成功時の
  // `setGenerateOutcome` を消しても17件すべて緑だった（実測）。生じるのは
  // 直した欠陥のちょうど裏返しで、3つの絶対パスの脇に 未生成 と出る
  // （`finally` の `refreshGenerateButton()` が古い結果で「生成中…」を上書き
  // するため）。行を書き換えるのはその復元経路なので、ビューを往復してから
  // もう一度読む。
  const genInfoAfterSuccess = el("gen-info").textContent;
  el("view-select").value = "gamut";
  el("view-select").__fire("change");
  await settled();
  const genInfoOnGamutAfterSuccess = el("gen-info").textContent;
  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await settled();
  check("a successful save is what the line beside 生成 says, and keeps saying",
    genInfoAfterSuccess === "3 件を保存"
      && genInfoOnGamutAfterSuccess === "このビューでは生成できません"
      && el("gen-info").textContent === "3 件を保存"
      && el("btn-generate").disabled === false,
    `after the press="${genInfoAfterSuccess}" on gamut="${genInfoOnGamutAfterSuccess}" `
      + `back on compensation="${el("gen-info").textContent}" `
      + `disabled=${el("btn-generate").disabled}`);

  // --- 2. "on screen" means the picture, not the boxes
  //
  // A staged-but-unapplied control is ahead of what is displayed: the tiles
  // still show the previous value and 適用 is ringed to say so. Generating
  // there would write a file that matches neither -- it would match the boxes,
  // which is the one thing nobody is looking at. Both directions are checked,
  // because "always refuse" passes the first half alone.
  // 未適用のコントロールは表示より先に進んでいる。そこで生成すると、絵とも
  // 一致せず、誰も見ていない入力欄とだけ一致するファイルが書かれる。
  // 「常に断る」実装でも前半は通るので両方向を見る。
  stageParam("target_gamma", "1.3");
  const postsBeforePending = generatePostLog.length;
  el("btn-generate").__fire("click");
  await sleep(60);
  check("生成 refuses while a staged change has not been applied, and says why",
    generatePostLog.length === postsBeforePending
      && /画面に出ている絵/.test(statusText())
      && el("status").classList.contains("error"),
    `posts=${postsBeforePending} -> ${generatePostLog.length} status=${statusText()}`);

  applyParams();
  await settled();
  el("btn-generate").__fire("click");
  await waitFor(() => generatePostLog.length === postsBeforePending + 1);
  check("after 適用 the newly displayed value is the one that is sent",
    generatePostLog[postsBeforePending].target_gamma === 1.3,
    `target_gamma=${generatePostLog[postsBeforePending].target_gamma}`);
  await waitFor(() => el("btn-generate").disabled === false);

  // --- 2b. ...and "on screen" means the matrix too, which no guard covers
  //
  // `POST /session/matrix` returns an id in milliseconds; the transfer that
  // draws it takes seconds on a 1.56 GB matrix. In between, `state.matrixId`
  // names the new one, the tiles still show the old one, and **not one control
  // has changed** -- so the staged-vs-applied guard above passes and the press
  // goes through. A body built from `state.matrixId` there compensates the
  // picture the user is looking at with a matrix they have never seen, and the
  // file is filed under the name of the one they were shown.
  //
  // This is the field-for-field check's blind spot: there the two ids are equal
  // and the mutation is invisible. Here they are deliberately different, which
  // is the only state that can tell them apart. Measured before this check
  // existed: `matrix_id: state.shownMatrixId` -> `state.matrixId` failed 0 of
  // the scenario's checks.
  //
  // 「画面に出ているもの」は行列にも及ぶ。`POST /session/matrix` はすぐ返るが
  // 描画までは数秒あり、その間コントロールは1つも変わらないので上のガードは
  // 通す。ここで `state.matrixId` を送ると、見ている絵を「見たことのない行列」
  // で補償したファイルが、見ていた絵の名前で残る。2つのIDが等しくない状態は
  // ここでしか作れない（フィールド単位の検査では両者が一致しており見えない）。
  viewDelayMs.B = 400;
  clickLoad("B");
  await waitFor(() => state.matrixId === "matrix_B");
  const shownWhileLoading = state.shownMatrixId;
  const registeredWhileLoading = state.matrixId;
  el("gen-output-dir").value = "data/out_during_load";
  el("gen-output-dir").__fire("input");
  const postsBeforeInFlight = generatePostLog.length;
  el("btn-generate").__fire("click");
  await waitFor(() => generatePostLog.length === postsBeforeInFlight + 1);
  const duringLoad = generatePostLog[postsBeforeInFlight];
  check("a press while a matrix is still loading saves the matrix on screen",
    registeredWhileLoading === "matrix_B"
      && shownWhileLoading === "matrix_A"
      && duringLoad.matrix_id === "matrix_A",
    `registered=${registeredWhileLoading} shown=${shownWhileLoading} `
      + `sent=${duringLoad.matrix_id}`);
  // Let B land before going on, so every later press is about a settled screen.
  await waitFor(() => state.shownMatrixId === "matrix_B");
  viewDelayMs.B = 0;
  await waitFor(() => el("btn-generate").disabled === false);

  // --- 3. one press is one generation
  //
  // The call takes seconds on a real matrix (the pipeline runs, two PNGs are
  // encoded), which is exactly long enough for a second click. Two overlapping
  // writes into the same directory is not a slow duplicate: the second one
  // meets the first one's files and comes back 409, so the visible result of
  // an impatient double-click is a conflict about a directory the user is in
  // the middle of filling.
  // 実機では数秒かかり、その間に2度目を押せてしまう。同じディレクトリへの
  // 二重書き込みは「遅い重複」では済まない——2つ目は1つ目のファイルに衝突して
  // 409 になり、書いている最中のディレクトリについて衝突を告げられる。
  holdGenerate = true;
  const postsBeforeDouble = generatePostLog.length;
  el("gen-output-dir").value = "data/out_twice";
  el("gen-output-dir").__fire("input");
  el("btn-generate").__fire("click");
  await waitFor(() => generatePostLog.length === postsBeforeDouble + 1);
  const disabledDuring = el("btn-generate").disabled;
  el("btn-generate").__fire("click");
  await sleep(60);
  check("the button is disabled while generating and a second press sends nothing",
    disabledDuring === true && generatePostLog.length === postsBeforeDouble + 1,
    `disabled during=${disabledDuring} posts=${postsBeforeDouble} -> `
      + `${generatePostLog.length}`);
  releaseGenerates();
  holdGenerate = false;
  await waitFor(() => el("btn-generate").disabled === false);

  // --- 4. a 409 is a question, not a retry
  //
  // `generate.py` refuses a non-empty directory and writes nothing -- not even
  // the directory itself -- because a mistyped path in this repository can name
  // 99 GB of captures that cost hardware, a subject and a day to reproduce. The
  // guard only works if the answer stays deliberate: a frontend that resent
  // with `overwrite: true` on its own, or that offered 上書き as the default
  // action, would turn a refusal that stops every typo into a dialog people
  // click through.
  // 409 は「やり直し」ではなく問いである。実物は空でないディレクトリへは
  // 何も書かない（打ち間違いが撮り直しの効かない実測データを指しうるため）。
  // 自動で再送する実装や、上書きを既定の動作にする実装は、あらゆる打ち間違いを
  // 止めるはずのガードを「押して通り過ぎるもの」に変えてしまう。
  el("gen-output-dir").value = "data/existing";
  el("gen-output-dir").__fire("input");
  const postsBefore409 = generatePostLog.length;
  el("btn-generate").__fire("click");
  await waitFor(() => generatePostLog.length === postsBefore409 + 1);
  await waitFor(() => el("gen-overwrite").hidden === false);
  // Long enough for an automatic resend to have landed. Nothing waits on a
  // second request here on purpose -- the assertion is that there isn't one.
  // 自動再送があれば届くだけの時間を置く。ここで2件目を待たないのは、
  // 「2件目が無いこと」が主張だからである。
  await sleep(80);
  const first409 = generatePostLog[postsBefore409];
  // The path list is cleared as part of this, and that is not decoration: it
  // describes *the last generation*, and this press did not produce one. Left
  // up, three absolute paths sit on screen at the moment nothing was written
  // -- and under a 409 they are (part of) what is in the way, so they read as
  // "here is what was just saved". Measured before this was fixed: the
  // successful run's three paths stayed on screen under 未生成.
  // 一覧が「直前の生成」を説明している以上、生成していない押下のあとに残っては
  // ならない。409 では、残った3つのパスは「邪魔をしている当のもの」であり、
  // 何も書かれていない瞬間に「いま保存されたもの」と読めてしまう。
  check("a 409 offers 上書き explicitly and resends nothing until it is confirmed",
    first409.overwrite === false
      && generatePostLog.length === postsBefore409 + 1
      && el("gen-overwrite").hidden === false
      && el("gen-overwrite-detail").textContent.includes("is not empty")
      && !/\{"detail"|\\"/.test(el("gen-overwrite-detail").textContent)
      && el("gen-files").children.length === 0,
    `overwrite=${first409.overwrite} posts=${postsBefore409} -> `
      + `${generatePostLog.length} hidden=${el("gen-overwrite").hidden} `
      + `files listed=${el("gen-files").children.length} `
      + `detail=${el("gen-overwrite-detail").textContent}`);

  // Confirming resends *the same request*, with one field changed. Compared
  // against the refused body rather than against the expectation above, so a
  // build that rebuilt the body from scratch -- and picked up whatever the
  // boxes hold now rather than what was refused -- is visible.
  // 確認すると「同じ要求」が1フィールドだけ変えて再送される。上の期待値では
  // なく断られた本文と突き合わせるので、組み直して現在の入力欄を拾う実装が
  // 見える。
  el("btn-generate-overwrite").__fire("click");
  await waitFor(() => generatePostLog.length === postsBefore409 + 2);
  // `?? null`, because a build that never resends leaves this undefined and
  // dereferencing it throws out of the scenario -- which pytest reports as
  // "the harness produced no result", detecting the regression while throwing
  // away the diagnosis. Measured: `overwrite: true` as a hardcoded default did
  // exactly that. A missing request is a failure of this check, with a detail
  // saying so.
  // 再送しない実装ではここが undefined になり、参照した瞬間にシナリオごと
  // 落ちる（pytest からは「結果が出なかった」に見え、診断が失われる）。
  const retried = generatePostLog[postsBefore409 + 1] ?? null;
  check("confirming 上書き resends the same request with overwrite set",
    retried !== null
      && retried.overwrite === true
      && canonical({ ...retried, overwrite: false }) === canonical(first409),
    `retried=${retried === null ? "(no second request was sent)" : canonical(retried)} `
      + `refused=${canonical(first409)}`);
  await waitFor(() => el("btn-generate").disabled === false);
  check("a confirmed overwrite takes the offer back down",
    el("gen-overwrite").hidden === true, `hidden=${el("gen-overwrite").hidden}`);

  // --- 5. a failure is prose, and the control survives it
  el("gen-output-dir").value = "Z:/nope";
  el("gen-output-dir").__fire("input");
  const postsBeforeUnwritable = generatePostLog.length;
  el("btn-generate").__fire("click");
  await waitFor(() => /could not write to/.test(statusText()));

  // The press immediately after a confirmed overwrite, which is the one that
  // says the answer was about that request and not a setting that was turned
  // on. A checkbox would have carried it forward; so would a remembered flag,
  // and the visible difference between the two designs is exactly this field
  // on exactly this request.
  //
  // It has its own check because the alternative is a wrong diagnosis: with
  // `overwrite` sticky, the 409 setups later in this scenario stop producing a
  // 409 at all, and the failure surfaces under the name of the `gamut` refusal
  // -- a headline constraint reported by an unrelated check.
  // 確認の直後の押下。「その要求についての答え」であって「入れた設定」では
  // ないことは、この要求のこのフィールドにしか現れない。専用のチェックにする
  // 理由は診断で、上書きが残る実装では以降の 409 が起きなくなり、失敗は
  // `gamut` の断りという無関係な名前で出る。
  const afterConfirm = generatePostLog[postsBeforeUnwritable] ?? null;
  check("the press after a confirmed 上書き starts from overwrite: false again",
    afterConfirm !== null && afterConfirm.overwrite === false,
    `after the confirmation: ${afterConfirm === null
      ? "(no request was sent)" : canonical(afterConfirm)}`);

  check("a failed generation is shown as its sentence and leaves the button usable",
    statusText().includes('could not write to "Z:/nope"')
      && !/\{"detail"|\\"/.test(statusText())
      && el("status").classList.contains("error")
      && el("btn-generate").disabled === false
      && el("gen-overwrite").hidden === true
      // Same property as the 409's, on the other branch: the paths from the
      // overwrite that just succeeded must not survive a press that wrote
      // nothing. 直前に成功した上書きのパスが、何も書かなかった押下のあとに
      // 残ってはならない。
      && el("gen-files").children.length === 0,
    `status=${statusText()} disabled=${el("btn-generate").disabled} `
      + `offer hidden=${el("gen-overwrite").hidden} `
      + `files listed=${el("gen-files").children.length}`);

  // --- 6. the confirmation is about one directory, and about no other
  //
  // 上書きして生成 is the only control in this application that destroys data,
  // and what makes it safe is not that it has to be pressed -- it is that the
  // sentence being agreed to names a place. An offer that outlived a change of
  // that box would carry an agreement about `data/out_A` onto whatever the box
  // says when the button is finally pressed, and in this repository the box can
  // say `data/full_recalibration_20260730_iso400`: a real 99 GB tree that cost
  // hardware, a subject and a day, and cannot be regenerated.
  //
  // Two independent mechanisms, so two arms. The listeners cover an edit, which
  // is what a person does; the directory comparison at press time covers a
  // value set **without an event** -- autofill, an extension, a driver, or a
  // refactor that drops one listener -- which no listener can see. Both were
  // argued at length in the report and neither was asserted anywhere: measured
  // on the shipped code, replacing the comparison with `if (false)` and
  // deleting both listeners each failed 0 of the scenario's checks.
  //
  // 確認を安全にしているのは「押させること」ではなく「同意する文が場所を名指し
  // していること」である。入力欄の変更を越えて残る確認は、Aについての同意を
  // 押下時点の値へ持ち越す——この repository ではその値が実測データ 99 GB の
  // ディレクトリでありうる。機構は2つあるので腕も2本: リスナは「人が編集した
  // 場合」を、押下時のディレクトリ照合は「イベントを伴わない設定」を覆う
  // （自動入力・拡張機能・自動操作・リスナを1つ落とす改修）。
  // Each arm waits on **its own** request rather than on the offer being
  // visible: an offer left standing by a previous arm satisfies that condition
  // before this press has even been issued, and the arm would then measure a
  // state it did not create. (Measured: with both listeners deleted, the
  // visibility-only wait made two unrelated checks fail as well, hiding which
  // guard the mutation had actually removed.)
  // 各腕は「自分が出した要求」を待つ（確認が出ていることだけを待つと、前の腕が
  // 残した確認で条件が既に満たされ、作っていない状態を測ることになる）。
  const offer409 = async () => {
    el("gen-output-dir").value = "data/existing";
    el("gen-output-dir").__fire("input");
    const before = generatePostLog.length;
    el("btn-generate").__fire("click");
    await waitFor(() => generatePostLog.length === before + 1);
    await waitFor(() => el("btn-generate").disabled === false);
    await waitFor(() => el("gen-overwrite").hidden === false);
  };

  await offer409();
  el("gen-output-dir").value = "data/out_typed_instead";
  el("gen-output-dir").__fire("input");
  const hiddenAfterInput = el("gen-overwrite").hidden;

  await offer409();
  el("gen-output-dir").value = "data/out_chosen_instead";
  // `change` as well as `input`: a value can arrive by either, and one listener
  // covering for the other is a coincidence of which events this stub fires.
  el("gen-output-dir").__fire("change");
  const hiddenAfterChange = el("gen-overwrite").hidden;
  check("editing the output directory retires the 上書き offer it was made for",
    hiddenAfterInput === true && hiddenAfterChange === true,
    `after input=${hiddenAfterInput} after change=${hiddenAfterChange}`);

  await offer409();
  // Set with no event at all, which is what every programmatic write looks
  // like. The offer is still on screen and still enabled; only the comparison
  // at press time stands between the agreement about `data/existing` and this.
  // イベントを伴わない設定。確認は画面に出たままで、押下時の照合だけが残る。
  el("gen-output-dir").value = "data/full_recalibration_20260730_iso400";
  const postsBeforeSilentSwap = generatePostLog.length;
  el("btn-generate-overwrite").__fire("click");
  await sleep(60);
  check("上書き agreed for one directory cannot be spent on another",
    generatePostLog.length === postsBeforeSilentSwap
      && el("gen-overwrite").hidden === true
      && /出力先が変わりました/.test(statusText())
      && el("status").classList.contains("error"),
    `posts=${postsBeforeSilentSwap} -> ${generatePostLog.length} `
      + `offer hidden=${el("gen-overwrite").hidden} status=${statusText()}`);

  // --- 7. a view that does not declare a linearizer does not generate
  //
  // The decision this task's brief leaves open, pinned as behaviour. `gamut`
  // has no 線形化 control and `views/gamut.py` argues at length that it must
  // not have one -- which pixels are out of gamut is a property of the
  // *unclipped linear* compensation, decided before any linearization. So
  // there is no on-screen answer to what a saved projector image should be
  // linearized with, and the two ways to invent one are both the failure this
  // whole scenario exists against: carrying `compensation`'s last value would
  // send a setting that is nowhere on screen, and giving the generate box its
  // own linearizer would put a second, disagreeing control beside the one the
  // picture is drawn from. Refusing is the only answer that keeps "what is
  // saved is what is displayed" true.
  // 本課題が実装者に委ねた判断を、挙動として固定する。`gamut` に線形化は無く、
  // `views/gamut.py` は「持たせてはならない」ことを長く論じている（色域外か
  // どうかはクリップ前の線形値の性質であり、線形化より前に決まる）。よって
  // 保存する投影画像を何で線形化するかについて、画面上に答えが存在しない。
  // 作り出す方法は2つともこのシナリオが相手にしている失敗そのものである
  // （`compensation` の最後の値を持ち回れば画面に無い設定で保存され、生成欄に
  // 独自の線形化を置けば絵を描いている側と食い違う2つ目のコントロールになる）。
  // A 409 is left standing first, unanswered, so the offer is genuinely on
  // screen when the view changes underneath it -- otherwise the assertion
  // about its button below would be about an element nobody can see.
  // 先に 409 を出したまま（確認せずに）ビューを切り替える。そうしないと、
  // 下の確認ボタンについての主張は「誰にも見えない要素」の話になる。
  await offer409();

  el("view-select").value = "gamut";
  el("view-select").__fire("change");
  await settled();
  const postsBeforeGamut = generatePostLog.length;
  const offerSurvived = el("gen-overwrite").hidden === false;
  const disabledUnderGamut = el("btn-generate").disabled;
  // The 上書き offer from the 409 above is deliberately still on screen -- its
  // sentence is the only written record of why that directory was refused --
  // so its button has to go dead with 生成. A live-looking 上書きして生成 under
  // a view that cannot generate is a control whose only outcome is the refusal
  // already printed beside 生成.
  // 上の 409 の確認は意図して残っている（断られた理由が書かれている唯一の場所
  // である）ため、そのボタンは 生成 と一緒に無効化されなければならない。
  const overwriteDisabledUnderGamut = el("btn-generate-overwrite").disabled;
  const genInfoUnderGamut = el("gen-info").textContent;
  // `__fire`, not a click a user could make: a disabled button dispatches
  // nothing, so this sentence is reachable only from something that calls
  // `runGenerate` regardless -- a driver, or an already-dispatched click. It is
  // the backstop, not the announcement. What a person sees before pressing is
  // the button's own appearance and the line beside it, which are the two
  // checks below.
  // `__fire` は「無効でも届く経路」（自動操作、発行済みのクリック）を表す。
  // 無効なボタンはクリックを配送しないので、この文は控えの砦であって告知では
  // ない。押す前に人が見るのはボタンの見た目と脇の行で、それは下の2件。
  el("btn-generate").__fire("click");
  await sleep(60);
  check("a view that declares no linearizer cannot generate, and names itself",
    disabledUnderGamut === true
      && offerSurvived === true
      && overwriteDisabledUnderGamut === true
      && generatePostLog.length === postsBeforeGamut
      // The parameter it is missing, named **alone**: the fixture declares the
      // same three `views/gamut.py` does, so a refusal that listed `target` as
      // well would be answering about a view nobody ships. Measured: against
      // the old `params: []` fixture the sentence read
      // 「target, linearizer を宣言していないため」 and a looser test of it
      // passed -- which is what made the fixture symmetric in the one dimension
      // this check exists for.
      && statusText().includes("は linearizer を宣言していないため")
      && statusText().includes("色域外判定")
      && el("status").classList.contains("error"),
    `disabled=${disabledUnderGamut} offer on screen=${offerSurvived} `
      + `overwrite disabled=${overwriteDisabledUnderGamut} `
      + `posts=${postsBeforeGamut} -> ${generatePostLog.length} status=${statusText()}`);

  // ...and it has to *look* disabled, which is a stylesheet declaration and so
  // is asserted the way `.readout`'s row is: no CSS engine here, the rule
  // itself is what is pinned.
  //
  // `button` sets a background, `cursor: pointer` and a hover brightening, and
  // none of them stop applying to a disabled button. Without these two rules
  // the 生成 button under `gamut` was full accent blue, showed a pointer, and
  // lit up under the cursor while dispatching nothing -- pixel-identical to the
  // live one two screenshots earlier. Since a disabled button dispatches no
  // click, that leaves a user who presses it with no answer of any kind, and
  // the whole `gamut` decision resting on a state with no visible form.
  // 無効なボタンは無効に見えること。CSS エンジンは無いので宣言そのものを固定
  // する（`.readout` の行と同じ方式）。この2規則が無いと `gamut` の 生成 は
  // 青いまま・ポインタのまま・ホバーで明るくなり、しかもクリックは配送されない
  // ので、押した人には何の答えも返らない。
  const disabledButtonRule = cssDeclarations("button:disabled");
  const disabledHoverRule = cssDeclarations("button:disabled:hover");
  check("a disabled button looks disabled and does not light up under the cursor",
    /opacity:\s*0?\.\d/.test(disabledButtonRule)
      && /cursor:\s*not-allowed/.test(disabledButtonRule)
      && /filter:\s*none/.test(disabledHoverRule)
      && /border-color:/.test(disabledHoverRule),
    `button:disabled { ${disabledButtonRule.trim().replace(/\s+/g, " ")} } `
      + `button:disabled:hover { ${disabledHoverRule.trim().replace(/\s+/g, " ")} }`);

  // --- 8. the line beside the button is about the obstacle in front of it
  //
  // `#gen-info` is the only text a user reads before pressing 生成, and it has
  // three states in this scenario: nothing loaded (the remedy is the path box),
  // `gamut` (the remedy is the view selector) and `compensation` after a 409
  // (there is no obstacle -- the button is live and the line reports what last
  // happened). All three are asserted together because the defect they were
  // written against is a single missing `else`, whose effect is that the second
  // state's sentence is printed in all three.
  //
  // Measured on the shipped code before the fix: 「このビューでは生成できません」
  // at the first paint, when nothing was loaded, and still there on
  // `compensation` beside an *enabled* button -- including in the screenshot
  // captioned "the settings about to be sent". A sentence that says the same
  // thing on every view says nothing on any of them, and it is the only textual
  // signal that `gamut` genuinely cannot generate.
  // 押す前に読まれる唯一の文字列。この場面では状態が3つあり（未読み込み /
  // `gamut` / 409 のあとの `compensation`）、3つまとめて検証する。相手にして
  // いる欠陥は `else` の欠落1つで、その効果は「2番目の文が3つとも出る」こと
  // だからである。
  el("view-select").value = "compensation";
  el("view-select").__fire("change");
  await settled();
  const genInfoBackOnCompensation = el("gen-info").textContent;
  const GAMUT_REFUSAL_LINE = "このビューでは生成できません";
  check("the line beside 生成 names the obstacle, and goes when the obstacle does",
    genInfoAtFirstPaint !== GAMUT_REFUSAL_LINE
      && /読み込/.test(genInfoAtFirstPaint)
      && genInfoUnderGamut === GAMUT_REFUSAL_LINE
      // Back where generating is possible: the button is live again and the
      // line is the last thing that actually happened -- the 409 two steps up.
      && el("btn-generate").disabled === false
      && genInfoBackOnCompensation === "既存のため中止",
    `first paint="${genInfoAtFirstPaint}" gamut="${genInfoUnderGamut}" `
      + `back on compensation="${genInfoBackOnCompensation}" `
      + `disabled=${el("btn-generate").disabled}`);
}

if (!checks.length) check(`unknown scenario ${SCENARIO}`, false);
emit();
process.exit(0);
