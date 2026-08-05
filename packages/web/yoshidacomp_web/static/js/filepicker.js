// The file picker behind every 参照 button.
//
// `<input type="file">` cannot be used here: it hands back a name and the
// bytes, never the path, and this backend takes a path and reads the disk
// itself. So the listing is the server's (`GET /api/browse`) and this module
// only draws it.
//
// It fills a text box and nothing else. It does not load, it does not
// generate, and it never creates, deletes or renames anything on disk -- there
// is no endpoint for any of that. In `dir` mode the returned path may not
// exist yet; making it is `POST /api/generate`'s job, as it always was.
//
// `<input type="file">` は名前と中身しか返さずパスを返さないので使えない
// （バックエンドはパスを受け取って自分で読む）。一覧はサーバのもので、この
// モジュールは描くだけである。テキスト欄を埋める以外のことはしない。

// The sentence a refused response turns into is `api.js`'s, not this module's.
// It was copied here once, and the copy had already drifted: it lost the
// empty-body fallback, so a response that arrives with no body rendered as
// `404: ` -- a status, a colon and nothing -- where `errorMessage` says
// `404: Not Found`. 拒否応答を文にするのは api.js の仕事である（複製は既に
// 空本文の分岐を落としていた）。
import { errorMessage } from "./api.js";

const backdrop = () => document.getElementById("picker-backdrop");
const $ = (id) => document.getElementById(id);

// One dialog, so one live session at a time. Held here rather than passed
// around because the DOM listeners are wired once and have to reach whichever
// call is open. 開いているのは常に1つなのでここに持つ。
let session = null;

// Suffix filters come from the server (`GET /api/browse/suffixes`) so that the
// set of image extensions has one home. Fetched once and kept.
// 拡張子はサーバから取る（出所を1つにするため）。1度だけ取得して保持する。
let suffixTable = null;

export async function suffixesFor(key) {
  if (suffixTable === null) {
    const res = await fetch("/api/browse/suffixes");
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    suffixTable = await res.json();
  }
  return suffixTable[key] ?? [];
}

function humanSize(bytes) {
  if (bytes === null || bytes === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes, i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${i === 0 ? value : value.toFixed(1)} ${units[i]}`;
}

// The string 選択 would hand back right now, and the string `#picker-chosen`
// shows. The three modes differ only here.
//
// In `dir` mode the new-folder box is appended to the current directory, so an
// empty box means "this directory" -- the ordinary case -- and a filled one
// names a directory that does not exist yet. That is the whole of the
// new-folder affordance; nothing is created here.
// 3つのモードの違いはこの関数だけに閉じている。`dir` では新規フォルダ名を
// 現在地に連結する（空なら現在地そのもの）。作成はしない。
function chosenPath() {
  if (!session) return "";
  // No directory has landed: the opening `navigate()` is still in flight, or
  // it came back 404/400/403 or never reached the server at all. There is
  // nothing to build a path *from* -- `${session.cwd}/${name}` on an empty
  // `cwd` is `/name`, the root of the current drive, and in `dir` mode that
  // string is what 選択 would hand `gen-output-dir` and what `POST
  // /api/generate` would then write into. 「まだどこにも着いていない」状態で
  // 連結すると `/入力名`（＝現在のドライブの直下）ができる。出力先の欄では
  // それがそのまま書き込み先になる。
  if (session.cwd === "") return "";
  if (session.mode === "dir") {
    const name = $("picker-newdir-name").value.trim();
    return name ? `${session.cwd}/${name}` : session.cwd;
  }
  if (session.selected) return `${session.cwd}/${session.selected}`;
  return session.mode === "file_or_dir" ? session.cwd : "";
}

function renderFoot() {
  const chosen = chosenPath();
  $("picker-chosen").textContent = chosen;
  // `file` is the only mode with nothing to fall back on: a directory is not
  // an answer there. 「ファイルのみ」だけは既定の答えを持たない。
  $("picker-choose").disabled = chosen === "";
}

function renderList(body) {
  const host = $("picker-list");
  host.textContent = "";        // replaces the children, unregistering their ids
  body.entries.forEach((entry, i) => {
    const row = document.createElement("div");
    row.id = `picker-entry-${i}`;
    row.className = "picker-row";
    const name = document.createElement("span");
    name.textContent = entry.is_dir ? `${entry.name}/` : entry.name;
    row.appendChild(name);
    const size = document.createElement("span");
    size.className = "size";
    size.textContent = humanSize(entry.size);
    row.appendChild(size);
    row.addEventListener("click", () => {
      if (entry.is_dir) { navigate(`${body.path}/${entry.name}`); return; }
      session.selected = entry.name;
      for (const sibling of host.children) sibling.classList.remove("selected");
      row.classList.add("selected");
      renderFoot();
    });
    host.appendChild(row);
  });
  // The count is stated whenever it is not the whole story. A silent cut reads
  // as "this is everything in the directory".
  // 切り詰めたことは必ず言う。黙って切ると「これが全部」と読まれる。
  $("picker-count").textContent = body.truncated
    ? `${body.total}件中${body.entries.length}件を表示`
    : `${body.total}件`;
}

function renderRoots(roots, cwd) {
  const select = $("picker-roots");
  select.textContent = "";
  for (const root of roots) {
    const option = document.createElement("option");
    option.value = root;
    option.textContent = root;
    select.appendChild(option);
  }
  const match = roots.find((r) => cwd.toLowerCase().startsWith(r.toLowerCase()));
  if (match) select.value = match;
}

async function navigate(path) {
  // `navigate` is called unawaited from four places -- a directory row
  // (see ::renderList below), `↑ 上へ`, the root <select>, and `pickPath`'s
  // own opening call -- none of which wait for it, and `session` is one
  // module-level binding every dialog shares. A response that lands after
  // the dialog has moved on writes into whatever `session` currently names,
  // not into the dialog it was issued for.
  //
  // After 取消/選択 (`close()` sets `session = null`) that throws --
  // `session.cwd = ...` on `null`, reproduced against the shipped code with
  // no mutation involved: click a directory row, then 取消 before the
  // listing arrives. After a second 参照 opens a *different* box
  // (`pickPath()` replaces `session` with a new object) it does not throw,
  // and is worse for it: the late response silently repaints that other
  // dialog's `picker-cwd` and listing with its own answer -- the ターゲット
  // dialog can end up showing the 行列 box's `.npy`-filtered listing of a
  // directory nobody asked it for, and 選択 then returns a path built from
  // someone else's answer.
  //
  // `mine` freezes which session this call belongs to at the moment it
  // started; every write below that follows an `await` is gated on
  // `session` still being that same object.
  // navigate() は一覧のディレクトリ行・↑ 上へ・ルート <select>・pickPath()
  // 自身の開始呼び出しのいずれからも await されずに呼ばれ、session はダイア
  // ログ全体で共有する単一のモジュール変数である。ダイアログが先に進んだ
  // あとに届いた応答は、その時点で session が指しているものへ書き込む
  // ——取消・選択のあと（session が null）なら例外（出荷コードそのものに
  // 対して、行クリック→取消→到着で再現する）。別の参照が別の欄を開いた
  // あと（session が別オブジェクトに差し替わる）なら例外にはならず、その
  // 欄の一覧を黙って上書きする——ターゲットのダイアログが行列欄の .npy
  // 絞り込み一覧を出し、選択がそこから組んだパスを返しうる。mine はこの
  // 呼び出しがどの session に属するかを開始時点で固定する。await の後に
  // 来る書き込みはすべて、session がまだ同じオブジェクトであることを
  // 条件にする。
  const mine = session;
  const params = new URLSearchParams({ path });
  if (session.suffixes.length) params.set("suffixes", session.suffixes.join(","));
  let res;
  try {
    res = await fetch(`/api/browse?${params.toString()}`);
  } catch (e) {
    // A transport-level failure (server down, connection dropped) is not an
    // HTTP refusal -- `fetch` rejects instead of resolving -- and `navigate`
    // is called from three unawaited sites (a directory row, `↑ 上へ`, the
    // root <select>) plus `pickPath`'s own initial call, none of which had a
    // `.catch`. Left uncaught, the rejection was unhandled *and*
    // `pickPath`'s own Promise -- whose `settle` only close() calls, and
    // close() is never reached here -- stayed pending forever, so the
    // backdrop never closed and 参照 stopped working for the rest of the
    // page's life. Handled the same way as an HTTP refusal: shown inside the
    // dialog, which stays open where it was.
    // 輸送層の失敗（サーバ停止・断線）は HTTP の拒否と違い fetch 自体が
    // reject する。navigate() は3つの結線＋pickPath 自身の初回呼び出しから
    // await も catch も無いまま呼ばれており、ここを捕らないと pickPath の
    // Promise（settle を呼ぶのは close() だけで、ここには届かない）が
    // 永久に解決せず、以後ページを再読み込みするまで 参照 が効かなくなる。
    // HTTP の拒否と同じ扱いにする（モーダル内に出し、開いたまま留まる）。
    if (session !== mine) return;   // the dialog moved on while this was in flight
    $("picker-error").textContent = `${e.message || e}`;
    return;
  }
  if (!res.ok) {
    // Shown inside the dialog and the dialog stays where it was. Putting this
    // on the page's status bar would print it behind the backdrop, and a
    // refusal for a directory the user cannot see is a refusal they cannot act
    // on. モーダル内に出し、直前のディレクトリに留まる。
    const message = await errorMessage(res);
    if (session !== mine) return;   // the dialog moved on while this was in flight
    $("picker-error").textContent = message;
    return;
  }
  const body = await res.json();
  if (session !== mine) return;   // the dialog moved on while this was in flight
  $("picker-error").textContent = "";
  session.cwd = body.path;
  session.parent = body.parent;
  session.selected = null;      // a listing the selection does not belong to
  $("picker-cwd").textContent = body.path;
  $("picker-up").disabled = body.parent === null;
  renderRoots(body.roots, body.path);
  renderList(body);
  renderFoot();
}

function close(result) {
  // Guarded like `picker-up` and the backdrop handler are. Unreachable in a
  // browser today -- `.picker-backdrop[hidden] { display: none }` takes the
  // dialog out of the layout, so nothing inside it can be clicked once
  // `session` is null -- but the two callers below used to be the only
  // listeners dereferencing `session` without asking, and that asymmetry reads
  // as if someone had judged them safe rather than not having reached them.
  // 他の結線と同じく `session` を確かめる。今日のCSSでは到達しないが、
  // 確かめていない結線が2つだけ在ると「安全だと判断した」ように読める。
  if (!session) return;
  const { settle, returnFocusTo } = session;
  session = null;
  backdrop().hidden = true;
  // Focus goes back to the 参照 button that opened the dialog (spec §6).
  // Without it the page's focus is wherever it was before the dialog took it,
  // which is behind a backdrop that has just been removed.
  // フォーカスは開いた 参照 ボタンへ戻す。
  if (returnFocusTo) returnFocusTo.focus();
  settle(result);
}

// Wired once, at module load, because the elements are in index.html and the
// dialog is reused. 要素は index.html にあり使い回すので結線は1度きり。
function wireOnce() {
  $("picker-cancel").addEventListener("click", () => { if (session) close(null); });
  $("picker-choose").addEventListener("click", () => {
    const chosen = chosenPath();
    if (chosen !== "") close(chosen);
  });
  $("picker-up").addEventListener("click", () => {
    if (session && session.parent !== null) navigate(session.parent);
  });
  $("picker-roots").addEventListener("change", (e) => { if (session) navigate(e.target.value); });
  $("picker-newdir-name").addEventListener("input", () => { renderFoot(); });
  // A click on the backdrop is a cancel; a click inside the dialog is not.
  // 背景のクリックは取消。ダイアログ内は違う。
  backdrop().addEventListener("click", (e) => {
    if (session && e.target === backdrop()) close(null);
  });
  // `Esc` is a cancel, the same as the backdrop (spec §6). It is listened for
  // here, on the backdrop, so it has to bubble from something inside it --
  // which is why `pickPath` moves focus into the dialog rather than leaving it
  // on the 参照 button behind it. That placement is not only about the key
  // arriving: focus left on 参照 makes Space or Enter press it a second time,
  // and a second 参照 replaces `session` without ever calling `close()`, so
  // the first call's Promise is orphaned and the dialog it belonged to is
  // answered into someone else's (see `navigate`'s note on `mine`).
  // Esc は取消。背景で拾うので、ダイアログ内にフォーカスが在ることが前提に
  // なる（pickPath で移す）。背後の 参照 にフォーカスが残っていると
  // Space/Enter で2度目の 参照 が走り、1つ目の Promise が宙に浮く。
  backdrop().addEventListener("keydown", (e) => {
    if (session && e.key === "Escape") close(null);
  });
}

let wired = false;

export async function pickPath({
  mode, suffixes = [], start = "", title = "", returnFocusTo = null,
}) {
  if (!wired) { wireOnce(); wired = true; }
  return new Promise((settle) => {
    session = { mode, suffixes, cwd: "", parent: null, selected: null, settle, returnFocusTo };
    $("picker-title").textContent = title;
    $("picker-error").textContent = "";
    $("picker-newdir-name").value = "";
    $("picker-newdir").hidden = mode !== "dir";
    // Nothing of the dialog this one replaces is left on screen.
    //
    // The listing, the path line and the count belong to the *previous*
    // dialog until this one's own `navigate()` lands -- and it may never land
    // (404 for a directory that has been renamed or whose drive is not
    // mounted, 400, 403, or a server that is not running). Left in place, the
    // dialog names a directory it is not in, over that directory's rows,
    // while `session.cwd` is `""`. `chosenPath()` refuses to build a path out
    // of that (see there), but a header and a listing inviting a choice about
    // someone else's directory are wrong on their own.
    //
    // 前のダイアログの一覧・現在地・件数を残さない。開始の navigate() が
    // 404/400/403 や輸送層の失敗で終わった場合、それらは「今どこに居るか」
    // ではなく「さっきどこに居たか」を映したままになる。
    $("picker-list").textContent = "";
    $("picker-cwd").textContent = "";
    $("picker-count").textContent = "";
    renderFoot();
    backdrop().hidden = false;
    // Into the dialog, and back to the 参照 button in `close()`. See the
    // `keydown` wiring for why this is where the keyboard contract starts.
    // フォーカスをダイアログへ移す（戻すのは close()）。
    $("picker").focus();
    // A box holding a file's path should open in that file's directory, not at
    // the working directory -- the server answers 400 for a file, so the
    // directory part is what to ask for. Empty stays empty and the server picks.
    // ファイルのパスが入っている欄は、そのファイルのあるディレクトリで開く。
    const at = start.includes("/") || start.includes("\\")
      ? start.replace(/\\/g, "/").replace(/\/[^/]*$/, "")
      : start;
    navigate(at);
  });
}
