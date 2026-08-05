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
    let detail = await res.text();
    if (session !== mine) return;   // the dialog moved on while this was in flight
    try {
      const parsed = JSON.parse(detail);
      if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
    } catch (_) { /* not JSON; show it as it arrived */ }
    $("picker-error").textContent = `${res.status}: ${detail}`;
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
  const settle = session.settle;
  session = null;
  backdrop().hidden = true;
  settle(result);
}

// Wired once, at module load, because the elements are in index.html and the
// dialog is reused. 要素は index.html にあり使い回すので結線は1度きり。
function wireOnce() {
  $("picker-cancel").addEventListener("click", () => close(null));
  $("picker-choose").addEventListener("click", () => {
    const chosen = chosenPath();
    if (chosen !== "") close(chosen);
  });
  $("picker-up").addEventListener("click", () => {
    if (session && session.parent !== null) navigate(session.parent);
  });
  $("picker-roots").addEventListener("change", (e) => { navigate(e.target.value); });
  $("picker-newdir-name").addEventListener("input", () => { renderFoot(); });
  // A click on the backdrop is a cancel; a click inside the dialog is not.
  // 背景のクリックは取消。ダイアログ内は違う。
  backdrop().addEventListener("click", (e) => {
    if (session && e.target === backdrop()) close(null);
  });
}

let wired = false;

export async function pickPath({ mode, suffixes = [], start = "", title = "" }) {
  if (!wired) { wireOnce(); wired = true; }
  return new Promise((settle) => {
    session = { mode, suffixes, cwd: "", parent: null, selected: null, settle };
    $("picker-title").textContent = title;
    $("picker-error").textContent = "";
    $("picker-newdir-name").value = "";
    $("picker-newdir").hidden = mode !== "dir";
    backdrop().hidden = false;
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
