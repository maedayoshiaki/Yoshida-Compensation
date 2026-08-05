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
  const params = new URLSearchParams({ path });
  if (session.suffixes.length) params.set("suffixes", session.suffixes.join(","));
  const res = await fetch(`/api/browse?${params.toString()}`);
  if (!res.ok) {
    // Shown inside the dialog and the dialog stays where it was. Putting this
    // on the page's status bar would print it behind the backdrop, and a
    // refusal for a directory the user cannot see is a refusal they cannot act
    // on. モーダル内に出し、直前のディレクトリに留まる。
    let detail = await res.text();
    try {
      const parsed = JSON.parse(detail);
      if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
    } catch (_) { /* not JSON; show it as it arrived */ }
    $("picker-error").textContent = `${res.status}: ${detail}`;
    return;
  }
  const body = await res.json();
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
