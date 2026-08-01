// Thin fetch layer over the FastAPI backend.
//
// View payloads come back as raw float32 binary (never JSON) with the shape and
// display range in X-* headers. The frontend is served same-origin, so those
// headers are readable without CORS exposure rules.
//
// ビューの中身は float32 のバイナリで返る。形状と表示レンジは X-* ヘッダ。
// 同一オリジン配信なので CORS の露出設定なしにヘッダを読める。

// Channel labels arrive as base64 of a UTF-8 JSON array (X-Labels-B64).
//
// They cannot be a plain header value: Starlette encodes headers as latin-1, so
// a Japanese label makes the whole response a 500. They cannot be comma-joined
// either: a label containing a comma would split into two, producing more
// labels than channels and shifting every tile caption after it. Decoding here
// mirrors routers/view.py::encode_labels.
//
// ラベルは UTF-8 JSON の base64 で届く。ヘッダ値はlatin-1なので日本語ラベルは
// 500になり、カンマ連結はラベル内のカンマで壊れる。routers/view.py の
// encode_labels と対になっている。
function decodeLabels(header) {
  if (!header) return [];
  const binary = atob(header);
  const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
  return JSON.parse(new TextDecoder("utf-8").decode(bytes));
}

async function jsonOrThrow(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch (_) {
      /* body was not JSON; keep the status text */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

// POST a filesystem path; the server loads the .npy and hands back an id + meta.
export async function loadMatrix(path) {
  return jsonOrThrow(
    await fetch("/api/session/matrix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    })
  );
}

export async function fetchMeta(matrixId) {
  return jsonOrThrow(await fetch(`/api/matrix/${matrixId}/meta`));
}

// The view registry drives the GUI: every registered view and its parameters.
export async function listViews() {
  return (await jsonOrThrow(await fetch("/api/views"))).views;
}

export async function fetchView(matrixId, key, params) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`/api/matrix/${matrixId}/view/${key}?${qs}`);
  if (!res.ok) throw new Error(await res.text());
  const buf = await res.arrayBuffer();
  return {
    data: new Float32Array(buf),
    height: Number(res.headers.get("X-Height")),
    width: Number(res.headers.get("X-Width")),
    channels: Number(res.headers.get("X-Channels")),
    // "planar" = (C, H, W); anything else (or a missing header) is the old
    // interleaved (H, W, C). Every consumer that indexes into `data` has to
    // branch on this, so it travels with the payload rather than being assumed.
    // "planar" は (C, H, W)。ヘッダが無ければ従来の (H, W, C) とみなす。
    layout: res.headers.get("X-Layout") || "interleaved",
    kind: res.headers.get("X-Kind"),
    vmin: Number(res.headers.get("X-Vmin")),
    vmax: Number(res.headers.get("X-Vmax")),
    labels: decodeLabels(res.headers.get("X-Labels-B64")),
  };
}
