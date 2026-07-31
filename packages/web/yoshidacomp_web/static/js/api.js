// Thin fetch layer over the FastAPI backend.
//
// View payloads come back as raw float32 binary (never JSON) with the shape and
// display range in X-* headers. The frontend is served same-origin, so those
// headers are readable without CORS exposure rules.
//
// ビューの中身は float32 のバイナリで返る。形状と表示レンジは X-* ヘッダ。
// 同一オリジン配信なので CORS の露出設定なしにヘッダを読める。

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
    kind: res.headers.get("X-Kind"),
    vmin: Number(res.headers.get("X-Vmin")),
    vmax: Number(res.headers.get("X-Vmax")),
    labels: (res.headers.get("X-Labels") || "").split(","),
  };
}
