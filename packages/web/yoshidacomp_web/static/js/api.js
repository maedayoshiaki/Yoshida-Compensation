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
import { LAYOUT_INTERLEAVED, requireKnownLayout } from "./contract.js";

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
  // "planar" = (C, H, W); "interleaved" = (H, W, C). A *missing* header means
  // interleaved, because that is what a pre-planar endpoint looks like and it
  // still has to render. A header that is present but says something else is
  // refused here, at the one place the wire format is parsed, rather than at
  // the four places that index into `data` -- each of which would otherwise
  // read it as interleaved and agree with the other three about the wrong
  // answer. See contract.js::requireKnownLayout.
  // ヘッダが無い場合はインタリーブ（planar 化前のエンドポイントの見え方であり、
  // それでも描けなければならない）。値が有って未知の場合は、`data` に添字を
  // 立てる4か所ではなくワイヤを解釈するここで断る。4か所に任せるとどこもが
  // インタリーブとして読み、互いに同じ間違いで一致してしまう。
  const layout = res.headers.get("X-Layout") || LAYOUT_INTERLEAVED;
  requireKnownLayout(layout, "サーバの応答");
  return {
    data: new Float32Array(buf),
    height: Number(res.headers.get("X-Height")),
    width: Number(res.headers.get("X-Width")),
    channels: Number(res.headers.get("X-Channels")),
    layout,
    kind: res.headers.get("X-Kind"),
    vmin: Number(res.headers.get("X-Vmin")),
    vmax: Number(res.headers.get("X-Vmax")),
    labels: decodeLabels(res.headers.get("X-Labels-B64")),
  };
}
