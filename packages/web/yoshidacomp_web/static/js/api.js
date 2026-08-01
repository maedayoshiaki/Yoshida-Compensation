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
import { requireKnownLayout } from "./contract.js";

function decodeLabels(header) {
  if (!header) return [];
  const binary = atob(header);
  const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
  return JSON.parse(new TextDecoder("utf-8").decode(bytes));
}

// Turn a failed response into the sentence the status bar should show.
//
// FastAPI puts its message in a JSON `{"detail": …}` body, so that is what a
// deliberate refusal looks like -- a 400 from `routers/view.py`, a 404 from
// `routers/session.py`. Anything else (a bare 500's plain text, a proxy's HTML
// error page) has no `detail` to find and is shown as it arrived.
//
// Both callers use this. `fetchView` used to `throw new Error(await
// res.text())` instead, which put the raw body on screen: the camera-space
// refusal rendered as the literal string `{"detail":"view \"surface\" needs a
// projector-space matrix; …"}`, backslashes and braces included, in a status
// bar that is otherwise Japanese prose.
//
// 失敗応答から、ステータス行に出すべき文を作る。FastAPI は JSON の
// `{"detail": …}` で理由を返すので、意図的な拒否はその形になる。それ以外
// （素の500のテキスト、プロキシのHTML）は `detail` を持たないのでそのまま出す。
// 以前 `fetchView` は本文をそのまま投げており、カメラ座標の拒否が
// `{"detail":"…"}` という生の文字列のまま画面に出ていた。
async function errorMessage(res) {
  const body = await res.text();
  try {
    const parsed = JSON.parse(body);
    if (parsed && typeof parsed.detail === "string") {
      return `${res.status}: ${parsed.detail}`;
    }
  } catch (_) {
    /* body was not JSON; fall through to showing it as-is */
  }
  return body ? `${res.status}: ${body}` : `${res.status}: ${res.statusText}`;
}

async function jsonOrThrow(res) {
  if (!res.ok) throw new Error(await errorMessage(res));
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

// One pixel's 4x3 coefficients. The matrix itself never crosses the wire --
// see routers/session.py::probe_matrix for why this is a point query.
//
// Non-finite coefficients arrive as `null`, deliberately -- see
// routers/session.py::_finite_or_none. That is a contract for matrices that
// carry NaN, not a description of the ones loaded here: the stored array is
// finite everywhere, including outside the active projection region (measured
// on data/linear_pipeline_1080/matrices_linear_gamma.npy: zero non-finite
// values against 74.86% of pixels outside the region). The NaN on screen is
// written by the views at render time, which is why the readout has its own
// out-of-region marker rather than letting a `null` stand for it.
//
// 1画素分の4x3係数だけを取る。行列本体は線に乗せない。非有限係数は `null` で
// 届くが、それはNaNを持ちうる行列のための契約であって、ここで読んでいる行列の
// 説明ではない。保存された配列は有効投影領域の外も含めて全て有限である
// （実測: 非有限値0個、領域外は全画素の74.86%）。画面上のNaNはビューが描画時に
// 書いているものであり、読み取り欄が独自の「領域外」の印を持つ理由がこれである。
export async function probeMatrix(matrixId, row, col) {
  const res = await fetch(
    `/api/matrix/${encodeURIComponent(matrixId)}/probe?row=${row}&col=${col}`,
  );
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function fetchView(matrixId, key, params) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`/api/matrix/${matrixId}/view/${key}?${qs}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  // "planar" = (C, H, W); "interleaved" = (H, W, C). The header is required;
  // an absent one is refused exactly like an unrecognised one.
  //
  // It used to default to "interleaved" when absent, on the reasoning that an
  // endpoint which had not been converted should still render. Nothing in this
  // repository is such an endpoint -- routers/view.py sends X-Layout on every
  // response, unconditionally -- so the default protected nothing and cost the
  // thing this guard exists for. Header *stripping* is at least as ordinary as
  // header mangling, and a stripped header landed in the silent class: a planar
  // body read as interleaved, with the tiles, the hover readout and the
  // per-tile ranges all corroborating the same wrong answer.
  //
  // The interleaved *decode* path is untouched and still exercised -- see
  // `legacy_tiles` in tests/js/frontend_harness.mjs, which now declares the
  // layout it is actually using. What is gone is inferring a layout from
  // silence.
  //
  // Checked before the body is read, mirroring routers/view.py: buffering a
  // camera-space payload is 1.5 GB, and there is no point paying that for a
  // response that is about to be rejected.
  //
  // X-Layout は必須。無い場合も未知の値と同じく断る。以前はヘッダが無ければ
  // インタリーブとみなしていたが、このリポジトリに未変換のエンドポイントは
  // 無く（routers/view.py は常に付ける）、既定値は何も守らずガードの目的
  // だけを損ねていた。ヘッダの欠落は書き換えと同程度にありふれており、
  // 欠落は静かな側に落ちる——planar の本体をインタリーブとして読み、タイル・
  // 読み取り値・レンジがそろって同じ誤りで一致する。インタリーブの復号経路
  // 自体は残っており、`legacy_tiles` が今度は明示的に申告して通す。判定は
  // 本体を読む前に行う（カメラ座標では 1.5 GB を捨てるために確保することになる）。
  const layout = res.headers.get("X-Layout");
  requireKnownLayout(layout, "サーバの応答");
  const buf = await res.arrayBuffer();
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
