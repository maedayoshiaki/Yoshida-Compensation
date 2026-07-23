"""Verify Canon shutter (``tv``) switching for the simple-HDR calibration.

Real-hardware check for the one outstanding inverse-LUT TODO: the HDR
bracket in ``inverse_lut.run_calibration_capture`` rebuilds the Canon
backend with ``scale_shutter_tv(cfg.camera.tv, scale)``. Canon only
accepts a discrete set of Tv values per body, so a scaled string may be
unparseable or unsupported, and the bracket loop has no error handling
(a failure aborts the whole calibration). This tool reports, per bracket
step, whether the requested ``tv`` would be accepted and how far the
nearest realizable exposure is from the nominal one -- WITHOUT projecting
anything or releasing the shutter.

簡易HDRキャリブレーションの Canon シャッター（``tv``）切替の実機検証。

逆LUTで唯一残った実機TODOの確認用。HDRブラケットは
``scale_shutter_tv(cfg.camera.tv, scale)`` で Canon バックエンドを
作り直すが、Canon は機種ごとに離散的な Tv 値しか受理しないため、
スケール後の文字列が解釈不能／非対応になり得る。ブラケットループには
例外処理が無く、失敗するとキャリブレーション全体が中断する。本ツールは
投影もシャッターも行わずに、各ブラケット段の ``tv`` が受理されるか、
最も近い実現可能露光が公称値からどれだけずれるかを報告する。

Usage:
    # カメラ不要。テーブル整合だけを静的に確認（今すぐ実行可）
    uv run python examples/python/verify_hdr_tv.py --dry-run [--config config.toml]

    # 実機（Canon 接続時）。list_supported() で機種の実受理値と突き合わせる
    uv run python examples/python/verify_hdr_tv.py [--config config.toml]
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.config import get_config, reload_config, split_cli_config_path
from src.python.inverse_lut import scale_shutter_tv


def _requested_seconds(tv: str) -> float:
    """Nominal exposure seconds encoded by a ``scale_shutter_tv`` string.

    ``scale_shutter_tv`` が返す文字列（``"1/N"`` か小数）を秒に直す。
    """
    text = str(tv).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _bracket_scales(max_stops: int) -> list[tuple[str, float]]:
    """Exposure scales the HDR bracket actually requests, in order.

    HDRブラケットが実際に要求する露光スケールを順に列挙する。
    ``capture_hdr_linear``: 短露光側 0.5, 0.25, ... と長露光側 2, 4, ...。
    """
    scales: list[tuple[str, float]] = [("base", 1.0)]
    scale = 1.0
    for _ in range(max_stops):
        scale *= 0.5
        scales.append(("shorter", scale))
    scale = 1.0
    for _ in range(max_stops):
        scale *= 2.0
        scales.append(("longer", scale))
    return scales


def _dry_run(base_tv: str, max_stops: int) -> int:
    """Check scaled tv strings against edsdk's global parse table only.

    カメラ無しで、スケール後 tv 文字列が edsdk のグローバル Tv テーブルに
    存在するか（＝``_parse_tv`` が通るか）だけを確認する。機種固有の
    サポート集合は分からないので、それは実機モードで確認する。
    """
    try:
        from edsdk.camera_controller import TvTable, _parse_tv, _tv_display_to_seconds
    except ImportError as exc:
        print(f"edsdk import failed: {exc}", file=sys.stderr)
        return 2

    # Precompute standard Canon seconds for nearest-value hints.
    standard: list[tuple[float, str]] = []
    for disp in TvTable.values():
        try:
            standard.append((_tv_display_to_seconds(disp), disp))
        except Exception:
            continue

    def nearest(seconds: float) -> str:
        best = min(standard, key=lambda item: abs(item[0] - seconds))
        return f"{best[1]} ({best[0]:g}s)"

    print(f"base tv = {base_tv!r}, hdr_max_stops = {max_stops}")
    print("(dry-run: global Canon Tv table only; per-body support needs the camera)\n")
    print(f"{'dir':<8} {'scale':>7}  {'requested tv':>14}  {'parse':<6}  hint / nearest standard")
    print("-" * 78)

    failures = 0
    for direction, scale in _bracket_scales(max_stops):
        requested = scale_shutter_tv(base_tv, scale)
        seconds = _requested_seconds(requested)
        try:
            _parse_tv(requested)
            verdict = "OK"
            hint = ""
        except ValueError:
            verdict = "FAIL"
            failures += 1
            hint = f"-> nearest: {nearest(seconds)}"
        print(
            f"{direction:<8} {scale:>7.4g}  {requested:>14}  {verdict:<6}  {hint}"
        )

    print("-" * 78)
    if failures:
        print(
            f"\n{failures} bracket step(s) produce a tv string edsdk cannot parse.\n"
            "These would raise ValueError inside the HDR loop and abort calibration.\n"
            "Fix: snap scale_shutter_tv to a supported Tv (nearest by seconds) and\n"
            "merge using the ACTUAL achieved exposure ratio, not the nominal scale."
        )
    else:
        print("\nAll bracket tv strings parse against the global table.")
        print("Run without --dry-run on the Canon to confirm per-body support.")
    return 1 if failures else 0


def _hardware_run(base_tv: str, max_stops: int) -> int:
    """Query the connected Canon's supported Tv set and check each step.

    接続中の Canon の受理 Tv 集合（``list_supported()``）と各ブラケット段を
    突き合わせる。撮影は行わず、プロパティ記述子の照会のみ。
    """
    try:
        from edsdk.camera_controller import (
            PropID,
            TvTable,
            _parse_tv,
            _tv_display_to_seconds,
        )
        from edsdk.camera_controller import CameraController
    except ImportError as exc:
        print(f"edsdk import failed: {exc}", file=sys.stderr)
        return 2

    try:
        with CameraController() as camera:
            supported_codes = set(camera._get_supported_codes(PropID.Tv))
            supported_displays = [
                TvTable.get(code, str(code)) for code in supported_codes
            ]
    except Exception as exc:  # noqa: BLE001 - surface any EDSDK/session error
        print(
            f"Could not query the Canon camera: {exc}\n"
            "Is the body connected, powered, and not busy in another app?\n"
            "You can still run the static check with --dry-run.",
            file=sys.stderr,
        )
        return 2

    supported_seconds = []
    for disp in supported_displays:
        try:
            supported_seconds.append((_tv_display_to_seconds(disp), disp))
        except Exception:
            continue
    supported_sorted = sorted(
        (d for d in supported_displays), key=str
    )
    print(f"Camera-supported Tv values ({len(supported_displays)}):")
    print("  " + ", ".join(supported_sorted) + "\n")

    def nearest(seconds: float) -> str:
        if not supported_seconds:
            return "?"
        best = min(supported_seconds, key=lambda item: abs(item[0] - seconds))
        return f"{best[1]} ({best[0]:g}s)"

    print(f"base tv = {base_tv!r}, hdr_max_stops = {max_stops}\n")
    print(f"{'dir':<8} {'scale':>7}  {'requested tv':>14}  {'verdict':<13}  detail")
    print("-" * 84)

    failures = 0
    for direction, scale in _bracket_scales(max_stops):
        requested = scale_shutter_tv(base_tv, scale)
        nominal = _requested_seconds(requested)
        try:
            code = _parse_tv(requested)
        except ValueError:
            failures += 1
            print(
                f"{direction:<8} {scale:>7.4g}  {requested:>14}  "
                f"{'PARSE_FAIL':<13}  nearest supported: {nearest(nominal)}"
            )
            continue
        if code not in supported_codes:
            failures += 1
            print(
                f"{direction:<8} {scale:>7.4g}  {requested:>14}  "
                f"{'NOT_SUPPORTED':<13}  nearest supported: {nearest(nominal)}"
            )
            continue
        actual = _tv_display_to_seconds(TvTable.get(code, ""))
        ratio = actual / nominal if nominal else float("nan")
        err = abs(ratio - scale) / scale * 100.0 if scale else 0.0
        print(
            f"{direction:<8} {scale:>7.4g}  {requested:>14}  "
            f"{'OK':<13}  actual={actual:g}s nominal={nominal:g}s "
            f"exposure_err={err:.1f}%"
        )

    print("-" * 84)
    if failures:
        print(
            f"\n{failures} bracket step(s) are rejected by this body -> "
            "calibration would abort.\n"
            "Fix: snap scale_shutter_tv to the nearest supported Tv and merge with\n"
            "the actual exposure ratio (see 'exposure_err' for the discrete drift)."
        )
    else:
        print(
            "\nAll bracket steps are accepted. Note any nonzero exposure_err: the "
            "HDR merge\ndivides by the NOMINAL scale, so large drift biases the "
            "recovered values."
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Run the tv-switching verification (dry-run or against the camera).

    tv 切替検証を実行する（dry-run または実機）。
    """
    if argv is None:
        argv = sys.argv

    clean_argv, config_path = split_cli_config_path(argv)
    dry_run = "--dry-run" in clean_argv
    clean_argv = [a for a in clean_argv if a != "--dry-run"]
    if len(clean_argv) != 1:
        print(
            "Usage: python examples/python/verify_hdr_tv.py "
            "[--dry-run] [--config <config.toml>]"
        )
        return 2

    default_config_path = _REPO_ROOT / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    cfg = get_config()
    base_tv = cfg.camera.tv
    max_stops = cfg.inverse_lut.hdr_max_stops

    if dry_run:
        return _dry_run(base_tv, max_stops)
    return _hardware_run(base_tv, max_stops)


if __name__ == "__main__":
    raise SystemExit(main())
