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

# Repository root, used only to resolve the default config.toml. No sys.path
# insert: nothing in this file imports ``examples.*``; ``yoshidacomp`` is an
# installed workspace package.
# リポジトリルート。既定の config.toml の解決にのみ使う。``examples.*`` を
# import しないため sys.path への挿入は不要（``yoshidacomp`` はインストール
# 済みのワークスペースパッケージ）。
_REPO_ROOT = Path(__file__).resolve().parents[2]

from yoshidacomp.config import get_config, reload_config, split_cli_config_path
from yoshidacomp.compensation.inverse_lut import scale_shutter_tv, tv_to_seconds


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
    """Confirm each bracket snaps to a parseable standard Canon Tv.

    カメラ無しで、各ブラケット段が ``scale_shutter_tv`` により実在の標準
    Canon Tv へスナップされ、``_parse_tv`` を通ることを確認する。機種固有の
    サポート集合は実機モードで確認する。
    """
    try:
        from edsdk.camera_controller import _parse_tv
    except ImportError as exc:
        print(f"edsdk import failed: {exc}", file=sys.stderr)
        return 2

    base_seconds = tv_to_seconds(base_tv)
    print(f"base tv = {base_tv!r} ({base_seconds:g}s), hdr_max_stops = {max_stops}")
    print("(dry-run: snaps to standard Canon Tv; per-body support needs the camera)\n")
    print(
        f"{'dir':<8} {'scale':>7}  {'target s':>10}  {'snapped tv':>12}  "
        f"{'snapped s':>10}  {'err%':>6}  parse"
    )
    print("-" * 78)

    failures = 0
    for direction, scale in _bracket_scales(max_stops):
        target = base_seconds * scale
        snapped = scale_shutter_tv(base_tv, scale)
        snapped_seconds = tv_to_seconds(snapped)
        err = abs(snapped_seconds - target) / target * 100.0 if target else 0.0
        try:
            _parse_tv(snapped)
            verdict = "OK"
        except ValueError:
            verdict = "FAIL"
            failures += 1
        print(
            f"{direction:<8} {scale:>7.4g}  {target:>10.5g}  {snapped:>12}  "
            f"{snapped_seconds:>10.5g}  {err:>6.1f}  {verdict}"
        )

    print("-" * 78)
    if failures:
        print(
            f"\n{failures} bracket step(s) still fail to parse -- unexpected after "
            "snapping.\nInvestigate scale_shutter_tv / _STANDARD_TV_DISPLAYS."
        )
    else:
        print(
            "\nAll bracket steps snap to a realizable standard Tv (parse OK).\n"
            "'err%' is the discrete-shutter drift; the HDR merge divides by the\n"
            "ACTUAL ratio, so this drift does not bias recovered values.\n"
            "Run without --dry-run on the Canon to confirm this body's own menu."
        )
    return 1 if failures else 0


def _hardware_run(base_tv: str, max_stops: int) -> int:
    """Snap each bracket onto the connected Canon's own Tv menu and check it.

    接続中の Canon の受理 Tv 集合（``list_supported()``）へ各ブラケット段を
    スナップし、実際に受理されるかを確認する。撮影は行わず、プロパティ記述子の
    照会のみ。
    """
    try:
        from edsdk.camera_controller import PropID, TvTable, _parse_tv
        from edsdk.camera_controller import CameraController
    except ImportError as exc:
        print(f"edsdk import failed: {exc}", file=sys.stderr)
        return 2

    try:
        with CameraController() as camera:
            supported_codes = set(camera._get_supported_codes(PropID.Tv))
            supported = list(camera.list_supported().get("Tv", []))
    except Exception as exc:  # noqa: BLE001 - surface any EDSDK/session error
        print(
            f"Could not query the Canon camera: {exc}\n"
            "Is the body connected, powered, and not busy in another app?\n"
            "You can still run the static check with --dry-run.",
            file=sys.stderr,
        )
        return 2

    print(f"Camera-supported Tv values ({len(supported)}):")
    print("  " + ", ".join(str(tv) for tv in sorted(supported, key=str)) + "\n")

    base_seconds = tv_to_seconds(base_tv)
    print(f"base tv = {base_tv!r} ({base_seconds:g}s), hdr_max_stops = {max_stops}\n")
    print(
        f"{'dir':<8} {'scale':>7}  {'target s':>10}  {'snapped tv':>12}  "
        f"{'verdict':<13}  err%"
    )
    print("-" * 80)

    failures = 0
    for direction, scale in _bracket_scales(max_stops):
        target = base_seconds * scale
        snapped = scale_shutter_tv(base_tv, scale, supported=supported)
        snapped_seconds = tv_to_seconds(snapped)
        err = abs(snapped_seconds - target) / target * 100.0 if target else 0.0
        try:
            code = _parse_tv(snapped)
            accepted = code in supported_codes
        except ValueError:
            accepted = False
        verdict = "OK" if accepted else "NOT_SUPPORTED"
        if not accepted:
            failures += 1
        print(
            f"{direction:<8} {scale:>7.4g}  {target:>10.5g}  {snapped:>12}  "
            f"{verdict:<13}  {err:>5.1f}"
        )

    print("-" * 80)
    if failures:
        print(
            f"\n{failures} bracket step(s) are still not accepted by this body.\n"
            "Unexpected: scale_shutter_tv was given this body's own menu. Check that\n"
            "list_supported()['Tv'] and _parse_tv agree on display formatting."
        )
    else:
        print(
            "\nAll bracket steps snap onto this body's Tv menu and are accepted.\n"
            "'err%' is the discrete-shutter drift; the HDR merge divides by the\n"
            "ACTUAL exposure ratio, so it does not bias recovered values."
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
