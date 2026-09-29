"""Hardware-independent tests for examples/python/measure_color_grid_response.py.

計測計画（パッチセット・番兵挿入・再開フィルタ・LUT駆動色）の純粋部分だけを
検査する。実機撮影ループは対象外。

各テストが「どの誤実装なら落ちるか」を docstring に書く（AGENTS.md 第8章）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from examples.python import measure_color_grid_response as mcgr
from yoshidacomp.compensation.inverse_lut import InverseLut


# ── パッチセット ──────────────────────────────────────────────────────


def test_raw_patch_set_is_exactly_321_unique_colors():
    """グリッド216 + ランプ新規60 + 対角新規45 = 321。

    落ちる誤実装: グリッドレベルの取り違え（51刻み以外）、端点の重複排除漏れ
    （321超）、ランプ段数の誤り（round(i*255/16) 以外）。
    """
    patches = mcgr.build_raw_patch_set()
    assert len(patches) == 321
    assert len(set(patches)) == 321


def test_raw_patch_set_contains_expected_members():
    """代表点の実在検査。

    落ちる誤実装: チャンネル順の取り違え（対角線 (v,v,0) が (v,0,v) になる等）、
    ランプ8bit値の丸め方式の違い（16 ではなく 15 を生む切り捨て等）。
    """
    patches = set(mcgr.build_raw_patch_set())
    # グリッドの角と内部点
    assert (0, 0, 0) in patches and (255, 255, 255) in patches
    assert (51, 102, 204) in patches
    # 17段ランプの2段目は round(255/16) = 16
    assert (16, 0, 0) in patches
    assert (0, 16, 0) in patches
    assert (0, 0, 16) in patches
    assert (16, 16, 16) in patches
    # 2チャンネル対角線（RG / RB / GB）
    assert (16, 16, 0) in patches
    assert (16, 0, 16) in patches
    assert (0, 16, 16) in patches
    # どのセットにも属さない色は存在しない（グリッド外レベル×非対角の組）
    assert (143, 16, 0) not in patches
    assert (100, 100, 100) not in patches  # 100 はグリッドにもランプにも無い


def test_lut_check_set_is_35_colors():
    """グレー17 + 単ch4×3 + 混合6 = 35。

    落ちる誤実装: 単chレベルにグリッド端点 255 を混ぜて重複させる、混合色の
    定義漏れ。
    """
    patches = mcgr.build_lut_check_set()
    assert len(patches) == 35
    assert len(set(patches)) == 35
    assert (128, 128, 0) in patches and (64, 64, 192) in patches
    assert (51, 0, 0) in patches and (0, 0, 204) in patches


# ── 計画（シャッフル・番兵） ──────────────────────────────────────────


def test_plan_jobs_is_deterministic_and_seed_sensitive():
    """同一シードで完全一致、異なるシードで順序が変わる。

    落ちる誤実装: シード無視（random.shuffle を素で呼ぶ）、job_id の
    非連番化。
    """
    raw = mcgr.build_raw_patch_set()
    lut = mcgr.build_lut_check_set()
    plan_a = mcgr.plan_jobs(raw, lut, seed=0)
    plan_b = mcgr.plan_jobs(raw, lut, seed=0)
    plan_c = mcgr.plan_jobs(raw, lut, seed=1)
    assert plan_a == plan_b
    assert [j.requested_rgb for j in plan_a] != [j.requested_rgb for j in plan_c]
    assert [j.job_id for j in plan_a] == list(range(len(plan_a)))


def test_plan_jobs_sentinel_placement_and_totals():
    """321+35=356 計測 + 番兵13（先頭・30枚ごと11・末尾）= 369 ジョブ。

    落ちる誤実装: 番兵の挿入間隔の off-by-one、先頭/末尾の番兵欠落、
    計測ジョブの脱落や重複。
    """
    raw = mcgr.build_raw_patch_set()
    lut = mcgr.build_lut_check_set()
    plan = mcgr.plan_jobs(raw, lut, seed=0, sentinel_every=30)
    sentinels = [j for j in plan if j.drive == mcgr.DRIVE_SENTINEL]
    measurements = [j for j in plan if j.drive != mcgr.DRIVE_SENTINEL]
    assert len(measurements) == 356
    assert len(sentinels) == 13
    assert len(plan) == 369
    assert plan[0].drive == mcgr.DRIVE_SENTINEL
    assert plan[-1].drive == mcgr.DRIVE_SENTINEL
    # 計測ジョブは (drive, rgb) で全ユニーク（raw と lut_gray は独立）
    keys = [(j.drive, j.requested_rgb) for j in measurements]
    assert len(set(keys)) == 356
    # 番兵はすべて同色
    assert all(j.requested_rgb == mcgr.SENTINEL_RGB for j in sentinels)


def test_plan_jobs_raw_before_lut():
    """生駆動セットが LUT 通しセットより先に計画される。

    落ちる誤実装: 2セットを混ぜてシャッフルする（LUT ファイル破損時に
    生データまで失う順序になる）。
    """
    plan = mcgr.plan_jobs(
        mcgr.build_raw_patch_set(), mcgr.build_lut_check_set(), seed=0
    )
    drives = [j.drive for j in plan if j.drive != mcgr.DRIVE_SENTINEL]
    first_lut = drives.index(mcgr.DRIVE_LUT_GRAY)
    assert all(d == mcgr.DRIVE_LUT_GRAY for d in drives[first_lut:])


# ── 再開フィルタ ──────────────────────────────────────────────────────


def test_resume_skips_only_ok_records(tmp_path):
    """status=="ok" の job_id だけがスキップされる。

    落ちる誤実装: error 行も計測済み扱いにする（失敗パッチが永遠に
    埋まらない）、壊れた行で全体を落とす。
    """
    jsonl = tmp_path / "measurements.jsonl"
    jsonl.write_text(
        json.dumps({"job_id": 0, "status": "ok"}) + "\n"
        + json.dumps({"job_id": 1, "status": "error", "error": "x"}) + "\n"
        + "{corrupt line\n"
        + json.dumps({"job_id": 3, "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    completed = mcgr.load_completed_job_ids(jsonl)
    assert completed == {0, 3}

    jobs = [
        mcgr.PatchJob(job_id=i, drive=mcgr.DRIVE_RAW, requested_rgb=(i, i, i))
        for i in range(5)
    ]
    pending = mcgr.filter_pending(jobs, completed)
    assert [j.job_id for j in pending] == [1, 2, 4]


def test_resume_with_missing_file_runs_everything(tmp_path):
    """JSONL が無ければ全ジョブが未計測扱い。

    落ちる誤実装: 存在しないファイルで例外を出す（初回実行が不能になる）。
    """
    completed = mcgr.load_completed_job_ids(tmp_path / "none.jsonl")
    assert completed == set()


# ── LUT 駆動色 ────────────────────────────────────────────────────────


def _make_lut(scale: float) -> InverseLut:
    """t -> scale*t の線形逆LUT（恒等は scale=1.0）。"""
    t_grid = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    lut = np.tile(t_grid * scale, (3, 1))
    return InverseLut(lut=lut, t_grid=t_grid, meta={"source": "gray"})


def test_projected_rgb_identity_lut_passes_through():
    """恒等LUTなら投影色 == 要求色。

    落ちる誤実装: LUT を raw/sentinel にも適用する、正規化の取り違え
    （/256 等）。
    """
    job = mcgr.PatchJob(job_id=0, drive=mcgr.DRIVE_LUT_GRAY, requested_rgb=(51, 102, 204))
    assert mcgr.projected_rgb_for_job(job, _make_lut(1.0)) == (51, 102, 204)
    raw_job = mcgr.PatchJob(job_id=1, drive=mcgr.DRIVE_RAW, requested_rgb=(10, 20, 30))
    assert mcgr.projected_rgb_for_job(raw_job, None) == (10, 20, 30)


def test_projected_rgb_half_lut_halves():
    """t->t/2 のLUTなら投影色は要求色の約半分。

    落ちる誤実装: LUT 適用が no-op（raw と同じ値を返す）。
    """
    job = mcgr.PatchJob(job_id=0, drive=mcgr.DRIVE_LUT_GRAY, requested_rgb=(200, 100, 50))
    projected = mcgr.projected_rgb_for_job(job, _make_lut(0.5))
    assert projected == (100, 50, 25)


def test_projected_rgb_lut_drive_without_lut_raises():
    """lut_gray 駆動で LUT が None なら明示的に失敗する。

    落ちる誤実装: None を黙って素通しし、生駆動の重複計測になる。
    """
    job = mcgr.PatchJob(job_id=0, drive=mcgr.DRIVE_LUT_GRAY, requested_rgb=(1, 2, 3))
    with pytest.raises(ValueError):
        mcgr.projected_rgb_for_job(job, None)
