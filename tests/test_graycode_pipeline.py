from pathlib import Path
from types import SimpleNamespace

from graycode import pipeline as graycode_pipeline


def _app_config(
    *,
    c2p_input: str = "result_c2p.npy",
    output_dir: str = "output",
    interpolation_method: str = "delaunay",
):
    return SimpleNamespace(
        pipeline=SimpleNamespace(default_input_file=c2p_input),
        paths=SimpleNamespace(output_dir=output_dir),
        interpolate_c2p=SimpleNamespace(default_method=interpolation_method),
    )


def test_run_graycode_pipeline_uses_latest_c2p_pipeline(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        graycode_pipeline,
        "get_config",
        lambda: _app_config(
            c2p_input="custom_c2p.npy",
            output_dir="custom_output",
            interpolation_method="delaunay",
        ),
    )
    monkeypatch.setattr(
        graycode_pipeline.gen_graycode,
        "main",
        lambda argv: calls.append(("gen", argv)),
    )
    monkeypatch.setattr(
        graycode_pipeline.cap_graycode,
        "main",
        lambda argv: calls.append(("cap", argv)),
    )

    def fake_decode(argv):
        calls.append(("decode", argv))
        return (720, 1280)

    monkeypatch.setattr(graycode_pipeline.decode, "main", fake_decode)
    monkeypatch.setattr(
        graycode_pipeline.interpolate_c2p,
        "main",
        lambda argv: calls.append(("c2p", argv)),
    )

    cfg = graycode_pipeline.GraycodePipelineConfig(
        proj_height=1080,
        proj_width=1920,
        height_step=2,
        width_step=3,
        window_pos_x=10,
        window_pos_y=20,
    )
    graycode_pipeline.run_graycode_pipeline(cfg)

    assert calls == [
        ("gen", ["gen_graycode.py", "1080", "1920", "2", "3"]),
        ("cap", ["cap_graycode.py", "10", "20"]),
        ("decode", ["decode.py", "1080", "1920", "2", "3"]),
        (
            "c2p",
            [
                "interpolate_c2p.py",
                str(Path("custom_output") / "custom_c2p.npy"),
                "720",
                "1280",
                "delaunay",
            ],
        ),
    ]


def test_run_graycode_pipeline_skips_decode_and_interpolation(monkeypatch, capsys) -> None:
    calls: list[str] = []
    monkeypatch.setattr(graycode_pipeline.gen_graycode, "main", lambda argv: None)
    monkeypatch.setattr(graycode_pipeline.cap_graycode, "main", lambda argv: None)
    monkeypatch.setattr(
        graycode_pipeline.decode,
        "main",
        lambda argv: calls.append("decode"),
    )
    monkeypatch.setattr(
        graycode_pipeline.interpolate_c2p,
        "main",
        lambda argv: calls.append("c2p"),
    )

    cfg = graycode_pipeline.GraycodePipelineConfig(
        proj_height=480,
        proj_width=640,
        run_decode=False,
    )
    graycode_pipeline.run_graycode_pipeline(cfg)

    assert calls == []
    captured = capsys.readouterr()
    assert "Skipped decode" in captured.out
    assert "Skipped interpolate" in captured.out
