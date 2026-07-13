from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).parents[2] / "tools" / "camera_smoke_test.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("camera_smoke_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_builds_camera_runs_smoke_saves_latest_sample_and_prints_json(tmp_path, monkeypatch, capsys):
    tool = load_tool()
    config = object()
    fake_camera = object()
    stats = tool.CameraSmokeStats(
        elapsed_s=2.0,
        frames=200,
        timeouts=1,
        sequence_gaps=0,
        non_monotonic_timestamps=0,
        first_sequence=10,
        last_sequence=209,
        start_rss_bytes=1000,
        end_rss_bytes=1200,
        disconnect=None,
    )
    sample = tmp_path / "sample.jpg"
    saved = []

    monkeypatch.setattr(tool, "load_config", lambda path: type("Cfg", (), {"camera": config})())
    monkeypatch.setattr(tool, "create_native_api", lambda: "api")
    monkeypatch.setattr(tool, "HikrobotCamera", lambda api, cfg, serial_number=None: fake_camera)
    monkeypatch.setattr(tool, "run_with_camera", lambda camera, **kwargs: (stats, "frame-image"))
    monkeypatch.setattr(tool, "save_image", lambda path, image: saved.append((path, image)))

    rc = tool.main(
        [
            "--config",
            "config/default.yaml",
            "--serial",
            "00G02809155",
            "--duration",
            "2",
            "--timeout-ms",
            "100",
            "--sample",
            str(sample),
        ]
    )

    assert rc == 0
    assert saved == [(sample, "frame-image")]
    output = capsys.readouterr().out
    assert '"frames": 200' in output
    assert '"timeout_rate": 0.004975124378109453' in output


def test_tool_returns_failure_on_disconnect(monkeypatch, capsys):
    tool = load_tool()
    config = object()
    fake_camera = object()
    stats = tool.CameraSmokeStats(
        elapsed_s=1.0,
        frames=5,
        timeouts=0,
        sequence_gaps=0,
        non_monotonic_timestamps=0,
        first_sequence=1,
        last_sequence=5,
        start_rss_bytes=1000,
        end_rss_bytes=1000,
        disconnect="camera removed",
    )
    monkeypatch.setattr(tool, "load_config", lambda path: type("Cfg", (), {"camera": config})())
    monkeypatch.setattr(tool, "create_native_api", lambda: "api")
    monkeypatch.setattr(tool, "HikrobotCamera", lambda api, cfg, serial_number=None: fake_camera)
    monkeypatch.setattr(tool, "run_with_camera", lambda camera, **kwargs: (stats, None))

    assert tool.main(["--duration", "1"]) == 2
    assert '"disconnect": "camera removed"' in capsys.readouterr().out
