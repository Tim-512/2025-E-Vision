from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_parser_defaults_and_safety_help() -> None:
    from ev_vision.web.camera_tuning_server import build_parser

    parser = build_parser()
    args = parser.parse_args([])

    assert args.config == Path("config/default.yaml")
    assert args.serial == "00G02809155"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.output == Path("artifacts/camera-tuning")
    assert args.preview_fps == 20.0
    assert args.detection_fps == 15.0
    assert args.diagnostic_fps == 10.0
    assert args.timeout_ms == 100
    assert args.shutdown_timeout == 2.0
    assert args.log_level == "info"

    help_text = parser.format_help().lower()
    assert "0.0.0.0" in help_text
    assert "trusted lan" in help_text
    assert "laser" in help_text
    assert "off" in help_text


def test_parser_rejects_non_positive_rates_ports_and_timeout() -> None:
    from ev_vision.web.camera_tuning_server import build_parser

    parser = build_parser()
    for argv in (
        ["--port", "0"],
        ["--port", "65536"],
        ["--preview-fps", "0"],
        ["--detection-fps", "-1"],
        ["--diagnostic-fps", "nan"],
        ["--timeout-ms", "0"],
        ["--shutdown-timeout", "0"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_build_application_enforces_format_and_wires_one_native_api(monkeypatch, tmp_path: Path) -> None:
    import ev_vision.web.camera_tuning_server as server
    from ev_vision.config import AppConfig, BoardConfig, CameraConfig, CircleConfig, ControlConfig, LaserConfig, SerialConfig

    camera_config = CameraConfig()
    config = AppConfig(camera_config, BoardConfig(), CircleConfig(), ControlConfig(), SerialConfig(), LaserConfig())
    native_api = object()
    calls: dict[str, Any] = {"native": 0, "cameras": []}

    monkeypatch.setattr(server, "load_config", lambda path: config)
    monkeypatch.setattr(server, "create_native_api", lambda: calls.__setitem__("native", calls["native"] + 1) or native_api)

    class FakeCamera:
        def __init__(self, api, candidate, *, serial_number):
            calls["cameras"].append((api, candidate, serial_number))

    class FakeService:
        def __init__(self, **kwargs):
            calls["service"] = kwargs

    class FakeStorage:
        def __init__(self, root, *, bounds):
            calls["storage"] = (root, bounds)

    fake_detector = object()
    fake_app = object()
    monkeypatch.setattr(server, "HikrobotCamera", FakeCamera)
    monkeypatch.setattr(server, "CameraTuningService", FakeService)
    monkeypatch.setattr(server, "TuningStorage", FakeStorage)
    monkeypatch.setattr(server, "BoardGeometryDetector", lambda: fake_detector)
    monkeypatch.setattr(server, "create_camera_tuning_app", lambda *args, **kwargs: calls.__setitem__("app", (args, kwargs)) or fake_app)

    args = server.build_parser().parse_args(["--output", str(tmp_path), "--serial", "SERIAL-X"])
    app = server.build_application(args)

    assert app is fake_app
    assert calls["native"] == 1
    service_kwargs = calls["service"]
    assert service_kwargs["base_config"] == camera_config
    assert service_kwargs["detector"] is fake_detector
    assert service_kwargs["read_timeout_ms"] == 100
    assert service_kwargs["diagnostics_fps"] == 10.0
    assert service_kwargs["detection_fps"] == 15.0
    assert service_kwargs["shutdown_timeout_s"] == 2.0
    assert service_kwargs["camera_identity"].model == "MV-CA013-21UC"
    assert service_kwargs["camera_identity"].serial == "SERIAL-X"

    candidate = CameraConfig(exposure_us=1500, gain_db=3.0)
    made = service_kwargs["camera_factory"](candidate)
    assert isinstance(made, FakeCamera)
    assert calls["cameras"] == [(native_api, candidate, "SERIAL-X")]
    assert calls["app"][1]["preview_fps"] == 20.0


def test_build_application_rejects_non_fixed_camera_format(monkeypatch) -> None:
    import ev_vision.web.camera_tuning_server as server
    from ev_vision.config import AppConfig, BoardConfig, CameraConfig, CircleConfig, ControlConfig, LaserConfig, SerialConfig

    bad = AppConfig(CameraConfig(width=640), BoardConfig(), CircleConfig(), ControlConfig(), SerialConfig(), LaserConfig())
    monkeypatch.setattr(server, "load_config", lambda path: bad)
    monkeypatch.setattr(server, "create_native_api", lambda: pytest.fail("native API must not be created"))

    with pytest.raises(ValueError, match="1280x1024.*BayerRG8.*buffer 2"):
        server.build_application(server.build_parser().parse_args([]))


def test_main_prints_safety_and_runs_uvicorn(monkeypatch, capsys) -> None:
    import ev_vision.web.camera_tuning_server as server

    fake_app = object()
    recorded: dict[str, Any] = {}
    monkeypatch.setattr(server, "build_application", lambda args: fake_app)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: recorded.update(app=app, **kwargs))

    result = server.main(["--host", "0.0.0.0", "--port", "8123", "--log-level", "warning"])

    assert result == 0
    assert recorded == {"app": fake_app, "host": "0.0.0.0", "port": 8123, "log_level": "warning"}
    output = capsys.readouterr().out.lower()
    assert "trusted lan" in output
    assert "laser" in output and "off" in output
    assert "http://0.0.0.0:8123" in output


def test_main_reports_build_errors_without_starting_uvicorn(monkeypatch, capsys) -> None:
    import ev_vision.web.camera_tuning_server as server

    monkeypatch.setattr(server, "build_application", lambda args: (_ for _ in ()).throw(ValueError("bad fixed format")))
    monkeypatch.setattr(server.uvicorn, "run", lambda *args, **kwargs: pytest.fail("must not run"))

    assert server.main([]) == 2
    assert "bad fixed format" in capsys.readouterr().err


def test_tool_wrapper_delegates_to_installed_module() -> None:
    text = Path("tools/camera_tuning_server.py").read_text(encoding="utf-8")
    assert "from ev_vision.web.camera_tuning_server import main" in text
    assert "SystemExit(main())" in text


def test_hybrid_acceptance_runbook_contains_required_safety_and_commands() -> None:
    text = Path("docs/hybrid-detector-acceptance.md").read_text(encoding="utf-8")
    for marker in (
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "models/target-board.onnx",
        "models/target-board.engine",
        "ev-camera-tuning",
        "TRACKING",
        "15-20 Hz",
        "20 minutes",
        "405 nm",
        "physically disconnected",
    ):
        assert marker in text
