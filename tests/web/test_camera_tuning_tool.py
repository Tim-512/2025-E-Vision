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
    native_api_factory = lambda: calls.__setitem__("native", calls["native"] + 1) or native_api

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
    monkeypatch.setattr(
        server,
        "build_detector",
        lambda detection_config, **kwargs: calls.__setitem__(
            "detector_build", (detection_config, kwargs)
        )
        or fake_detector,
    )
    monkeypatch.setattr(server, "create_camera_tuning_app", lambda *args, **kwargs: calls.__setitem__("app", (args, kwargs)) or fake_app)

    args = server.build_parser().parse_args(["--output", str(tmp_path), "--serial", "SERIAL-X"])
    app = server.build_application(args, native_api_factory=native_api_factory)

    assert app is fake_app
    assert calls["native"] == 1
    service_kwargs = calls["service"]
    assert service_kwargs["base_config"] == camera_config
    assert service_kwargs["detector"] is fake_detector
    assert service_kwargs["detection_config"] == config.detection
    assert calls["detector_build"][0] == config.detection
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

class _PortableBackend:
    def __init__(self, artifact_path: Path, *, input_size: tuple[int, int], device: int) -> None:
        self.artifact_path = Path(artifact_path)
        self.input_size = input_size
        self.device = device
        self.artifact_kind = self.artifact_path.suffix.lstrip(".")

    def infer(self, tensor):
        return ()


class _RecordingBackendFactory:
    def __init__(self, *, fail_paths: set[Path] | None = None) -> None:
        self.fail_paths = fail_paths or set()
        self.paths: list[Path] = []

    def __call__(self, artifact_path: Path, **kwargs):
        path = Path(artifact_path)
        self.paths.append(path)
        if path in self.fail_paths:
            raise RuntimeError(f"cannot load {path.name}")
        return _PortableBackend(path, **kwargs)


def test_server_prefers_engine_then_falls_back_to_onnx(tmp_path: Path) -> None:
    from ev_vision.config import DetectionConfig, ModelDetectionConfig
    from ev_vision.web.camera_tuning_server import build_detector

    engine = tmp_path / "target-board.engine"
    onnx = tmp_path / "target-board.onnx"
    engine.write_bytes(b"engine")
    onnx.write_bytes(b"onnx")
    loader = _RecordingBackendFactory(fail_paths={engine})
    config = DetectionConfig(
        model=ModelDetectionConfig(
            path=str(engine),
            fallback_path=str(onnx),
            input_width=320,
            input_height=256,
            device=7,
        )
    )

    detector = build_detector(config, backend_factory=loader)

    assert loader.paths == [engine, onnx]
    assert detector.model_state == "READY"
    assert detector.model_backend == "onnx"
    assert detector.model_path == str(onnx)
    assert detector.model_errors == ()


def test_backend_selection_collects_missing_and_load_errors(tmp_path: Path) -> None:
    from ev_vision.config import DetectionConfig, ModelDetectionConfig
    from ev_vision.web.camera_tuning_server import build_detection_backend

    engine = tmp_path / "missing.engine"
    onnx = tmp_path / "target-board.onnx"
    onnx.write_bytes(b"onnx")
    loader = _RecordingBackendFactory(fail_paths={onnx})
    config = DetectionConfig(
        model=ModelDetectionConfig(path=str(engine), fallback_path=str(onnx))
    )

    selection = build_detection_backend(config, backend_factory=loader)

    assert selection.backend is None
    assert selection.path is None
    assert selection.artifact_kind == "classical-diagnostic"
    assert loader.paths == [onnx]
    assert len(selection.errors) == 2
    assert str(engine) in selection.errors[0] and "missing" in selection.errors[0]
    assert str(onnx) in selection.errors[1] and "cannot load" in selection.errors[1]


def test_missing_models_builds_invalid_diagnostic_detector_without_blocking_app(
    monkeypatch, tmp_path: Path
) -> None:
    import numpy as np
    import ev_vision.web.camera_tuning_server as server
    from ev_vision.config import (
        AppConfig,
        BoardConfig,
        CameraConfig,
        CircleConfig,
        ControlConfig,
        DetectionConfig,
        LaserConfig,
        ModelDetectionConfig,
        SerialConfig,
    )
    from ev_vision.detection.failures import DetectionFailure

    detection = DetectionConfig(
        model=ModelDetectionConfig(
            path=str(tmp_path / "missing.engine"),
            fallback_path=str(tmp_path / "missing.onnx"),
        )
    )
    config = AppConfig(
        CameraConfig(),
        BoardConfig(),
        CircleConfig(),
        ControlConfig(),
        SerialConfig(),
        LaserConfig(),
        detection,
    )
    monkeypatch.setattr(server, "load_config", lambda path: config)

    app = server.build_application(
        server.build_parser().parse_args(["--output", str(tmp_path / "output")]),
        native_api_factory=lambda: object(),
        backend_factory=lambda *args, **kwargs: pytest.fail(
            "backend factory must not be called for missing artifacts"
        ),
    )

    service = app.state.service
    detector = service._detector
    result = detector.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        captured_ns=1,
        source_sequence=1,
    )
    status = service.latest_detection()

    assert status.model_state == "UNAVAILABLE"
    assert status.target_valid is False
    assert result.model_state == "UNAVAILABLE"
    assert result.model_backend == "classical-diagnostic"
    assert result.model_path is None
    assert result.failure_reason is DetectionFailure.MODEL_UNAVAILABLE
    assert len(detector.model_errors) == 2
