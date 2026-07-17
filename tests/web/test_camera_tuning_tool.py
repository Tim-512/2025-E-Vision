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
    assert "hardware-always-on" in help_text
    assert "jetson/v2" in help_text and "cannot" in help_text


def test_build_detector_uses_classical_without_model_factory() -> None:
    from ev_vision.config import BoardConfig, DetectionConfig
    from ev_vision.detection.classical_board import ClassicalBoardDetector
    from ev_vision.web.camera_tuning_server import build_detector

    detector = build_detector(
        DetectionConfig(backend="classical"),
        board=BoardConfig(),
        backend_factory=lambda *_args, **_kwargs: pytest.fail("YOLO factory called"),
    )

    assert isinstance(detector, ClassicalBoardDetector)


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
    assert "laser" in output and "hardware-always-on" in output
    assert "cannot make it safe" in output
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
        backend="hybrid",
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
        backend="hybrid",
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



def _static_text(name: str) -> str:
    return Path("src/ev_vision/web/static", name).read_text(encoding="utf-8")


def _javascript_function(script: str, name: str) -> str:
    marker = f"async function {name}"
    start = script.index(marker)
    next_function = script.find("\nasync function ", start + len(marker))
    if next_function < 0:
        next_function = len(script)
    return script[start:next_function]


def test_dashboard_contains_hybrid_detection_controls_and_current_routes() -> None:
    html = _static_text("camera-tuning.html")
    script = _static_text("camera-tuning.js")

    for marker in (
        'id="detection-model-state"',
        'id="detection-model-backend"',
        'id="detection-model-path"',
        'id="detection-tracking-state"',
        'id="detection-failure-reason"',
        'id="detection-candidate-list"',
        'id="detection-target-valid"',
        'id="detection-processing-rate"',
        'id="detection-confirmation-count"',
        'id="detection-miss-count"',
        'id="detection-result-age"',
        'id="detection-score-components"',
        'id="detection-confidence-threshold"',
        'id="detection-max-candidates"',
        'id="detection-canny-low"',
        'id="detection-canny-high"',
        'id="detection-min-edge-support"',
        'id="detection-min-geometry-score"',
        'id="detection-model-weight"',
        'id="detection-geometry-weight"',
        'id="detection-structure-weight"',
        'id="detection-temporal-weight"',
        'id="detection-ambiguity-margin"',
        'id="detection-confirm-frames"',
        'id="detection-predict-frames"',
        'id="detection-lost-frames"',
        'id="detection-max-center-jump-px"',
        'id="detection-max-result-age-ms"',
        'id="apply-detection-config"',
        'id="restore-detection-defaults"',
        'id="detection-debug-view"',
        'id="refresh-detection-debug"',
        'id="detection-debug-image"',
        'id="reload-detection-model"',
    ):
        assert marker in html

    assert "/api/detection/status" in script
    assert "/api/detection/config" in script
    assert "/api/detection/debug" in script
    assert "/api/detection/model/reload" in script
    assert "/api/detection/debug/" in script
    assert "payload.detail" in script
    assert "button.disabled = true" in script
    assert '$("detection-processing-rate").textContent = formatRate(runtime.detection_fps);' in script

    status_function = _javascript_function(script, "refreshDetectionStatus")
    assert "refreshDetectionDebug" not in status_function


def test_dashboard_hybrid_labels_are_utf8_and_not_question_mark_placeholders() -> None:
    html = _static_text("camera-tuning.html")
    script = _static_text("camera-tuning.js")

    for label in (
        "混合靶面检测",
        "目标有效",
        "目标无效",
        "确认帧数",
        "最大中心跳变",
        "重新加载模型",
    ):
        assert label in html + script
    assert "????" not in html
    assert "????" not in script


def test_invalid_hybrid_target_with_observation_does_not_draw_green_confirmation_geometry() -> None:
    import numpy as np

    import ev_vision.web.camera_tuning_app as camera_tuning_app
    from ev_vision.models import BoardObservation
    from ev_vision.tuning.models import DetectionSnapshot, OverlayOptions

    observation = BoardObservation(
        captured_ns=123,
        corners_px=((15.0, 55.0), (100.0, 55.0), (98.0, 82.0), (17.0, 82.0)),
        center_px=(57.0, 68.0),
        confidence=0.86,
        homography_valid=True,
    )
    snapshot = DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=9,
        observation=observation,
        target_valid=False,
        tracking_state="PREDICTING",
        center_px=observation.center_px,
        corners_px=observation.corners_px,
        failure_reason="PREDICTING",
    )
    rendered = camera_tuning_app.render_overlay(
        np.zeros((100, 120, 3), dtype=np.uint8),
        source_sequence=9,
        detection=snapshot,
        options=OverlayOptions(
            enabled=True,
            show_board_outline=True,
            show_corners=True,
            show_center=True,
            show_crosshair=False,
            show_detection_text=False,
            show_center_roi=False,
        ),
    )

    assert not np.any(np.all(rendered == np.array([0, 255, 0], dtype=np.uint8), axis=2))
    assert not np.any(np.all(rendered == np.array([0, 200, 0], dtype=np.uint8), axis=2))


def test_hybrid_overlay_draws_candidate_decisions_and_tracking_semantics(monkeypatch) -> None:
    import cv2
    import numpy as np

    import ev_vision.web.camera_tuning_app as camera_tuning_app
    from ev_vision.tuning.models import (
        DetectionCandidateSnapshot,
        DetectionSnapshot,
        OverlayOptions,
    )

    accepted = DetectionCandidateSnapshot(
        xyxy_px=(10.0, 10.0, 50.0, 45.0),
        accepted=True,
        model_confidence=0.91,
        geometry_score=0.88,
        edge_support_score=0.82,
        structure_score=0.79,
        temporal_score=0.76,
        combined_score=0.86,
    )
    rejected = DetectionCandidateSnapshot(
        xyxy_px=(65.0, 12.0, 105.0, 47.0),
        accepted=False,
        model_confidence=0.72,
        geometry_score=0.21,
        edge_support_score=0.18,
        structure_score=0.35,
        temporal_score=0.40,
        combined_score=0.39,
        failure_reason="GEOMETRY_REJECTED",
    )
    snapshot = DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=7,
        target_valid=True,
        tracking_state="TRACKING",
        model_state="READY",
        model_backend="onnx",
        model_path="models/target-board.onnx",
        combined_score=0.86,
        failure_reason=None,
        corners_px=((15.0, 55.0), (100.0, 55.0), (98.0, 82.0), (17.0, 82.0)),
        center_px=(57.0, 68.0),
        candidates=(accepted, rejected),
    )
    recorded_text: list[str] = []
    original_put_text = cv2.putText

    def record_put_text(image, text, *args, **kwargs):
        recorded_text.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(camera_tuning_app.cv2, "putText", record_put_text)
    rendered = camera_tuning_app.render_overlay(
        np.zeros((100, 120, 3), dtype=np.uint8),
        source_sequence=7,
        detection=snapshot,
        options=OverlayOptions(
            enabled=True,
            show_board_outline=True,
            show_corners=True,
            show_center=True,
            show_crosshair=False,
            show_detection_text=True,
            show_center_roi=False,
        ),
    )

    assert tuple(rendered[10, 25]) == (0, 255, 255)  # accepted/model candidate: yellow
    assert tuple(rendered[12, 80]) == (0, 0, 255)  # geometry rejection: red
    assert tuple(rendered[55, 40]) == (0, 255, 0)  # confirmed quadrilateral: green
    assert rendered[68, 57, 1] > 0  # target center marker
    overlay_text = " ".join(recorded_text)
    assert "TRACKING" in overlay_text
    assert "valid" in overlay_text.lower()
    assert "score=0.860" in overlay_text
    assert "GEOMETRY_REJECTED" in overlay_text
