from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import ev_vision.gimbal_usb.cli as cli
from ev_vision.gimbal_usb.config import GimbalUsbConfig
from ev_vision.gimbal_usb.protocol import GimbalTargetCommand
from ev_vision.gimbal_usb.target_angles import (
    TargetAngleConverter,
    UnavailableTargetAngleConverter,
)


class AlreadySetEvent:
    def wait(self, timeout: float) -> bool:
        del timeout
        return True

    def set(self) -> None:
        pass


class FakeService:
    def __init__(self, events: list[str], *, start_error: Exception | None = None) -> None:
        self.events = events
        self.start_error = start_error

    def start(self) -> None:
        self.events.append("service.start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.events.append("service.stop")

    def runtime_snapshot(self):
        return SimpleNamespace(state="Connected", acquisition_fps=50.0, detection_fps=49.0)


class FakeWorker:
    def __init__(self, events: list[str], *, start_error: Exception | None = None) -> None:
        self.events = events
        self.start_error = start_error

    def start(self) -> None:
        self.events.append("worker.start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.events.append("worker.stop:5")

    def snapshot(self):
        return SimpleNamespace(
            ticks=10,
            sent_valid=3,
            sent_invalid=7,
            overruns=0,
            last_error=None,
            serial=SimpleNamespace(connected=True, sent_frames=10, last_error=None),
        )


def camera_runtime(events: list[str], *, start_error: Exception | None = None):
    camera = SimpleNamespace(
        width=1280,
        height=1024,
        pixel_format="BayerRG8",
        buffer_size=2,
        exposure_us=15000,
        gain_db=14.0,
        acquisition_fps=50,
        auto_exposure=False,
        auto_gain=False,
        auto_white_balance=False,
    )
    return SimpleNamespace(
        service=FakeService(events, start_error=start_error),
        config=SimpleNamespace(camera=camera),
    )


def gimbal_config(**changes) -> GimbalUsbConfig:
    base = GimbalUsbConfig(
        port="/dev/serial/by-id/usb-test",
        baudrate=115200,
        output_hz=50.0,
        reconnect_interval_s=1.0,
        calibration_path=Path("config/camera_calibration.yaml"),
        max_calibration_rms_px=0.5,
        max_result_age_ms=60.0,
        predicted_control_max_frames=3,
        predicted_control_max_age_ms=60.0,
        predicted_max_angle_step_deg=1.5,
        yaw_sign=1,
        pitch_sign=1,
    )
    return replace(base, **changes)


def test_parser_uses_runtime_defaults() -> None:
    args = cli.build_parser().parse_args([])

    assert args.config == Path("config/jetson-local.yaml")
    assert args.gimbal_config == Path("config/gimbal_usb.yaml")
    assert args.serial == "00G02809155"
    assert args.display is False
    assert (args.width, args.display_fps, args.detection_fps) == (640, 45.0, 50.0)
    assert args.timeout_ms == 100
    assert args.shutdown_timeout == 2.0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--width", "0"],
        ["--display-fps", "nan"],
        ["--detection-fps", "0"],
        ["--timeout-ms", "-1"],
        ["--shutdown-timeout", "inf"],
    ],
)
def test_parser_rejects_unsafe_numeric_values(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_only_display_mode_requires_display_before_building(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        cli,
        "load_gimbal_usb_config",
        lambda path: pytest.fail(f"must not load config: {path}"),
    )

    assert cli.main(["--display"]) == 2
    assert "DISPLAY is not set" in capsys.readouterr().err


def test_headless_mode_does_not_require_display(monkeypatch) -> None:
    events: list[str] = []
    runtime = camera_runtime(events)
    worker = FakeWorker(events)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(cli, "load_gimbal_usb_config", lambda path: gimbal_config())
    monkeypatch.setattr(cli, "build_camera_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(cli, "build_angle_converter", lambda *args, **kwargs: UnavailableTargetAngleConverter("test"))
    monkeypatch.setattr(cli, "build_worker", lambda *args, **kwargs: worker)
    monkeypatch.setattr(cli, "_new_stop_event", AlreadySetEvent)

    assert cli.main([]) == 0
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]


def test_missing_calibration_builds_fail_closed_converter(tmp_path: Path) -> None:
    config = gimbal_config(calibration_path=tmp_path / "missing.yaml")

    converter = cli.build_angle_converter(config, image_size=(1280, 1024))

    assert isinstance(converter, UnavailableTargetAngleConverter)
    assert converter.available is False
    assert "cannot load calibration" in converter.reason


def test_build_angle_converter_loads_valid_calibration(monkeypatch) -> None:
    expected = object()
    recorded = {}

    def from_file(path, **kwargs):
        recorded.update(path=path, **kwargs)
        return expected

    monkeypatch.setattr(TargetAngleConverter, "from_file", from_file)
    config = gimbal_config(
        calibration_path=Path("calibration.yaml"),
        max_calibration_rms_px=0.4,
        yaw_sign=-1,
        pitch_sign=1,
    )

    result = cli.build_angle_converter(config, image_size=(1280, 1024))

    assert result is expected
    assert recorded == {
        "path": Path("calibration.yaml"),
        "image_size": (1280, 1024),
        "max_rms_px": 0.4,
        "yaw_sign": -1,
        "pitch_sign": 1,
    }


def test_build_worker_wires_gate_transport_and_output_rate(monkeypatch) -> None:
    runtime = camera_runtime([])
    config = gimbal_config()
    converter = UnavailableTargetAngleConverter("missing")
    created = {}

    class Gate:
        def __init__(self, candidate, limits):
            created["gate"] = (candidate, limits)

    class Transport:
        def __init__(self, port, **kwargs):
            created["transport"] = (port, kwargs)

    class Worker:
        def __init__(self, service, gate, transport, **kwargs):
            created["worker"] = (service, gate, transport, kwargs)

    monkeypatch.setattr(cli, "GimbalControlGate", Gate)
    monkeypatch.setattr(cli, "GimbalSerialTransport", Transport)
    monkeypatch.setattr(cli, "GimbalOutputWorker", Worker)

    worker = cli.build_worker(runtime, config, converter=converter)

    assert isinstance(worker, Worker)
    candidate, limits = created["gate"]
    assert candidate is converter
    assert limits.image_size == (1280, 1024)
    assert limits.max_result_age_ms == 60.0
    assert limits.predicted_control_max_frames == 3
    assert limits.predicted_control_max_age_us == 60_000
    assert limits.predicted_max_angle_step_deg == 1.5
    assert created["transport"] == (
        "/dev/serial/by-id/usb-test",
        {"baudrate": 115200, "reconnect_interval_s": 1.0},
    )
    assert created["worker"][0] is runtime.service
    assert created["worker"][3] == {"output_hz": 50.0}


def test_run_application_starts_camera_before_worker_and_stops_worker_first() -> None:
    events: list[str] = []

    result = cli.run_application(
        camera_runtime(events),
        FakeWorker(events),
        display=False,
        stop_event=AlreadySetEvent(),
    )

    assert result == 0
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]


def test_run_application_uses_existing_local_preview(monkeypatch) -> None:
    events: list[str] = []
    runtime = camera_runtime(events)
    worker = FakeWorker(events)
    recorded = {}
    monkeypatch.setattr(
        cli,
        "run_local_preview",
        lambda service, **kwargs: recorded.update(service=service, kwargs=kwargs) or "key",
    )

    assert cli.run_application(
        runtime,
        worker,
        display=True,
        width=720,
        display_fps=40.0,
    ) == 0
    assert recorded == {
        "service": runtime.service,
        "kwargs": {"max_width": 720, "display_fps": 40.0},
    }
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]


def test_camera_start_failure_returns_startup_code_without_stopping_unstarted_components(capsys) -> None:
    events: list[str] = []

    result = cli.run_application(
        camera_runtime(events, start_error=RuntimeError("camera unavailable")),
        FakeWorker(events),
        display=False,
        stop_event=AlreadySetEvent(),
    )

    assert result == 2
    assert events == ["service.start"]
    assert "camera startup failed" in capsys.readouterr().err


def test_worker_start_failure_returns_runtime_code_and_stops_camera(capsys) -> None:
    events: list[str] = []

    result = cli.run_application(
        camera_runtime(events),
        FakeWorker(events, start_error=RuntimeError("worker failed")),
        display=False,
        stop_event=AlreadySetEvent(),
    )

    assert result == 3
    assert events == ["service.start", "worker.start", "service.stop"]
    assert "gimbal worker startup failed" in capsys.readouterr().err


def test_keyboard_interrupt_is_normal_and_cleanup_order_is_safe(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "run_local_preview", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    result = cli.run_application(
        camera_runtime(events),
        FakeWorker(events),
        display=True,
    )

    assert result == 0
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]


def test_runtime_preview_error_returns_three_after_safe_cleanup(monkeypatch, capsys) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "run_local_preview", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("display failed")))

    result = cli.run_application(camera_runtime(events), FakeWorker(events), display=True)

    assert result == 3
    assert events == ["service.start", "worker.start", "worker.stop:5", "service.stop"]
    assert "gimbal vision runtime failed" in capsys.readouterr().err


def test_main_configuration_failure_returns_two(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_gimbal_usb_config", lambda path: (_ for _ in ()).throw(ValueError("bad config")))

    assert cli.main([]) == 2
    assert "startup failed" in capsys.readouterr().err


def test_main_prints_camera_usb_calibration_and_laser_safety(monkeypatch, capsys) -> None:
    events: list[str] = []
    runtime = camera_runtime(events)
    worker = FakeWorker(events)
    converter = UnavailableTargetAngleConverter("RMS 0.700px exceeds 0.500px")
    monkeypatch.setattr(cli, "load_gimbal_usb_config", lambda path: gimbal_config())
    monkeypatch.setattr(cli, "build_camera_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(cli, "build_angle_converter", lambda *args, **kwargs: converter)
    monkeypatch.setattr(cli, "build_worker", lambda *args, **kwargs: worker)
    monkeypatch.setattr(cli, "_new_stop_event", AlreadySetEvent)

    assert cli.main([]) == 0
    output = capsys.readouterr()
    combined = output.out + output.err
    assert "15000" in combined
    assert "14.0" in combined
    assert "50" in combined
    assert "/dev/serial/by-id/usb-test" in combined
    assert "3 frames" in combined
    assert "60.0 ms" in combined
    assert "1.5" in combined
    assert "CALIBRATION ERROR" in combined
    assert "tracking=0" in combined
    assert "laser" in combined.lower()
    assert "physically" in combined.lower()


def test_main_passes_runtime_options_to_camera_builder(monkeypatch) -> None:
    events: list[str] = []
    runtime = camera_runtime(events)
    recorded = {}
    monkeypatch.setattr(cli, "load_gimbal_usb_config", lambda path: gimbal_config())

    def build_runtime(**kwargs):
        recorded.update(kwargs)
        return runtime

    monkeypatch.setattr(cli, "build_camera_runtime", build_runtime)
    monkeypatch.setattr(cli, "build_angle_converter", lambda *args, **kwargs: UnavailableTargetAngleConverter("test"))
    monkeypatch.setattr(cli, "build_worker", lambda *args, **kwargs: FakeWorker(events))
    monkeypatch.setattr(cli, "_new_stop_event", AlreadySetEvent)

    assert cli.main([
        "--config", "camera.yaml",
        "--gimbal-config", "gimbal.yaml",
        "--serial", "SERIAL",
        "--detection-fps", "40",
        "--timeout-ms", "120",
        "--shutdown-timeout", "3",
    ]) == 0
    assert recorded == {
        "config_path": Path("camera.yaml"),
        "serial": "SERIAL",
        "read_timeout_ms": 120,
        "detection_fps": 40.0,
        "diagnostics_fps": None,
        "shutdown_timeout_s": 3.0,
    }


def test_display_mode_does_not_replace_keyboard_interrupt_signal_handling(monkeypatch) -> None:
    events: list[str] = []
    runtime = camera_runtime(events)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli, "load_gimbal_usb_config", lambda path: gimbal_config())
    monkeypatch.setattr(cli, "build_camera_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(cli, "build_angle_converter", lambda *args, **kwargs: UnavailableTargetAngleConverter("test"))
    monkeypatch.setattr(cli, "build_worker", lambda *args, **kwargs: FakeWorker(events))
    monkeypatch.setattr(
        cli,
        "_install_stop_handlers",
        lambda event: pytest.fail("display mode must retain normal KeyboardInterrupt handling"),
    )
    monkeypatch.setattr(cli, "run_application", lambda *args, **kwargs: 0)

    assert cli.main(["--display"]) == 0


def test_cli_source_has_no_web_server_imports() -> None:
    source = Path("src/ev_vision/gimbal_usb/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "uvicorn" or name == "fastapi" or name.startswith("ev_vision.web") for name in imports)


def test_project_registers_formal_command() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'ev-gimbal-vision = "ev_vision.gimbal_usb.cli:main"' in text


def test_tool_wrapper_delegates_to_cli() -> None:
    text = Path("tools/gimbal_vision.py").read_text(encoding="utf-8")
    assert "from ev_vision.gimbal_usb.cli import main" in text
    assert "raise SystemExit(main())" in text
