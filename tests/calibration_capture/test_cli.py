from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TOOL_PATH = Path(__file__).parents[2] / "tools" / "camera_calibration_capture.py"


def _analysis(*, save_allowed: bool, signature=(0.5, 0.5, 0.2, 0.0)):
    from ev_vision.calibration_capture.local import ChessboardFrameAnalysis

    return ChessboardFrameAnalysis(
        found=save_allowed,
        corners=None,
        focus_score=100.0,
        coverage_fraction=0.2 if save_allowed else 0.0,
        edge_margin_ok=save_allowed,
        pose_signature=signature if save_allowed else None,
        save_allowed=save_allowed,
        reason="ready" if save_allowed else "corners_not_found",
    )


class FakeCamera:
    def __init__(self, events, frames):
        self.events = events
        self.frames = iter(frames)

    def open(self):
        self.events.append("open")
        return self

    def read(self, *, timeout_ms):
        self.events.append(("read", timeout_ms))
        item = next(self.frames)
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(image=item)

    def close(self):
        self.events.append("close")


class FakeCv:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 1
    INTER_AREA = 3

    def __init__(self, events, keys, *, visible=1.0):
        self.events = events
        self.keys = iter(keys)
        self.visible = visible
        self.images = []
        self.resize_inputs = []

    def namedWindow(self, name, flags):
        self.events.append(("named", name, flags))

    def resize(self, image, size, interpolation=None):
        self.resize_inputs.append((image, size, interpolation))
        height = max(1, round(image.shape[0] * size[0] / image.shape[1]))
        return np.zeros((height, size[0], 3), np.uint8)

    def imshow(self, name, image):
        self.images.append(image)

    def waitKey(self, delay):
        return next(self.keys)

    def getWindowProperty(self, name, prop):
        return self.visible

    def destroyWindow(self, name):
        self.events.append(("destroy", name))


@pytest.mark.parametrize(
    "argv",
    [
        ["--columns", "1"],
        ["--rows", "1"],
        ["--square-mm", "0"],
        ["--square-mm", "nan"],
        ["--width", "0"],
        ["--timeout-ms", "0"],
    ],
)
def test_parser_rejects_invalid_values(argv) -> None:
    from ev_vision.calibration_capture.cli import build_parser

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [["--columns", "1"], ["--square-mm", "inf"], ["--timeout-ms", "0"]],
)
def test_main_returns_argument_error_code(argv) -> None:
    from ev_vision.calibration_capture.cli import main

    assert main(argv) == 2


def test_parser_defaults_match_jetson_capture_command() -> None:
    from ev_vision.calibration_capture.cli import build_parser

    args = build_parser().parse_args([])
    assert args.config == Path("config/jetson-local.yaml")
    assert args.serial == "00G02809155"
    assert (args.columns, args.rows, args.square_mm) == (8, 5, 22.0)
    assert args.output == Path("artifacts/calibration/images")
    assert args.width == 960
    assert args.timeout_ms == 100


def test_pyproject_registers_entrypoint_and_wrapper_is_two_lines() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["ev-camera-calibration-capture"] == (
        "ev_vision.calibration_capture.cli:main"
    )
    assert TOOL_PATH.read_text(encoding="utf-8").splitlines() == [
        "from ev_vision.calibration_capture.cli import main",
        "raise SystemExit(main())",
    ]


def test_build_camera_loads_config_requires_fixed_format_and_wires_native_api(monkeypatch) -> None:
    import ev_vision.calibration_capture.cli as cli

    camera_config = object()
    app_config = SimpleNamespace(camera=camera_config)
    calls = []
    expected_camera = object()
    monkeypatch.setattr(cli, "load_config", lambda path: calls.append(("load", path)) or app_config)
    monkeypatch.setattr(cli, "require_fixed_format", lambda config: calls.append(("fixed", config)))
    monkeypatch.setattr(cli, "create_native_api", lambda: calls.append(("api",)) or "native")
    monkeypatch.setattr(
        cli,
        "HikrobotCamera",
        lambda api, config, serial_number=None: calls.append(
            ("camera", api, config, serial_number)
        )
        or expected_camera,
    )

    camera, loaded = cli.build_camera(Path("config/test.yaml"), serial="SERIAL")

    assert camera is expected_camera
    assert loaded is app_config
    assert calls == [
        ("load", Path("config/test.yaml")),
        ("fixed", camera_config),
        ("api",),
        ("camera", "native", camera_config, "SERIAL"),
    ]


def test_capture_saves_original_frame_despite_duplicate_and_resizes_only_preview(monkeypatch, tmp_path) -> None:
    import ev_vision.calibration_capture.cli as cli

    events = []
    first = np.full((1024, 1280, 3), 11, np.uint8)
    second = np.full((1024, 1280, 3), 22, np.uint8)
    third = np.full((1024, 1280, 3), 33, np.uint8)
    camera = FakeCamera(events, [first, second, third])
    cv = FakeCv(events, [ord(" "), ord(" "), ord("q")])
    saved = []
    analyses = iter(
        [
            _analysis(save_allowed=True),
            _analysis(save_allowed=True),
            _analysis(save_allowed=False),
        ]
    )
    duplicate_flags = []

    monkeypatch.setattr(cli, "analyze_chessboard_frame", lambda image, **kwargs: next(analyses))
    monkeypatch.setattr(
        cli,
        "render_capture_overlay",
        lambda image, analysis, saved_count, duplicate_warning, pattern_size: duplicate_flags.append(
            (duplicate_warning, pattern_size)
        )
        or image.copy(),
    )
    monkeypatch.setattr(
        cli,
        "save_original_frame",
        lambda output, image, index: saved.append((output, image, index))
        or output / f"calibration-{index:03d}.png",
    )

    result = cli.run_capture_session(
        camera,
        pattern_size=(8, 5),
        square_mm=22.0,
        output=tmp_path,
        preview_width=960,
        timeout_ms=100,
        cv=cv,
    )

    assert result == "key"
    assert events.count("open") == 1 and events.count("close") == 1
    assert events[-1][0] == "destroy"
    assert [(image is first, image is second, index) for _, image, index in saved] == [
        (True, False, 1),
        (False, True, 2),
    ]
    assert [entry[0] for entry in duplicate_flags] == [False, True, False]
    assert all(entry[1] == (8, 5) for entry in duplicate_flags)
    assert all(image.shape == (768, 960, 3) for image in cv.images)


def test_space_only_saves_allowed_and_remove_is_current_session_scoped(monkeypatch, tmp_path) -> None:
    import ev_vision.calibration_capture.cli as cli

    events = []
    frames = [np.zeros((20, 30, 3), np.uint8) for _ in range(4)]
    camera = FakeCamera(events, frames)
    cv = FakeCv(events, [ord(" "), ord(" "), ord("r"), ord("q")])
    analyses = iter(
        [
            _analysis(save_allowed=False),
            _analysis(save_allowed=True),
            _analysis(save_allowed=True),
            _analysis(save_allowed=False),
        ]
    )
    saved_path = tmp_path / "calibration-001.png"
    removed_paths = []
    session_signatures = []

    monkeypatch.setattr(cli, "analyze_chessboard_frame", lambda image, **kwargs: next(analyses))
    monkeypatch.setattr(cli, "render_capture_overlay", lambda image, *args, **kwargs: image.copy())
    monkeypatch.setattr(cli, "save_original_frame", lambda output, image, index: saved_path)

    def remove(paths):
        removed_paths.append(list(paths))
        return paths.pop() if paths else None

    monkeypatch.setattr(cli, "remove_last_saved", remove)

    result = cli.run_capture_session(
        camera,
        pattern_size=(8, 5),
        square_mm=22.0,
        output=tmp_path,
        preview_width=960,
        timeout_ms=100,
        cv=cv,
        session_signatures=session_signatures,
    )

    assert result == "key"
    assert removed_paths == [[saved_path]]
    assert session_signatures == []


def test_transient_timeout_continues_and_window_close_exits(monkeypatch, tmp_path) -> None:
    import ev_vision.calibration_capture.cli as cli

    events = []
    frame = np.zeros((20, 30, 3), np.uint8)
    camera = FakeCamera(events, [TimeoutError("late"), frame])
    cv = FakeCv(events, [-1], visible=0.0)
    monkeypatch.setattr(cli, "analyze_chessboard_frame", lambda image, **kwargs: _analysis(save_allowed=False))
    monkeypatch.setattr(cli, "render_capture_overlay", lambda image, *args, **kwargs: image.copy())

    assert (
        cli.run_capture_session(
            camera,
            pattern_size=(8, 5),
            square_mm=22.0,
            output=tmp_path,
            preview_width=960,
            timeout_ms=100,
            cv=cv,
        )
        == "window"
    )
    assert events.count(("read", 100)) == 2
    assert events.count("close") == 1
    assert events[-1][0] == "destroy"


def test_exception_still_closes_camera_and_window(monkeypatch, tmp_path) -> None:
    import ev_vision.calibration_capture.cli as cli

    events = []
    frame = np.zeros((20, 30, 3), np.uint8)
    camera = FakeCamera(events, [frame])
    cv = FakeCv(events, [ord("q")])
    monkeypatch.setattr(
        cli,
        "analyze_chessboard_frame",
        lambda image, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        cli.run_capture_session(
            camera,
            pattern_size=(8, 5),
            square_mm=22.0,
            output=tmp_path,
            preview_width=960,
            timeout_ms=100,
            cv=cv,
        )
    assert events.count("close") == 1
    assert events[-1][0] == "destroy"


def test_main_requires_display_before_building_camera(monkeypatch, capsys) -> None:
    import ev_vision.calibration_capture.cli as cli

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        cli,
        "build_camera",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    assert cli.main([]) == 2
    assert "DISPLAY" in capsys.readouterr().err


def test_main_prints_parameters_guidance_and_safety_then_runs(monkeypatch, capsys) -> None:
    import ev_vision.calibration_capture.cli as cli

    camera = object()
    config = SimpleNamespace(
        camera=SimpleNamespace(exposure_us=50000, gain_db=14.0, acquisition_fps=20)
    )
    recorded = {}
    monkeypatch.setattr(cli, "build_camera", lambda path, serial: (camera, config))
    monkeypatch.setattr(
        cli,
        "run_capture_session",
        lambda candidate, **kwargs: recorded.update(camera=candidate, **kwargs) or "key",
    )

    assert cli.main([]) == 0
    assert recorded["camera"] is camera
    output = capsys.readouterr().out.lower()
    assert "50000" in output and "14.0" in output and "20" in output
    assert "20-25" in output and "at least 15" in output and "varied poses" in output
    assert "405" in output and ("disconnect" in output or "cover" in output)


def test_main_exit_codes_for_keyboard_interrupt_startup_and_runtime_failures(monkeypatch, capsys) -> None:
    import ev_vision.calibration_capture.cli as cli

    monkeypatch.setattr(
        cli,
        "build_camera",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    )
    assert cli.main([]) == 2
    assert "startup" in capsys.readouterr().err.lower()

    config = SimpleNamespace(
        camera=SimpleNamespace(exposure_us=1, gain_db=2, acquisition_fps=3)
    )
    monkeypatch.setattr(cli, "build_camera", lambda *args, **kwargs: (object(), config))
    monkeypatch.setattr(
        cli,
        "run_capture_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )
    assert cli.main([]) == 3
    assert "capture failed" in capsys.readouterr().err

    monkeypatch.setattr(
        cli,
        "run_capture_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert cli.main([]) == 0


def test_tools_wrapper_delegates_to_cli(monkeypatch) -> None:
    import ev_vision.calibration_capture.cli as cli

    monkeypatch.setattr(cli, "main", lambda: 7)
    spec = importlib.util.spec_from_file_location("camera_calibration_capture", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(SystemExit) as exc_info:
        spec.loader.exec_module(module)
    assert exc_info.value.code == 7
