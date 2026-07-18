from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_parser_defaults_to_project_yaml_and_low_memory_settings() -> None:
    from ev_vision.preview.cli import build_parser

    args = build_parser().parse_args([])
    assert args.config == Path("config/default.yaml")
    assert args.serial == "00G02809155"
    assert args.width == 640
    assert args.display_fps == 30.0
    assert args.detection_fps == 30.0


def test_main_builds_service_without_diagnostics_and_uses_yaml_camera_parameters(monkeypatch, capsys) -> None:
    import ev_vision.preview.cli as cli

    service = SimpleNamespace(start_calls=0, stop_calls=0)
    service.start = lambda: setattr(service, "start_calls", service.start_calls + 1)
    service.stop = lambda: setattr(service, "stop_calls", service.stop_calls + 1)
    config = SimpleNamespace(camera=SimpleNamespace(
        exposure_us=50000, gain_db=14.0, acquisition_fps=20,
        auto_exposure=False, auto_gain=False, auto_white_balance=False,
    ))
    recorded = {}

    def build_runtime(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(service=service, config=config)

    monkeypatch.setattr(cli, "build_camera_runtime", build_runtime)
    monkeypatch.setattr(cli, "run_local_preview", lambda candidate, **kwargs: recorded.update(preview=(candidate, kwargs)) or "key")

    assert cli.main(["--config", "config/test.yaml", "--display-fps", "25"]) == 0
    assert recorded["config_path"] == Path("config/test.yaml")
    assert recorded["diagnostics_fps"] is None
    assert recorded["detection_fps"] == 30.0
    assert recorded["preview"][0] is service
    assert recorded["preview"][1]["display_fps"] == 25.0
    assert service.start_calls == 1
    assert service.stop_calls == 1
    output = capsys.readouterr().out
    assert "50000" in output and "14.0" in output and "20" in output
    assert "laser" in output.lower()


def test_standalone_cli_module_does_not_import_web_or_uvicorn() -> None:
    text = Path("src/ev_vision/preview/cli.py").read_text(encoding="utf-8")
    assert "ev_vision.web" not in text
    assert "uvicorn" not in text