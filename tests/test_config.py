from pathlib import Path

import pytest

from ev_vision.config import ConfigError, load_config


@pytest.fixture
def default_config_path() -> Path:
    return Path("config/default.yaml")


def test_default_config_is_valid_for_mock_mode(default_config_path: Path) -> None:
    cfg = load_config(default_config_path, hardware_required=False)
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 1024
    assert cfg.serial.baudrate == 921600
    assert cfg.control.command_hz == 100
    assert cfg.board.rectified_px_per_cm == pytest.approx(40.0)


def test_hardware_mode_rejects_missing_rate_limits(default_config_path: Path) -> None:
    with pytest.raises(ConfigError, match="max_yaw_rate_deg_s"):
        load_config(default_config_path, hardware_required=True)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("camera: {}\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration keys"):
        load_config(path)


def test_invalid_command_rate_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    text = default_config_path.read_text(encoding="utf-8").replace("command_hz: 100", "command_hz: 0")
    path = tmp_path / "bad-rate.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="command_hz"):
        load_config(path)
