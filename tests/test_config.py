from pathlib import Path

import pytest
import yaml

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

def write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def base_config_without_detection() -> dict[str, object]:
    return {}


def write_detection_config(tmp_path: Path, patch: dict[str, object]) -> Path:
    return write_config(tmp_path, {"detection": patch})


def test_detection_defaults_and_weight_normalization(tmp_path: Path) -> None:
    path = write_config(tmp_path, base_config_without_detection())
    config = load_config(path)

    assert config.detection.model.max_candidates == 3
    assert config.detection.roi_geometry.padding_fraction == pytest.approx(0.08)
    assert sum(config.detection.candidate_scoring.weights) == pytest.approx(1.0)
    assert config.detection.tracking.confirm_frames == 3


def test_detection_weights_are_normalized(tmp_path: Path) -> None:
    path = write_detection_config(
        tmp_path,
        {
            "candidate_scoring": {
                "model_weight": 4.0,
                "geometry_weight": 3.0,
                "structure_weight": 2.0,
                "temporal_weight": 1.0,
            }
        },
    )
    config = load_config(path)

    assert config.detection.candidate_scoring.weights == pytest.approx(
        (0.4, 0.3, 0.2, 0.1)
    )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"model": {"confidence_threshold": 1.2}}, "confidence_threshold"),
        ({"model": {"max_candidates": 0}}, "max_candidates"),
        ({"roi_geometry": {"canny_low": 200, "canny_high": 100}}, "canny_low"),
        ({"candidate_scoring": {"model_weight": -1.0}}, "weight"),
        ({"tracking": {"max_result_age_ms": 0}}, "max_result_age_ms"),
    ],
)
def test_invalid_detection_configuration_is_rejected(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    path = write_detection_config(tmp_path, patch)
    with pytest.raises(ConfigError, match=message):
        load_config(path)

@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"backend": "unsupported"}, "backend"),
        ({"model": {"unknown": 1}}, "unknown configuration keys"),
        ({"roi_geometry": {"min_geometry_score": 1.1}}, "min_geometry_score"),
        ({"candidate_scoring": {"ambiguity_margin": -0.01}}, "ambiguity_margin"),
        ({"tracking": {"confirm_frames": 0}}, "confirm_frames"),
    ],
)
def test_detection_configuration_rejects_additional_invalid_values(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    path = write_detection_config(tmp_path, patch)
    with pytest.raises(ConfigError, match=message):
        load_config(path)
