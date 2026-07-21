import dataclasses
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

@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"model": {"input_width": 640.5}}, "input_width"),
        ({"model": {"input_width": True}}, "input_width"),
        ({"model": {"input_height": 640.5}}, "input_height"),
        ({"model": {"input_height": True}}, "input_height"),
        ({"model": {"max_candidates": 3.5}}, "max_candidates"),
        ({"model": {"max_candidates": True}}, "max_candidates"),
        ({"model": {"device": 0.5}}, "device"),
        ({"model": {"device": False}}, "device"),
        ({"roi_geometry": {"canny_low": 60.5}}, "canny_low"),
        ({"roi_geometry": {"canny_low": False}}, "canny_low"),
        ({"roi_geometry": {"canny_high": 180.5}}, "canny_high"),
        ({"roi_geometry": {"canny_high": True}}, "canny_high"),
        ({"tracking": {"confirm_frames": 3.5}}, "confirm_frames"),
        ({"tracking": {"confirm_frames": True}}, "confirm_frames"),
        ({"tracking": {"predict_frames": 2.5}}, "predict_frames"),
        ({"tracking": {"predict_frames": True}}, "predict_frames"),
        ({"tracking": {"lost_frames": 3.5}}, "lost_frames"),
        ({"tracking": {"lost_frames": True}}, "lost_frames"),
    ],
)
def test_detection_integer_settings_reject_float_and_bool(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    path = write_detection_config(tmp_path, patch)
    with pytest.raises(ConfigError, match=rf"{message} must be .*integer"):
        load_config(path)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"model": {"confidence_threshold": float("nan")}}, "confidence_threshold"),
        ({"model": {"confidence_threshold": float("inf")}}, "confidence_threshold"),
        ({"model": {"confidence_threshold": float("-inf")}}, "confidence_threshold"),
        ({"candidate_scoring": {"ambiguity_margin": float("nan")}}, "ambiguity_margin"),
        ({"candidate_scoring": {"ambiguity_margin": float("inf")}}, "ambiguity_margin"),
        ({"tracking": {"max_result_age_ms": float("nan")}}, "max_result_age_ms"),
        ({"tracking": {"max_result_age_ms": float("inf")}}, "max_result_age_ms"),
    ],
)
def test_detection_ranges_reject_non_finite_values(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    path = write_detection_config(tmp_path, patch)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_default_detection_backend_is_classical() -> None:
    config = load_config("config/default.yaml")

    assert config.detection.backend == "classical"
    assert config.detection.normalization.clahe_clip_limit == 2.0
    assert config.detection.white_board.expected_aspect_ratio == pytest.approx(210 / 297)
    assert config.detection.rings.expected_radius_ratios == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert config.detection.tracking.predict_max_frames == 3
    assert config.detection.tracking.predict_max_ms == 150.0
    assert not any(
        "red" in item.name.lower()
        for item in dataclasses.fields(config.detection.rings)
    )


def test_rejects_invalid_prediction_limit(tmp_path: Path) -> None:
    path = write_config(
        tmp_path, {"detection": {"tracking": {"predict_max_frames": 0}}}
    )
    with pytest.raises(ConfigError, match="predict_max_frames"):
        load_config(path)


def test_rejects_non_monotonic_ring_ratios(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "detection": {
                "rings": {"expected_radius_ratios": [1, 2, 2, 4, 5]}
            }
        },
    )
    with pytest.raises(ConfigError, match="expected_radius_ratios"):
        load_config(path)


def test_ring_first_configuration_defaults_and_yaml_override(tmp_path: Path) -> None:
    defaults = load_config(write_config(tmp_path, {}))
    ring = defaults.detection.ring_first
    assert ring.allow_medium_acquisition is True
    assert ring.ring_only is False
    assert ring.immediate_strong_acquisition is True
    assert ring.medium_confirm_frames == 2
    assert ring.medium_common_center_score == pytest.approx(0.60)
    assert ring.medium_ratio_score == pytest.approx(0.65)
    assert ring.medium_coverage_score == pytest.approx(0.12)
    assert ring.roi_min_half_extent_px == pytest.approx(72.0)
    assert ring.roi_safety_factor == pytest.approx(1.30)
    assert ring.roi_full_frame_after_misses == 3

    path = write_detection_config(
        tmp_path,
        {
            "ring_first": {
                "ring_only": True,
                "medium_confirm_frames": 3,
                "medium_common_center_score": 0.68,
                "roi_safety_factor": 1.4,
            }
        },
    )
    config = load_config(path)
    assert config.detection.ring_first.ring_only is True
    assert config.detection.ring_first.medium_confirm_frames == 3
    assert config.detection.ring_first.medium_common_center_score == pytest.approx(0.68)
    assert config.detection.ring_first.roi_safety_factor == pytest.approx(1.4)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"white_board_interval_frames": 0}, "white_board_interval_frames"),
        ({"medium_confirm_frames": 0}, "medium_confirm_frames"),
        ({"roi_min_half_extent_px": 0}, "roi_min_half_extent_px"),
        ({"roi_safety_factor": 0.9}, "roi_safety_factor"),
        ({"roi_miss_expand_px": -1}, "roi_miss_expand_px"),
        ({"roi_full_frame_after_misses": 0}, "roi_full_frame_after_misses"),
        ({"medium_min_arcs": 1}, "medium_min_arcs"),
        ({"strong_min_arcs": 1}, "strong_min_arcs"),
        ({"strong_common_center_score": 0.59}, "strong_common_center_score"),
        ({"strong_ratio_score": 0.64}, "strong_ratio_score"),
        ({"strong_coverage_score": 0.11}, "strong_coverage_score"),
    ],
)
def test_ring_first_configuration_rejects_invalid_values(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    path = write_detection_config(tmp_path, {"ring_first": patch})
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_jetson_local_uses_field_tuned_ring_first_settings() -> None:
    config = load_config("config/jetson-local.yaml")

    assert config.camera.exposure_us == 10_000
    assert config.camera.gain_db == pytest.approx(5.0)
    assert config.camera.acquisition_fps == pytest.approx(60.0)
    assert config.detection.normalization.clahe_clip_limit == pytest.approx(10.0)
    assert config.detection.white_board.min_white_occupancy == pytest.approx(0.5)
    assert config.detection.rings.ratio_tolerance == pytest.approx(0.18)
    assert config.detection.rings.min_arc_coverage == pytest.approx(0.18)
    assert config.detection.classical_scoring.tracking_threshold == pytest.approx(0.52)
    assert config.detection.classical_scoring.acquisition_threshold == pytest.approx(0.66)
    assert config.detection.ring_first.enabled is True
    assert config.detection.ring_first.allow_medium_acquisition is True
    assert config.detection.ring_first.ring_only is True
    assert config.detection.ring_first.immediate_strong_acquisition is True
    assert config.detection.ring_first.medium_confirm_frames == 2
    assert config.detection.ring_first.medium_ratio_score == pytest.approx(0.65)
