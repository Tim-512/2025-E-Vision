from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 1024
    pixel_format: str = "BayerRG8"
    acquisition_fps: int = 120
    exposure_us: int = 800
    gain_db: float = 6.0
    auto_exposure: bool = False
    auto_gain: bool = False
    auto_white_balance: bool = False
    buffer_size: int = 2


@dataclass(frozen=True)
class BoardConfig:
    width_cm: float = 21.0
    height_cm: float = 29.7
    rectified_px_per_cm: float = 40.0


@dataclass(frozen=True)
class CircleConfig:
    radius_cm: float = 6.0
    start_phase_deg: float = 0.0
    phase_direction: int = 1
    progress_timeout_ms: int = 300
    max_phase_correction_deg_s: float = 45.0


@dataclass(frozen=True)
class ControlConfig:
    command_hz: int = 100
    estimated_latency_ms: float = 25.0
    source_timeout_ms: int = 100
    kp_yaw: float = 1.0
    kd_yaw: float = 0.0
    kp_pitch: float = 1.0
    kd_pitch: float = 0.0
    derivative_alpha: float = 0.25
    deadband_enter_mdeg: int = 50
    deadband_exit_mdeg: int = 100
    max_yaw_rate_deg_s: float | None = None
    max_pitch_rate_deg_s: float | None = None
    max_yaw_accel_deg_s2: float | None = None
    max_pitch_accel_deg_s2: float | None = None
    yaw_sign: int = 1
    pitch_sign: int = 1
    swap_axes: bool = False


@dataclass(frozen=True)
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 921600
    feedback_timeout_ms: int = 300


@dataclass(frozen=True)
class LaserConfig:
    backend: str = "mock"
    active_high: bool = True
    gpio_line: int | None = None


@dataclass(frozen=True)
class ModelDetectionConfig:
    path: str = "models/target-board.engine"
    fallback_path: str = "models/target-board.onnx"
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.45
    max_candidates: int = 3
    device: int = 0


@dataclass(frozen=True)
class RoiGeometryConfig:
    padding_fraction: float = 0.08
    canny_low: int = 60
    canny_high: int = 180
    min_edge_support: float = 0.45
    min_geometry_score: float = 0.55
    expected_aspect_ratio: float = 0.707
    aspect_ratio_tolerance: float = 0.35
    minimum_side_px: float = 40.0
    minimum_area_fraction: float = 0.25
    maximum_area_fraction: float = 1.15


@dataclass(frozen=True)
class CandidateScoringConfig:
    model_weight: float = 0.45
    geometry_weight: float = 0.30
    structure_weight: float = 0.15
    temporal_weight: float = 0.10
    ambiguity_margin: float = 0.08

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (
            self.model_weight,
            self.geometry_weight,
            self.structure_weight,
            self.temporal_weight,
        )


@dataclass(frozen=True)
class ImageNormalizationConfig:
    gaussian_kernel: int = 3
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    illumination_kernel: int = 81
    white_percentile: float = 72.0
    white_local_offset: float = 10.0
    saturation_threshold: int = 250
    canny_low: int = 40
    canny_high: int = 120


@dataclass(frozen=True)
class WhiteBoardConfig:
    expected_aspect_ratio: float = 210.0 / 297.0
    aspect_ratio_tolerance: float = 0.24
    min_area_fraction: float = 0.015
    max_area_fraction: float = 0.92
    min_white_occupancy: float = 0.58
    max_texture_std: float = 58.0
    min_convexity: float = 0.90
    min_side_px: float = 45.0
    border_band_fraction: float = 0.045


@dataclass(frozen=True)
class RingGeometryConfig:
    expected_radius_ratios: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    ratio_tolerance: float = 0.18
    center_tolerance_fraction: float = 0.08
    min_arc_coverage: float = 0.18
    min_multiple_arcs: int = 2
    max_single_arc_frames: int = 2
    saturation_mask_radius_px: int = 12


@dataclass(frozen=True)
class RingFirstConfig:
    enabled: bool = True
    allow_medium_acquisition: bool = True
    white_board_interval_frames: int = 6
    strong_min_arcs: int = 3
    strong_common_center_score: float = 0.75
    strong_ratio_score: float = 0.80
    strong_coverage_score: float = 0.20
    medium_min_arcs: int = 2
    medium_common_center_score: float = 0.65
    medium_ratio_score: float = 0.70
    medium_coverage_score: float = 0.15


@dataclass(frozen=True)
class ClassicalScoringConfig:
    white_weight: float = 0.24
    geometry_weight: float = 0.22
    ring_weight: float = 0.30
    border_weight: float = 0.08
    temporal_weight: float = 0.16
    acquisition_threshold: float = 0.66
    tracking_threshold: float = 0.52
    ambiguity_margin: float = 0.08
    max_texture_penalty: float = 0.20

    @property
    def weights(self) -> tuple[float, float, float, float, float]:
        return (
            self.white_weight,
            self.geometry_weight,
            self.ring_weight,
            self.border_weight,
            self.temporal_weight,
        )


@dataclass(frozen=True)
class BoardTrackingConfig:
    confirm_frames: int = 3
    predict_frames: int = 2
    predict_max_frames: int = 3
    predict_max_ms: float = 150.0
    lost_frames: int = 4
    max_single_arc_frames: int = 2
    max_center_jump_px: float = 160.0
    max_scale_jump_fraction: float = 0.30
    max_velocity_px_s: float = 5000.0
    max_acceleration_px_s2: float = 30000.0
    max_result_age_ms: float = 100.0


@dataclass(frozen=True)
class DetectionConfig:
    backend: str = "classical"
    model: ModelDetectionConfig = field(default_factory=ModelDetectionConfig)
    roi_geometry: RoiGeometryConfig = field(default_factory=RoiGeometryConfig)
    candidate_scoring: CandidateScoringConfig = field(
        default_factory=CandidateScoringConfig
    )
    tracking: BoardTrackingConfig = field(default_factory=BoardTrackingConfig)
    normalization: ImageNormalizationConfig = field(
        default_factory=ImageNormalizationConfig
    )
    white_board: WhiteBoardConfig = field(default_factory=WhiteBoardConfig)
    rings: RingGeometryConfig = field(default_factory=RingGeometryConfig)
    ring_first: RingFirstConfig = field(default_factory=RingFirstConfig)
    classical_scoring: ClassicalScoringConfig = field(
        default_factory=ClassicalScoringConfig
    )


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    board: BoardConfig
    circle: CircleConfig
    control: ControlConfig
    serial: SerialConfig
    laser: LaserConfig
    detection: DetectionConfig = field(default_factory=DetectionConfig)


T = TypeVar("T")
_SECTIONS: dict[str, type[Any]] = {
    "camera": CameraConfig,
    "board": BoardConfig,
    "circle": CircleConfig,
    "control": ControlConfig,
    "serial": SerialConfig,
    "laser": LaserConfig,
}
_DETECTION_SECTIONS: dict[str, type[Any]] = {
    "model": ModelDetectionConfig,
    "roi_geometry": RoiGeometryConfig,
    "candidate_scoring": CandidateScoringConfig,
    "tracking": BoardTrackingConfig,
    "normalization": ImageNormalizationConfig,
    "white_board": WhiteBoardConfig,
    "rings": RingGeometryConfig,
    "ring_first": RingFirstConfig,
    "classical_scoring": ClassicalScoringConfig,
}


def _mapping(values: object, section: str) -> Mapping[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ConfigError(f"{section} configuration must be a mapping")
    return values


def _build(cls: type[T], values: Mapping[str, Any] | None, section: str) -> T:
    values = _mapping(values, section)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown configuration keys in {section}: {', '.join(unknown)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {section} configuration: {exc}") from exc


def _finite(name: str, value: float) -> None:
    try:
        finite = math.isfinite(value)
    except TypeError as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not finite:
        raise ConfigError(f"{name} must be a finite number")


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _positive_integer(name: str, value: object) -> int:
    integer = _integer(name, value)
    if integer <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return integer


def _non_negative_integer(name: str, value: object) -> int:
    integer = _integer(name, value)
    if integer < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return integer


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _non_negative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ConfigError(f"{name} must be non-negative")


def _unit_interval(name: str, value: float) -> None:
    _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be within [0, 1]")


def _validate_detection(cfg: DetectionConfig) -> None:
    if cfg.backend not in {"hybrid", "classical"}:
        raise ConfigError("detection.backend must be 'hybrid' or 'classical'")

    _positive_integer("detection.model.input_width", cfg.model.input_width)
    _positive_integer("detection.model.input_height", cfg.model.input_height)
    _unit_interval(
        "detection.model.confidence_threshold", cfg.model.confidence_threshold
    )
    _positive_integer("detection.model.max_candidates", cfg.model.max_candidates)
    _non_negative_integer("detection.model.device", cfg.model.device)

    _non_negative(
        "detection.roi_geometry.padding_fraction", cfg.roi_geometry.padding_fraction
    )
    _non_negative_integer(
        "detection.roi_geometry.canny_low", cfg.roi_geometry.canny_low
    )
    _positive_integer(
        "detection.roi_geometry.canny_high", cfg.roi_geometry.canny_high
    )
    if cfg.roi_geometry.canny_low >= cfg.roi_geometry.canny_high:
        raise ConfigError(
            "detection.roi_geometry.canny_low must be less than canny_high"
        )
    _unit_interval(
        "detection.roi_geometry.min_edge_support",
        cfg.roi_geometry.min_edge_support,
    )
    _unit_interval(
        "detection.roi_geometry.min_geometry_score",
        cfg.roi_geometry.min_geometry_score,
    )
    _positive(
        "detection.roi_geometry.expected_aspect_ratio",
        cfg.roi_geometry.expected_aspect_ratio,
    )
    _non_negative(
        "detection.roi_geometry.aspect_ratio_tolerance",
        cfg.roi_geometry.aspect_ratio_tolerance,
    )
    _positive(
        "detection.roi_geometry.minimum_side_px", cfg.roi_geometry.minimum_side_px
    )
    _positive(
        "detection.roi_geometry.minimum_area_fraction",
        cfg.roi_geometry.minimum_area_fraction,
    )
    _positive(
        "detection.roi_geometry.maximum_area_fraction",
        cfg.roi_geometry.maximum_area_fraction,
    )
    if (
        cfg.roi_geometry.minimum_area_fraction
        > cfg.roi_geometry.maximum_area_fraction
    ):
        raise ConfigError(
            "detection.roi_geometry.minimum_area_fraction must not exceed "
            "maximum_area_fraction"
        )

    for name, value in zip(
        (
            "model_weight",
            "geometry_weight",
            "structure_weight",
            "temporal_weight",
        ),
        cfg.candidate_scoring.weights,
        strict=True,
    ):
        _positive(f"detection.candidate_scoring.{name}", value)
    _non_negative(
        "detection.candidate_scoring.ambiguity_margin",
        cfg.candidate_scoring.ambiguity_margin,
    )

    _positive_integer("detection.tracking.confirm_frames", cfg.tracking.confirm_frames)
    _positive_integer("detection.tracking.predict_frames", cfg.tracking.predict_frames)
    _positive_integer(
        "detection.tracking.predict_max_frames", cfg.tracking.predict_max_frames
    )
    _positive("detection.tracking.predict_max_ms", cfg.tracking.predict_max_ms)
    _positive_integer("detection.tracking.lost_frames", cfg.tracking.lost_frames)
    _positive_integer(
        "detection.tracking.max_single_arc_frames",
        cfg.tracking.max_single_arc_frames,
    )
    if cfg.tracking.max_single_arc_frames > cfg.tracking.predict_max_frames:
        raise ConfigError(
            "detection.tracking.max_single_arc_frames must not exceed "
            "predict_max_frames"
        )
    _positive(
        "detection.tracking.max_center_jump_px", cfg.tracking.max_center_jump_px
    )
    _positive(
        "detection.tracking.max_scale_jump_fraction",
        cfg.tracking.max_scale_jump_fraction,
    )
    _positive(
        "detection.tracking.max_velocity_px_s", cfg.tracking.max_velocity_px_s
    )
    _positive(
        "detection.tracking.max_acceleration_px_s2",
        cfg.tracking.max_acceleration_px_s2,
    )
    _positive("detection.tracking.max_result_age_ms", cfg.tracking.max_result_age_ms)

    for name in ("gaussian_kernel", "illumination_kernel"):
        value = getattr(cfg.normalization, name)
        _positive_integer(f"detection.normalization.{name}", value)
        if value % 2 == 0:
            raise ConfigError(f"detection.normalization.{name} must be odd")
    _positive_integer(
        "detection.normalization.clahe_grid_size",
        cfg.normalization.clahe_grid_size,
    )
    _positive(
        "detection.normalization.clahe_clip_limit",
        cfg.normalization.clahe_clip_limit,
    )
    _unit_interval(
        "detection.normalization.white_percentile",
        cfg.normalization.white_percentile / 100.0,
    )
    _non_negative(
        "detection.normalization.white_local_offset",
        cfg.normalization.white_local_offset,
    )
    for name in ("saturation_threshold", "canny_low", "canny_high"):
        value = _non_negative_integer(
            f"detection.normalization.{name}", getattr(cfg.normalization, name)
        )
        if value > 255:
            raise ConfigError(f"detection.normalization.{name} must be within [0, 255]")
    if cfg.normalization.canny_low >= cfg.normalization.canny_high:
        raise ConfigError(
            "detection.normalization.canny_low must be less than canny_high"
        )

    _positive(
        "detection.white_board.expected_aspect_ratio",
        cfg.white_board.expected_aspect_ratio,
    )
    _non_negative(
        "detection.white_board.aspect_ratio_tolerance",
        cfg.white_board.aspect_ratio_tolerance,
    )
    _positive(
        "detection.white_board.min_area_fraction",
        cfg.white_board.min_area_fraction,
    )
    _positive(
        "detection.white_board.max_area_fraction",
        cfg.white_board.max_area_fraction,
    )
    if cfg.white_board.min_area_fraction >= cfg.white_board.max_area_fraction:
        raise ConfigError(
            "detection.white_board.min_area_fraction must be less than "
            "max_area_fraction"
        )
    _unit_interval(
        "detection.white_board.min_white_occupancy",
        cfg.white_board.min_white_occupancy,
    )
    _positive(
        "detection.white_board.max_texture_std", cfg.white_board.max_texture_std
    )
    _unit_interval(
        "detection.white_board.min_convexity", cfg.white_board.min_convexity
    )
    _positive("detection.white_board.min_side_px", cfg.white_board.min_side_px)
    _positive(
        "detection.white_board.border_band_fraction",
        cfg.white_board.border_band_fraction,
    )

    ratios = tuple(cfg.rings.expected_radius_ratios)
    if not ratios or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        for value in ratios
    ) or any(left >= right for left, right in zip(ratios, ratios[1:])):
        raise ConfigError(
            "detection.rings.expected_radius_ratios must be finite, positive, "
            "and strictly increasing"
        )
    _positive("detection.rings.ratio_tolerance", cfg.rings.ratio_tolerance)
    _positive(
        "detection.rings.center_tolerance_fraction",
        cfg.rings.center_tolerance_fraction,
    )
    _unit_interval("detection.rings.min_arc_coverage", cfg.rings.min_arc_coverage)
    _positive_integer(
        "detection.rings.min_multiple_arcs", cfg.rings.min_multiple_arcs
    )
    _positive_integer(
        "detection.rings.max_single_arc_frames",
        cfg.rings.max_single_arc_frames,
    )
    _non_negative_integer(
        "detection.rings.saturation_mask_radius_px",
        cfg.rings.saturation_mask_radius_px,
    )

    for name, value in zip(
        ("white_weight", "geometry_weight", "ring_weight", "border_weight", "temporal_weight"),
        cfg.classical_scoring.weights,
        strict=True,
    ):
        _non_negative(f"detection.classical_scoring.{name}", value)
    ring_first = cfg.ring_first
    _positive_integer(
        "detection.ring_first.white_board_interval_frames",
        ring_first.white_board_interval_frames,
    )
    _positive_integer("detection.ring_first.strong_min_arcs", ring_first.strong_min_arcs)
    _positive_integer("detection.ring_first.medium_min_arcs", ring_first.medium_min_arcs)
    if ring_first.medium_min_arcs < 2:
        raise ConfigError("detection.ring_first.medium_min_arcs must be at least 2")
    if ring_first.strong_min_arcs < ring_first.medium_min_arcs:
        raise ConfigError(
            "detection.ring_first.strong_min_arcs must be at least medium_min_arcs"
        )
    for name in (
        "strong_common_center_score",
        "strong_ratio_score",
        "strong_coverage_score",
        "medium_common_center_score",
        "medium_ratio_score",
        "medium_coverage_score",
    ):
        _unit_interval(f"detection.ring_first.{name}", getattr(ring_first, name))
    for strong_name, medium_name in (
        ("strong_common_center_score", "medium_common_center_score"),
        ("strong_ratio_score", "medium_ratio_score"),
        ("strong_coverage_score", "medium_coverage_score"),
    ):
        if getattr(ring_first, strong_name) < getattr(ring_first, medium_name):
            raise ConfigError(
                f"detection.ring_first.{strong_name} must be at least {medium_name}"
            )

    if abs(sum(cfg.classical_scoring.weights) - 1.0) > 1e-6:
        raise ConfigError(
            "detection.classical_scoring weights must sum to 1.0"
        )
    for name in (
        "acquisition_threshold",
        "tracking_threshold",
        "ambiguity_margin",
        "max_texture_penalty",
    ):
        _unit_interval(
            f"detection.classical_scoring.{name}",
            getattr(cfg.classical_scoring, name),
        )


def _build_detection(values: Mapping[str, Any] | None) -> DetectionConfig:
    values = _mapping(values, "detection")
    allowed = {"backend", *_DETECTION_SECTIONS}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown configuration keys in detection: {', '.join(unknown)}"
        )

    nested = {
        name: _build(cls, values.get(name), f"detection.{name}")
        for name, cls in _DETECTION_SECTIONS.items()
    }
    rings = nested["rings"]
    assert isinstance(rings, RingGeometryConfig)
    nested["rings"] = replace(
        rings, expected_radius_ratios=tuple(rings.expected_radius_ratios)
    )

    scoring = nested["candidate_scoring"]
    assert isinstance(scoring, CandidateScoringConfig)
    for name, value in zip(
        (
            "model_weight",
            "geometry_weight",
            "structure_weight",
            "temporal_weight",
        ),
        scoring.weights,
        strict=True,
    ):
        _positive(f"detection.candidate_scoring.{name}", value)
    weight_sum = sum(scoring.weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ConfigError("detection candidate scoring weight sum must be positive")
    nested["candidate_scoring"] = CandidateScoringConfig(
        model_weight=scoring.model_weight / weight_sum,
        geometry_weight=scoring.geometry_weight / weight_sum,
        structure_weight=scoring.structure_weight / weight_sum,
        temporal_weight=scoring.temporal_weight / weight_sum,
        ambiguity_margin=scoring.ambiguity_margin,
    )

    cfg = DetectionConfig(backend=values.get("backend", "classical"), **nested)
    _validate_detection(cfg)
    return cfg


def _validate(cfg: AppConfig, hardware_required: bool) -> None:
    _positive("camera.width", cfg.camera.width)
    _positive("camera.height", cfg.camera.height)
    _positive("camera.acquisition_fps", cfg.camera.acquisition_fps)
    _positive("board.width_cm", cfg.board.width_cm)
    _positive("board.height_cm", cfg.board.height_cm)
    _positive("board.rectified_px_per_cm", cfg.board.rectified_px_per_cm)
    _positive("circle.radius_cm", cfg.circle.radius_cm)
    _positive("control.command_hz", cfg.control.command_hz)
    _positive("serial.baudrate", cfg.serial.baudrate)
    if cfg.circle.phase_direction not in (-1, 1):
        raise ConfigError("circle.phase_direction must be -1 or 1")
    if cfg.control.yaw_sign not in (-1, 1) or cfg.control.pitch_sign not in (-1, 1):
        raise ConfigError("control axis signs must be -1 or 1")
    if not 0.0 <= cfg.control.derivative_alpha <= 1.0:
        raise ConfigError("control.derivative_alpha must be within [0, 1]")
    _validate_detection(cfg.detection)
    if hardware_required:
        required = (
            "max_yaw_rate_deg_s",
            "max_pitch_rate_deg_s",
            "max_yaw_accel_deg_s2",
            "max_pitch_accel_deg_s2",
        )
        for name in required:
            if getattr(cfg.control, name) is None:
                raise ConfigError(f"control.{name} is required in hardware mode")
        if cfg.laser.backend == "gpio" and cfg.laser.gpio_line is None:
            raise ConfigError("laser.gpio_line is required for the gpio backend")


def load_config(path: str | Path, hardware_required: bool = False) -> AppConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load configuration {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    unknown = sorted(set(raw) - {*_SECTIONS, "detection"})
    if unknown:
        raise ConfigError(f"unknown configuration keys: {', '.join(unknown)}")
    sections = {
        name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()
    }
    cfg = AppConfig(**sections, detection=_build_detection(raw.get("detection")))
    _validate(cfg, hardware_required)
    return cfg
