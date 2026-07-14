from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from ev_vision.config import CameraConfig
from ev_vision.models import BoardObservation, Frame


@dataclass(frozen=True)
class EditableCameraParameters:
    """Camera controls that may change without editing the fixed format."""

    exposure_us: float
    gain_db: float
    acquisition_fps: float
    auto_exposure: bool
    auto_gain: bool
    auto_white_balance: bool

    @classmethod
    def from_camera_config(cls, config: CameraConfig) -> EditableCameraParameters:
        return cls(
            exposure_us=float(config.exposure_us),
            gain_db=float(config.gain_db),
            acquisition_fps=float(config.acquisition_fps),
            auto_exposure=config.auto_exposure,
            auto_gain=config.auto_gain,
            auto_white_balance=config.auto_white_balance,
        )

    def to_camera_config(self, base: CameraConfig) -> CameraConfig:
        """Apply editable values while preserving the base acquisition format."""
        return replace(
            base,
            exposure_us=self.exposure_us,
            gain_db=self.gain_db,
            acquisition_fps=self.acquisition_fps,
            auto_exposure=self.auto_exposure,
            auto_gain=self.auto_gain,
            auto_white_balance=self.auto_white_balance,
        )

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterBounds:
    exposure_min_us: float = 20.0
    exposure_max_us: float = 1_000_000.0
    gain_min_db: float = 0.0
    gain_max_db: float = 24.0
    acquisition_fps_min: float = 1.0
    acquisition_fps_max: float = 120.0

    def validate(self, parameters: EditableCameraParameters) -> EditableCameraParameters:
        ranges = (
            (
                "exposure_us",
                parameters.exposure_us,
                self.exposure_min_us,
                self.exposure_max_us,
            ),
            ("gain_db", parameters.gain_db, self.gain_min_db, self.gain_max_db),
            (
                "acquisition_fps",
                parameters.acquisition_fps,
                self.acquisition_fps_min,
                self.acquisition_fps_max,
            ),
        )
        for name, value, minimum, maximum in ranges:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
        return parameters


@dataclass(frozen=True)
class ImageDiagnostics:
    source_sequence: int
    computed_ns: int
    gray_histogram: tuple[int, ...]
    blue_histogram: tuple[int, ...]
    green_histogram: tuple[int, ...]
    red_histogram: tuple[int, ...]
    dark_percent: float
    bright_percent: float
    focus_score: float
    roi_px: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionSnapshot:
    enabled: bool
    detected: bool
    source_sequence: int | None = None
    observation: BoardObservation | None = None
    result_age_ms: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class OverlayOptions:
    enabled: bool = True
    show_board_outline: bool = True
    show_corners: bool = True
    show_center: bool = True
    show_crosshair: bool = True
    show_detection_text: bool = True
    show_center_roi: bool = True


@dataclass(frozen=True)
class RuntimeSnapshot:
    state: str
    acquisition_fps: float = 0.0
    preview_fps: float = 0.0
    detection_fps: float = 0.0
    diagnostics_fps: float = 0.0
    frame_count: int = 0
    timeout_count: int = 0
    sequence_gap_count: int = 0
    frame_age_ms: float | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class CaptureSnapshot:
    frame: Frame
    parameters: EditableCameraParameters
    runtime: RuntimeSnapshot
    diagnostics: ImageDiagnostics | None
    detection: DetectionSnapshot
    overlay_options: OverlayOptions
    camera_config: CameraConfig
    camera_identity: dict[str, Any] | None = None
