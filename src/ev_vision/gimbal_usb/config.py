from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import yaml


class GimbalUsbConfigError(ValueError):
    """Raised when the independent gimbal USB configuration is unsafe."""


@dataclass(frozen=True)
class GimbalUsbConfig:
    port: str
    baudrate: int = 115200
    output_hz: float = 50.0
    reconnect_interval_s: float = 1.0
    calibration_path: Path = Path("config/camera_calibration.yaml")
    max_calibration_rms_px: float = 0.5
    max_result_age_ms: float = 60.0
    predicted_control_max_frames: int = 3
    predicted_control_max_age_ms: float = 60.0
    predicted_max_angle_step_deg: float = 1.5
    yaw_sign: int = 1
    pitch_sign: int = 1

    def validate(self) -> None:
        if not isinstance(self.port, str) or not self.port.strip():
            raise GimbalUsbConfigError("port must be a non-empty string")

        if isinstance(self.calibration_path, str):
            calibration_path_text = self.calibration_path.strip()
        elif isinstance(self.calibration_path, Path):
            calibration_path_text = str(self.calibration_path).strip()
        else:
            calibration_path_text = ""
        if not calibration_path_text or calibration_path_text == ".":
            raise GimbalUsbConfigError("calibration_path must be a non-empty path")

        integer_fields = {
            "baudrate": self.baudrate,
            "predicted_control_max_frames": self.predicted_control_max_frames,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise GimbalUsbConfigError(f"{name} must be an integer")
        if self.baudrate <= 0:
            raise GimbalUsbConfigError("baudrate must be positive")
        if self.predicted_control_max_frames < 1:
            raise GimbalUsbConfigError(
                "predicted_control_max_frames must be at least one"
            )

        positive_finite_fields = (
            "output_hz",
            "reconnect_interval_s",
            "max_calibration_rms_px",
            "max_result_age_ms",
            "predicted_control_max_age_ms",
            "predicted_max_angle_step_deg",
        )
        for name in positive_finite_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise GimbalUsbConfigError(f"{name} must be positive and finite")

        for name in ("yaw_sign", "pitch_sign"):
            value = getattr(self, name)
            if isinstance(value, bool) or value not in (-1, 1):
                raise GimbalUsbConfigError(f"{name} must be -1 or 1")


def load_gimbal_usb_config(path: str | Path) -> GimbalUsbConfig:
    source = Path(path)
    try:
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise GimbalUsbConfigError(
            f"unable to load gimbal USB config {source}: {exc}"
        ) from exc

    if not isinstance(values, dict):
        raise GimbalUsbConfigError("gimbal USB YAML must contain a mapping")

    allowed = {field.name for field in dataclasses.fields(GimbalUsbConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise GimbalUsbConfigError(
            f"unknown gimbal USB config keys: {', '.join(unknown)}"
        )

    if "calibration_path" in values:
        raw_path = values["calibration_path"]
        if not isinstance(raw_path, (str, Path)):
            raise GimbalUsbConfigError("calibration_path must be a path string")
        values["calibration_path"] = Path(raw_path)

    try:
        config = GimbalUsbConfig(**values)
    except TypeError as exc:
        raise GimbalUsbConfigError(str(exc)) from exc
    config.validate()
    return config
