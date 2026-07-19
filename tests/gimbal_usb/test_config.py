from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import yaml

from ev_vision.config import load_config
from ev_vision.gimbal_usb.config import (
    GimbalUsbConfig,
    GimbalUsbConfigError,
    load_gimbal_usb_config,
)


APPROVED_VALUES = {
    "port": "/dev/serial/by-id/test",
    "baudrate": 115200,
    "output_hz": 50.0,
    "reconnect_interval_s": 1.0,
    "calibration_path": "config/camera_calibration.yaml",
    "max_calibration_rms_px": 0.5,
    "max_result_age_ms": 60.0,
    "predicted_control_max_frames": 3,
    "predicted_control_max_age_ms": 60.0,
    "predicted_max_angle_step_deg": 1.5,
    "yaw_sign": 1,
    "pitch_sign": 1,
}


def write_yaml(path: Path, values: object) -> Path:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_loads_approved_gimbal_values(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "gimbal.yaml", APPROVED_VALUES)

    value = load_gimbal_usb_config(path)

    assert value == GimbalUsbConfig(
        port="/dev/serial/by-id/test",
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


def test_minimal_yaml_uses_safe_defaults(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "gimbal.yaml", {"port": "/dev/test"})

    value = load_gimbal_usb_config(path)

    assert value.baudrate == 115200
    assert value.output_hz == 50.0
    assert value.reconnect_interval_s == 1.0
    assert value.calibration_path == Path("config/camera_calibration.yaml")
    assert value.max_calibration_rms_px == 0.5
    assert value.max_result_age_ms == 60.0
    assert value.predicted_control_max_frames == 3
    assert value.predicted_control_max_age_ms == 60.0
    assert value.predicted_max_angle_step_deg == 1.5
    assert value.yaw_sign == 1
    assert value.pitch_sign == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", ""),
        ("port", "   "),
        ("port", None),
        ("baudrate", True),
        ("baudrate", 115200.0),
        ("baudrate", 0),
        ("baudrate", -1),
        ("predicted_control_max_frames", True),
        ("predicted_control_max_frames", 3.0),
        ("predicted_control_max_frames", 0),
        ("predicted_control_max_frames", -1),
        ("output_hz", True),
        ("output_hz", 0),
        ("output_hz", -1),
        ("output_hz", math.nan),
        ("output_hz", math.inf),
        ("reconnect_interval_s", 0),
        ("reconnect_interval_s", math.nan),
        ("max_calibration_rms_px", 0),
        ("max_calibration_rms_px", math.inf),
        ("max_result_age_ms", 0),
        ("max_result_age_ms", math.nan),
        ("predicted_control_max_age_ms", 0),
        ("predicted_control_max_age_ms", -math.inf),
        ("predicted_max_angle_step_deg", 0),
        ("predicted_max_angle_step_deg", math.nan),
        ("yaw_sign", 0),
        ("yaw_sign", 2),
        ("yaw_sign", True),
        ("pitch_sign", 0),
        ("pitch_sign", -2),
        ("pitch_sign", False),
    ],
)
def test_rejects_unsafe_values(field: str, value: object) -> None:
    values = dataclasses.asdict(GimbalUsbConfig(port="/dev/test"))
    values[field] = value

    with pytest.raises(GimbalUsbConfigError, match=field):
        GimbalUsbConfig(**values).validate()


def test_rejects_invalid_calibration_path() -> None:
    values = dataclasses.asdict(GimbalUsbConfig(port="/dev/test"))
    values["calibration_path"] = ""

    with pytest.raises(GimbalUsbConfigError, match="calibration_path"):
        GimbalUsbConfig(**values).validate()


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "gimbal.yaml",
        {"port": "/dev/test", "extra": "unsafe"},
    )

    with pytest.raises(GimbalUsbConfigError, match="unknown.*extra"):
        load_gimbal_usb_config(path)


@pytest.mark.parametrize("document", [None, [], "port: /dev/test"])
def test_yaml_root_must_be_mapping(tmp_path: Path, document: object) -> None:
    path = write_yaml(tmp_path / "gimbal.yaml", document)

    with pytest.raises(GimbalUsbConfigError, match="mapping"):
        load_gimbal_usb_config(path)


def test_deployment_gimbal_profile_has_exact_approved_values() -> None:
    value = load_gimbal_usb_config("config/gimbal_usb.yaml")

    assert dataclasses.asdict(value) == {
        **APPROVED_VALUES,
        "port": "/dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00",
        "calibration_path": Path("config/camera_calibration.yaml"),
    }


def test_jetson_profile_has_tuned_camera_and_complete_detection_mapping() -> None:
    default_raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    jetson_raw = yaml.safe_load(Path("config/jetson-local.yaml").read_text(encoding="utf-8"))

    expected = dict(default_raw)
    expected["camera"] = {
        **default_raw["camera"],
        "acquisition_fps": 50,
        "exposure_us": 15000,
        "gain_db": 14.0,
    }
    assert jetson_raw == expected

    config = load_config("config/jetson-local.yaml")
    assert config.camera.exposure_us == 15000
    assert config.camera.gain_db == 14.0
    assert config.camera.acquisition_fps == 50
    assert config.detection.backend == "classical"
    assert config.detection.tracking.predict_max_ms == 150.0
