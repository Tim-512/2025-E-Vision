from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import get_type_hints

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import pytest

from ev_vision.config import CameraConfig
from ev_vision.models import BoardObservation, Frame
from ev_vision.tuning import CameraIdentity
from ev_vision.tuning.models import (
    CaptureSnapshot,
    DetectionSnapshot,
    EditableCameraParameters,
    ImageDiagnostics,
    OverlayOptions,
    ParameterBounds,
    RuntimeSnapshot,
)


def test_tuning_dependencies_and_static_package_data_are_declared() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    optional = project["project"]["optional-dependencies"]
    assert "httpx>=0.27" in optional["dev"]
    assert "httpx2>=2.0" in optional["dev"]
    assert "tomli>=2.0; python_version < '3.11'" in optional["dev"]
    assert optional["tuning"] == ["fastapi>=0.110", "uvicorn>=0.27"]
    assert project["tool"]["setuptools"]["package-data"]["ev_vision.web"] == [
        "static/*"
    ]
    assert project["project"]['scripts']["ev-vision"] == "ev_vision.cli:main"
    assert project["project"]['scripts']["ev-camera-tuning"] == (
        "ev_vision.web.camera_tuning_server:main"
    )


def test_editable_parameters_preserve_fixed_camera_format() -> None:
    base = CameraConfig(
        width=640,
        height=480,
        pixel_format="Mono8",
        acquisition_fps=60,
        exposure_us=500,
        gain_db=2.0,
        auto_exposure=False,
        auto_gain=False,
        auto_white_balance=False,
        buffer_size=5,
    )
    editable = EditableCameraParameters.from_camera_config(base)
    candidate = replace(
        editable,
        exposure_us=1250.0,
        gain_db=7.5,
        acquisition_fps=90.0,
        auto_white_balance=True,
    ).to_camera_config(base)

    assert candidate.width == base.width
    assert candidate.height == base.height
    assert candidate.pixel_format == base.pixel_format
    assert candidate.buffer_size == base.buffer_size
    assert candidate.exposure_us == 1250.0
    assert candidate.gain_db == 7.5
    assert candidate.acquisition_fps == 90.0
    assert candidate.auto_white_balance is True
    assert editable.to_dict() == {
        "exposure_us": 500.0,
        "gain_db": 2.0,
        "acquisition_fps": 60.0,
        "auto_exposure": False,
        "auto_gain": False,
        "auto_white_balance": False,
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("exposure_us", float("nan")),
        ("gain_db", float("inf")),
        ("acquisition_fps", float("-inf")),
        ("exposure_us", 19.0),
        ("exposure_us", 1_000_001.0),
        ("gain_db", -0.1),
        ("gain_db", 24.1),
        ("acquisition_fps", 0.9),
        ("acquisition_fps", 120.1),
    ],
)
def test_parameter_bounds_reject_non_finite_and_out_of_range_values(
    field_name: str, value: float
) -> None:
    candidate = replace(
        EditableCameraParameters.from_camera_config(CameraConfig()),
        **{field_name: value},
    )

    with pytest.raises(ValueError, match=field_name):
        ParameterBounds().validate(candidate)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("exposure_us", "800"),
        ("gain_db", None),
        ("acquisition_fps", True),
    ],
)
def test_parameter_bounds_reject_invalid_numeric_runtime_types(
    field_name: str, value: object
) -> None:
    candidate = replace(
        EditableCameraParameters.from_camera_config(CameraConfig()),
        **{field_name: value},
    )

    with pytest.raises(ValueError, match=field_name):
        ParameterBounds().validate(candidate)


def test_parameter_bounds_reject_non_boolean_auto_fields() -> None:
    candidate = replace(
        EditableCameraParameters.from_camera_config(CameraConfig()),
        auto_exposure="false",
    )

    with pytest.raises(ValueError, match="auto_exposure"):
        ParameterBounds().validate(candidate)


def test_parameter_bounds_accept_inclusive_defaults_and_integer_values() -> None:
    bounds = ParameterBounds()
    low = EditableCameraParameters(20, 0, 1, False, False, False)
    high = EditableCameraParameters(1_000_000, 24, 120, True, True, True)

    assert bounds.validate(low) is low
    assert bounds.validate(high) is high


def test_snapshot_models_are_frozen_and_reference_exact_domain_types() -> None:
    model_types = (
        EditableCameraParameters,
        ParameterBounds,
        CameraIdentity,
        ImageDiagnostics,
        DetectionSnapshot,
        OverlayOptions,
        CaptureSnapshot,
        RuntimeSnapshot,
    )
    assert all(model.__dataclass_params__.frozen for model in model_types)

    capture_hints = get_type_hints(CaptureSnapshot)
    detection_hints = get_type_hints(DetectionSnapshot)
    assert capture_hints["frame"] is Frame
    assert capture_hints["camera_identity"] is CameraIdentity
    assert detection_hints["observation"] == BoardObservation | None

    options = OverlayOptions()
    with pytest.raises(FrozenInstanceError):
        options.show_corners = False

    identity = CameraIdentity(model="MV-CA013-21UC", serial="00G02809155")
    with pytest.raises(FrozenInstanceError):
        identity.serial = "changed"
