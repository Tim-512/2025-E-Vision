from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ev_vision.camera.hikrobot import HikrobotCamera, MvsApi, create_native_api
from ev_vision.config import AppConfig, BoardConfig, CameraConfig, DetectionConfig, load_config
from ev_vision.detection.classical_board import ClassicalBoardDetector
from ev_vision.detection.hybrid_board import DetectionBackendSelection, HybridBoardDetector
from ev_vision.detection.yolo_board import InferencePort, UltralyticsBackend, YoloBoardDetector
from ev_vision.tracking.board_tracker import BoardTracker
from ev_vision.tuning.models import (
    CameraIdentity,
    EditableCameraParameters,
    ParameterBounds,
)
from ev_vision.tuning.service import CameraTuningService

FIXED_FORMAT = (1280, 1024, "BayerRG8", 2)
DEFAULT_SERIAL = "00G02809155"
CAMERA_MODEL = "MV-CA013-21UC"


@dataclass(frozen=True)
class CameraRuntime:
    config: AppConfig
    service: CameraTuningService


def require_fixed_format(camera: CameraConfig) -> None:
    actual = (camera.width, camera.height, camera.pixel_format, camera.buffer_size)
    if actual != FIXED_FORMAT:
        raise ValueError(
            "camera runtime requires fixed format 1280x1024, BayerRG8, buffer 2; "
            f"got {camera.width}x{camera.height}, {camera.pixel_format}, buffer {camera.buffer_size}"
        )


def build_detection_backend(
    config: DetectionConfig,
    *,
    backend_factory: Callable[..., InferencePort] = UltralyticsBackend,
) -> DetectionBackendSelection:
    errors: list[str] = []
    for configured_path in (config.model.path, config.model.fallback_path):
        path = Path(configured_path)
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            backend = backend_factory(
                path,
                input_size=(config.model.input_width, config.model.input_height),
                device=config.model.device,
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        artifact_kind = getattr(backend, "artifact_kind", path.suffix.lower().lstrip("."))
        return DetectionBackendSelection(backend, path, str(artifact_kind), ())
    return DetectionBackendSelection(None, None, "classical-diagnostic", tuple(errors))


def build_detector(
    config: DetectionConfig,
    *,
    board: BoardConfig | None = None,
    backend_factory: Callable[..., InferencePort] = UltralyticsBackend,
) -> ClassicalBoardDetector | HybridBoardDetector:
    if config.backend == "classical":
        return ClassicalBoardDetector(config)
    if config.backend != "hybrid":
        raise ValueError(f"unsupported detection backend: {config.backend}")

    def model_loader(candidate: DetectionConfig) -> DetectionBackendSelection:
        return build_detection_backend(candidate, backend_factory=backend_factory)

    selection = model_loader(config)
    model = None
    if selection.backend is not None:
        model = YoloBoardDetector(
            selection.backend,
            confidence_threshold=config.model.confidence_threshold,
            max_candidates=config.model.max_candidates,
        )
    return HybridBoardDetector(
        model=model,
        config=config,
        tracker=BoardTracker(config.tracking),
        board=board,
        model_state="READY" if model is not None else "UNAVAILABLE",
        model_backend=selection.artifact_kind,
        model_path=str(selection.path) if selection.path is not None else None,
        model_errors=selection.errors,
        model_loader=model_loader,
    )


def build_camera_runtime(
    *,
    config_path: str | Path,
    serial: str = DEFAULT_SERIAL,
    read_timeout_ms: int = 100,
    detection_fps: float = 30.0,
    diagnostics_fps: float | None = None,
    shutdown_timeout_s: float = 2.0,
    native_api_factory: Callable[[], MvsApi] = create_native_api,
    backend_factory: Callable[..., InferencePort] = UltralyticsBackend,
) -> CameraRuntime:
    config = load_config(config_path)
    require_fixed_format(config.camera)
    bounds = ParameterBounds()
    bounds.validate(EditableCameraParameters.from_camera_config(config.camera))

    native_api = native_api_factory()
    detector = build_detector(
        config.detection,
        board=config.board,
        backend_factory=backend_factory,
    )

    def camera_factory(camera_config: CameraConfig) -> HikrobotCamera:
        return HikrobotCamera(native_api, camera_config, serial_number=serial)

    service = CameraTuningService(
        camera_factory=camera_factory,
        base_config=config.camera,
        detector=detector,
        detection_config=config.detection,
        bounds=bounds,
        camera_identity=CameraIdentity(model=CAMERA_MODEL, serial=serial),
        read_timeout_ms=read_timeout_ms,
        diagnostics_fps=diagnostics_fps,
        detection_fps=detection_fps,
        shutdown_timeout_s=shutdown_timeout_s,
        detection_history_size=1,
    )
    return CameraRuntime(config=config, service=service)

