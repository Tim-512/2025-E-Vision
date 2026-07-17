from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, field_validator

from ev_vision.config import (
    BoardTrackingConfig,
    CandidateScoringConfig,
    ClassicalScoringConfig,
    ConfigError,
    DetectionConfig,
    ImageNormalizationConfig,
    ModelDetectionConfig,
    RingGeometryConfig,
    RoiGeometryConfig,
    WhiteBoardConfig,
    _validate_detection,
)
from ev_vision.tuning.diagnostics import render_overlay as _render_base_overlay
from ev_vision.tuning.models import (
    CameraIdentity,
    EditableCameraParameters,
    OverlayOptions,
    ParameterBounds,
)
from ev_vision.tuning.service import ParameterApplyError, StaleDetectionFrameError


_CANDIDATE_COLOR = (0, 255, 255)
_REJECTED_COLOR = (0, 0, 255)
_CONFIRMED_COLOR = (0, 255, 0)


def _render_hybrid_overlay(image: np.ndarray, *, source_sequence: int, detection: Any, options: OverlayOptions) -> np.ndarray:
    base_detection = detection
    if hasattr(detection, "target_valid") and not bool(detection.target_valid) and is_dataclass(detection):
        base_detection = replace(detection, observation=None)
    output = _render_base_overlay(image, source_sequence=source_sequence, detection=base_detection, options=options)
    if not options.enabled or getattr(detection, "source_sequence", None) != source_sequence:
        return output
    height, width = output.shape[:2]
    for index, candidate in enumerate(getattr(detection, "candidates", ()) or (), 1):
        box = np.asarray(getattr(candidate, "xyxy_px", ()), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all(): continue
        x1, y1, x2, y2 = box
        p1 = (int(np.clip(round(x1), 0, width-1)), int(np.clip(round(y1), 0, height-1)))
        p2 = (int(np.clip(round(x2), 0, width-1)), int(np.clip(round(y2), 0, height-1)))
        accepted = bool(getattr(candidate, "accepted", False))
        color = _CANDIDATE_COLOR if accepted else _REJECTED_COLOR
        cv2.rectangle(output, p1, p2, color, 2, cv2.LINE_8)
        failure = getattr(candidate, "failure_reason", None) or ("ACCEPTED" if accepted else "REJECTED")
        cv2.putText(output, f"C{index} score={float(getattr(candidate, 'combined_score', 0.0)):.3f} {failure}", (p1[0], max(10, p1[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    corners = np.asarray(getattr(detection, "corners_px", ()) or (), dtype=float)
    if bool(getattr(detection, "target_valid", False)) and corners.shape == (4, 2) and np.isfinite(corners).all():
        corners[:, 0] = np.clip(corners[:, 0], 0, width-1); corners[:, 1] = np.clip(corners[:, 1], 0, height-1)
        cv2.polylines(output, [np.rint(corners).astype(np.int32)], True, _CONFIRMED_COLOR, 2, cv2.LINE_8)
    center = np.asarray(getattr(detection, "center_px", ()) or (), dtype=float)
    if bool(getattr(detection, "target_valid", False)) and center.shape == (2,) and np.isfinite(center).all():
        point = (int(np.clip(round(center[0]), 0, width-1)), int(np.clip(round(center[1]), 0, height-1)))
        cv2.line(output, (width//2, height//2), point, _CONFIRMED_COLOR, 1, cv2.LINE_AA)
        cv2.drawMarker(output, point, _CONFIRMED_COLOR, cv2.MARKER_CROSS, 11, 2, cv2.LINE_8)
    if options.show_detection_text:
        tracking = str(getattr(detection, "tracking_state", "SEARCHING")); valid = "valid" if getattr(detection, "target_valid", False) else "invalid"
        failure = str(getattr(detection, "failure_reason", None) or "NONE")
        cv2.putText(output, f"{tracking} {valid} score={float(getattr(detection, 'combined_score', 0.0)):.3f} failure={failure}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .48, _CONFIRMED_COLOR if valid == "valid" else _REJECTED_COLOR, 1, cv2.LINE_AA)
    return output


def render_overlay(image: np.ndarray, *, source_sequence: int, detection: Any, options: OverlayOptions) -> np.ndarray:
    return _render_hybrid_overlay(image, source_sequence=source_sequence, detection=detection, options=options)


_NUMERIC = StrictFloat | StrictInt
_DEFAULT_OVERLAY = OverlayOptions()
_FIXED_FORMAT = {
    "width": 1280,
    "height": 1024,
    "pixel_format": "BayerRG8",
    "buffer_size": 2,
}
_CLASSICAL_DEBUG_IMAGE_NAMES = frozenset({
    "normalized-gray",
    "white-mask",
    "edge-mask",
    "ring-arcs",
    "candidate-scores",
})
DEBUG_IMAGE_NAMES = frozenset({
    "model-candidates",
    "geometry-accepted",
    "geometry-rejected",
    "roi",
    "roi-edges",
    "roi-geometry",
    "final-overlay",
}) | _CLASSICAL_DEBUG_IMAGE_NAMES
_PLACEHOLDER_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Camera Tuning</title>
<link rel="stylesheet" href="/static/camera-tuning.css"></head>
<body><main><h1>Camera Tuning</h1><p>Dashboard assets will be installed by Task 6.</p></main>
<script src="/static/camera-tuning.js"></script></body></html>"""


class ParameterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exposure_us: _NUMERIC
    gain_db: _NUMERIC
    acquisition_fps: _NUMERIC
    auto_exposure: StrictBool
    auto_gain: StrictBool
    auto_white_balance: StrictBool

    @field_validator("exposure_us", "gain_db", "acquisition_fps")
    @classmethod
    def reject_non_finite(cls, value: int | float) -> float:
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("numeric camera parameters must be finite")
        return converted

    def domain(self) -> EditableCameraParameters:
        return EditableCameraParameters(
            exposure_us=float(self.exposure_us),
            gain_db=float(self.gain_db),
            acquisition_fps=float(self.acquisition_fps),
            auto_exposure=self.auto_exposure,
            auto_gain=self.auto_gain,
            auto_white_balance=self.auto_white_balance,
        )


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    parameters: ParameterRequest


class OverlayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlay: StrictBool = True
    show_board_outline: StrictBool = True
    show_corners: StrictBool = True
    show_center: StrictBool = True
    show_crosshair: StrictBool = True
    show_detection_text: StrictBool = True
    show_center_roi: StrictBool = True

    def domain(self) -> OverlayOptions:
        return OverlayOptions(
            enabled=self.overlay,
            show_board_outline=self.show_board_outline,
            show_corners=self.show_corners,
            show_center=self.show_center,
            show_crosshair=self.show_crosshair,
            show_detection_text=self.show_detection_text,
            show_center_roi=self.show_center_roi,
        )


class _FiniteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def reject_non_finite(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite(float(value)):
                raise ValueError("numeric detection parameters must be finite")
        return value


class DetectionModelRequest(_FiniteRequest):
    confidence_threshold: _NUMERIC
    max_candidates: StrictInt

    def domain(self, current: ModelDetectionConfig) -> ModelDetectionConfig:
        return ModelDetectionConfig(
            path=current.path,
            fallback_path=current.fallback_path,
            input_width=current.input_width,
            input_height=current.input_height,
            confidence_threshold=float(self.confidence_threshold),
            max_candidates=self.max_candidates,
            device=current.device,
        )


class DetectionRoiGeometryRequest(_FiniteRequest):
    padding_fraction: _NUMERIC
    canny_low: StrictInt
    canny_high: StrictInt
    min_edge_support: _NUMERIC
    min_geometry_score: _NUMERIC
    expected_aspect_ratio: _NUMERIC
    aspect_ratio_tolerance: _NUMERIC
    minimum_side_px: _NUMERIC
    minimum_area_fraction: _NUMERIC
    maximum_area_fraction: _NUMERIC

    def domain(self) -> RoiGeometryConfig:
        return RoiGeometryConfig(
            padding_fraction=float(self.padding_fraction),
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            min_edge_support=float(self.min_edge_support),
            min_geometry_score=float(self.min_geometry_score),
            expected_aspect_ratio=float(self.expected_aspect_ratio),
            aspect_ratio_tolerance=float(self.aspect_ratio_tolerance),
            minimum_side_px=float(self.minimum_side_px),
            minimum_area_fraction=float(self.minimum_area_fraction),
            maximum_area_fraction=float(self.maximum_area_fraction),
        )


class DetectionCandidateScoringRequest(_FiniteRequest):
    model_weight: _NUMERIC
    geometry_weight: _NUMERIC
    structure_weight: _NUMERIC
    temporal_weight: _NUMERIC
    ambiguity_margin: _NUMERIC

    def domain(self) -> CandidateScoringConfig:
        return CandidateScoringConfig(
            model_weight=float(self.model_weight),
            geometry_weight=float(self.geometry_weight),
            structure_weight=float(self.structure_weight),
            temporal_weight=float(self.temporal_weight),
            ambiguity_margin=float(self.ambiguity_margin),
        )


class DetectionTrackingRequest(_FiniteRequest):
    confirm_frames: StrictInt | None = None
    predict_frames: StrictInt | None = None
    predict_max_frames: StrictInt | None = None
    predict_max_ms: _NUMERIC | None = None
    lost_frames: StrictInt | None = None
    max_single_arc_frames: StrictInt | None = None
    max_center_jump_px: _NUMERIC | None = None
    max_scale_jump_fraction: _NUMERIC | None = None
    max_velocity_px_s: _NUMERIC | None = None
    max_acceleration_px_s2: _NUMERIC | None = None
    max_result_age_ms: _NUMERIC | None = None

    def domain(self, current: BoardTrackingConfig) -> BoardTrackingConfig:
        values = self.model_dump(exclude_none=True)
        return replace(current, **values)


class DetectionNormalizationRequest(_FiniteRequest):
    gaussian_kernel: StrictInt | None = None
    clahe_clip_limit: _NUMERIC | None = None
    clahe_grid_size: StrictInt | None = None
    illumination_kernel: StrictInt | None = None
    white_percentile: _NUMERIC | None = None
    white_local_offset: _NUMERIC | None = None
    saturation_threshold: StrictInt | None = None
    canny_low: StrictInt | None = None
    canny_high: StrictInt | None = None

    def domain(self, current: ImageNormalizationConfig) -> ImageNormalizationConfig:
        return replace(current, **self.model_dump(exclude_none=True))


class DetectionWhiteBoardRequest(_FiniteRequest):
    expected_aspect_ratio: _NUMERIC | None = None
    aspect_ratio_tolerance: _NUMERIC | None = None
    min_area_fraction: _NUMERIC | None = None
    max_area_fraction: _NUMERIC | None = None
    min_white_occupancy: _NUMERIC | None = None
    max_texture_std: _NUMERIC | None = None
    min_convexity: _NUMERIC | None = None
    min_side_px: _NUMERIC | None = None
    border_band_fraction: _NUMERIC | None = None

    def domain(self, current: WhiteBoardConfig) -> WhiteBoardConfig:
        return replace(current, **self.model_dump(exclude_none=True))


class DetectionRingsRequest(_FiniteRequest):
    expected_radius_ratios: tuple[_NUMERIC, ...] | None = None
    ratio_tolerance: _NUMERIC | None = None
    center_tolerance_fraction: _NUMERIC | None = None
    min_arc_coverage: _NUMERIC | None = None
    min_multiple_arcs: StrictInt | None = None
    max_single_arc_frames: StrictInt | None = None
    saturation_mask_radius_px: StrictInt | None = None

    def domain(self, current: RingGeometryConfig) -> RingGeometryConfig:
        return replace(current, **self.model_dump(exclude_none=True))


class DetectionClassicalScoringRequest(_FiniteRequest):
    white_weight: _NUMERIC | None = None
    geometry_weight: _NUMERIC | None = None
    ring_weight: _NUMERIC | None = None
    border_weight: _NUMERIC | None = None
    temporal_weight: _NUMERIC | None = None
    acquisition_threshold: _NUMERIC | None = None
    tracking_threshold: _NUMERIC | None = None
    ambiguity_margin: _NUMERIC | None = None
    max_texture_penalty: _NUMERIC | None = None

    def domain(self, current: ClassicalScoringConfig) -> ClassicalScoringConfig:
        return replace(current, **self.model_dump(exclude_none=True))


class DetectionConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str | None = None
    model: DetectionModelRequest | None = None
    roi_geometry: DetectionRoiGeometryRequest | None = None
    candidate_scoring: DetectionCandidateScoringRequest | None = None
    tracking: DetectionTrackingRequest | None = None
    normalization: DetectionNormalizationRequest | None = None
    white_board: DetectionWhiteBoardRequest | None = None
    rings: DetectionRingsRequest | None = None
    classical_scoring: DetectionClassicalScoringRequest | None = None

    def domain(self, current: DetectionConfig) -> DetectionConfig:
        if self.backend is not None and self.backend != current.backend:
            raise ConfigError("detection backend cannot be changed while camera is running")
        candidate = replace(
            current,
            model=current.model if self.model is None else self.model.domain(current.model),
            roi_geometry=(
                current.roi_geometry
                if self.roi_geometry is None
                else self.roi_geometry.domain()
            ),
            candidate_scoring=(
                current.candidate_scoring
                if self.candidate_scoring is None
                else self.candidate_scoring.domain()
            ),
            tracking=(
                current.tracking
                if self.tracking is None
                else self.tracking.domain(current.tracking)
            ),
            normalization=(
                current.normalization
                if self.normalization is None
                else self.normalization.domain(current.normalization)
            ),
            white_board=(
                current.white_board
                if self.white_board is None
                else self.white_board.domain(current.white_board)
            ),
            rings=(
                current.rings if self.rings is None else self.rings.domain(current.rings)
            ),
            classical_scoring=(
                current.classical_scoring
                if self.classical_scoring is None
                else self.classical_scoring.domain(current.classical_scoring)
            ),
        )
        _validate_detection(candidate)
        return candidate


def _package_static_dir() -> Path:
    return Path(__file__).with_name("static")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _parameters_response(value: EditableCameraParameters) -> dict[str, Any]:
    return _json_value(value)


_DETECTION_STATUS_FIELDS = (
    "enabled",
    "detected",
    "source_sequence",
    "observation",
    "observation_source",
    "confidence",
    "scale_px_per_mm",
    "velocity_px_s",
    "predicted_frames",
    "source_age_us",
    "near_image_edge",
    "partially_outside",
    "rejection_reasons",
    "error",
    "target_valid",
    "tracking_state",
    "model_state",
    "model_backend",
    "model_path",
    "candidate_count",
    "model_confidence",
    "geometry_score",
    "edge_support_score",
    "structure_score",
    "temporal_score",
    "combined_score",
    "confirmation_count",
    "miss_count",
    "failure_reason",
    "inference_ms",
    "geometry_ms",
    "total_ms",
    "result_age_ms",
    "homography_valid",
    "target_x_mm",
    "target_y_mm",
    "corners_px",
    "center_px",
    "candidates",
)


def _detection_status_response(value: Any) -> dict[str, Any]:
    return {name: _json_value(getattr(value, name)) for name in _DETECTION_STATUS_FIELDS}


def _detection_config_response(value: DetectionConfig) -> dict[str, Any]:
    return {
        "backend": value.backend,
        "model": {
            "confidence_threshold": value.model.confidence_threshold,
            "max_candidates": value.model.max_candidates,
        },
        "roi_geometry": _json_value(value.roi_geometry),
        "candidate_scoring": _json_value(value.candidate_scoring),
        "normalization": _json_value(value.normalization),
        "white_board": _json_value(value.white_board),
        "rings": _json_value(value.rings),
        "classical_scoring": _json_value(value.classical_scoring),
        "tracking": _json_value(value.tracking),
    }


def _profile_response(profile: Any) -> dict[str, Any]:
    return {
        "name": profile.name,
        "display_name": profile.display_name,
        "draft": _parameters_response(profile.parameters),
    }


def _storage_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail="profile not found")
    return HTTPException(status_code=500, detail="storage operation failed")


def _camera_identity(service: Any) -> CameraIdentity:
    identity = getattr(service, "camera_identity", None)
    if callable(identity):
        identity = identity()
    if identity is None:
        identity = getattr(service, "identity", None)
    if identity is None:
        identity = getattr(service, "_camera_identity", None)
    if not isinstance(identity, CameraIdentity):
        return CameraIdentity(model="", serial="")
    return identity


def _resize_max_width(image: np.ndarray, max_width: int | None) -> np.ndarray:
    if max_width is None or image.shape[1] <= max_width:
        return image
    height = max(1, round(image.shape[0] * max_width / image.shape[1]))
    return cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("OpenCV failed to encode preview JPEG")
    return encoded.tobytes()


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("OpenCV failed to encode detection debug PNG")
    return encoded.tobytes()


def _multipart_frame(jpeg: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
        + jpeg
        + b"\r\n"
    )


def _safe_capture_contract(storage: Any, capture_dir: Path) -> str:
    try:
        root = Path(storage.captures_dir).parent.resolve()
        relative = capture_dir.resolve().relative_to(root)
        return relative.as_posix()
    except (AttributeError, OSError, ValueError):
        parent = capture_dir.parent.name
        return f"{parent}/{capture_dir.name}" if parent else capture_dir.name


_CAPTURE_RESPONSE_FILES = frozenset(
    {
        "original.png",
        "overlay.png",
        "metadata.yaml",
        "model-candidates.png",
        "geometry-accepted.png",
        "geometry-rejected.png",
    }
)


def _capture_response_files(capture_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in capture_dir.iterdir()
        if path.name in _CAPTURE_RESPONSE_FILES and path.is_file()
    )


def create_camera_tuning_app(
    service: Any,
    storage: Any,
    project_defaults: EditableCameraParameters,
    preview_fps: float = 20.0,
) -> FastAPI:
    if isinstance(preview_fps, bool) or not isinstance(preview_fps, (int, float)):
        raise ValueError("preview_fps must be numeric")
    if not np.isfinite(preview_fps) or preview_fps <= 0:
        raise ValueError("preview_fps must be positive and finite")
    ParameterBounds().validate(project_defaults)
    interval_s = 1.0 / float(preview_fps)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            service.start()
            yield
        finally:
            service.stop()

    app = FastAPI(title="Camera Tuning", lifespan=lifespan)
    static_dir = _package_static_dir()
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        html = static_dir / "camera-tuning.html"
        if html.is_file():
            return HTMLResponse(html.read_text(encoding="utf-8"))
        return HTMLResponse(_PLACEHOLDER_PAGE)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "camera": _json_value(_camera_identity(service)),
            "fixed_format": dict(_FIXED_FORMAT),
            "runtime": _json_value(service.runtime_snapshot()),
            "applied": _parameters_response(service.applied_parameters()),
            "overlay": _json_value(_DEFAULT_OVERLAY),
            "detection": _detection_status_response(service.latest_detection()),
        }

    @app.get("/api/detection/config")
    def get_detection_config() -> dict[str, Any]:
        return _detection_config_response(service.detection_config())

    @app.put("/api/detection/config")
    def put_detection_config(request: DetectionConfigRequest) -> dict[str, Any]:
        try:
            candidate = request.domain(service.detection_config())
            applied = service.apply_detection_config(candidate)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=f"detection config apply failed: {exc}") from exc
        return _detection_config_response(applied)

    @app.get("/api/detection/status")
    def get_detection_status() -> dict[str, Any]:
        return _detection_status_response(service.latest_detection())

    def _detection_debug_response(image_name: str, sequence: int | None) -> Response:
        if image_name not in DEBUG_IMAGE_NAMES:
            raise HTTPException(status_code=404, detail="detection debug image not found")
        try:
            if image_name == "final-overlay":
                retained = service.detection_frame_for_latest(expected_sequence=sequence)
                if retained is None:
                    raise HTTPException(status_code=404, detail="detection frame unavailable")
                frame, detection = retained
                image = render_overlay(
                    frame.image,
                    source_sequence=frame.sequence,
                    detection=detection,
                    options=OverlayOptions(),
                )
            else:
                debug = service.detection_debug_for_latest(expected_sequence=sequence)
                if debug is None:
                    raise HTTPException(status_code=404, detail="detection debug unavailable")
                image = debug.images.get(image_name)
                if image is None:
                    raise HTTPException(status_code=404, detail="detection debug image unavailable")
        except StaleDetectionFrameError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        classical_debug = image_name in _CLASSICAL_DEBUG_IMAGE_NAMES
        encoded = _encode_png(image) if classical_debug else _encode_jpeg(image)
        return Response(
            content=encoded,
            media_type="image/png" if classical_debug else "image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/detection/debug")
    def get_detection_debug(
        image_name: str = Query(...),
        sequence: int | None = Query(None, ge=0),
    ) -> Response:
        return _detection_debug_response(image_name, sequence)

    @app.get("/api/detection/debug/{image_name}")
    def get_detection_debug_by_name(
        image_name: str,
        sequence: int | None = Query(None, ge=0),
    ) -> Response:
        return _detection_debug_response(image_name, sequence)

    @app.post("/api/detection/reload")
    @app.post("/api/detection/model/reload")
    def reload_detection_model() -> dict[str, Any]:
        config = service.detection_config()
        if config.backend == "classical":
            return {"status": "ready", "backend": "classical", "reloaded": False}
        try:
            service.reload_detection_model()
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"model reload failed: {exc}") from exc
        return _detection_status_response(service.latest_detection())

    @app.get("/api/parameters")
    def get_parameters() -> dict[str, Any]:
        return {
            "applied": _parameters_response(service.applied_parameters()),
            "project_defaults": _parameters_response(project_defaults),
            "bounds": _json_value(ParameterBounds()),
        }

    @app.put("/api/parameters")
    def put_parameters(request: ParameterRequest) -> dict[str, Any]:
        candidate = request.domain()
        try:
            ParameterBounds().validate(candidate)
            applied = service.apply_parameters(candidate)
        except ParameterApplyError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "apply_error": str(exc.apply_error),
                    "rollback_error": None if exc.rollback_error is None else str(exc.rollback_error),
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "apply_error": str(exc),
                    "rollback_error": None,
                },
            ) from exc
        return {"applied": _parameters_response(applied)}

    @app.get("/api/diagnostics")
    def get_diagnostics() -> dict[str, Any]:
        return {
            "diagnostics": _json_value(service.latest_diagnostics()),
            "detection": _json_value(service.latest_detection()),
        }

    @app.get("/api/profiles")
    def list_profiles() -> dict[str, Any]:
        try:
            return {"profiles": storage.list_profiles()}
        except Exception as exc:
            raise _storage_error(exc) from exc

    @app.get("/api/profiles/{name}")
    def load_profile(name: str) -> dict[str, Any]:
        try:
            return _profile_response(storage.load_profile(name))
        except Exception as exc:
            raise _storage_error(exc) from exc

    @app.put("/api/profiles/{name}")
    def save_profile(name: str, request: ProfileRequest) -> dict[str, Any]:
        candidate = request.parameters.domain()
        try:
            ParameterBounds().validate(candidate)
            profile = storage.save_profile(name, candidate, display_name=request.display_name)
        except Exception as exc:
            raise _storage_error(exc) from exc
        return _profile_response(profile)

    @app.delete("/api/profiles/{name}")
    def delete_profile(name: str) -> dict[str, str]:
        try:
            storage.delete_profile(name)
        except Exception as exc:
            raise _storage_error(exc) from exc
        return {"deleted": name}

    @app.post("/api/captures")
    def create_capture(request: OverlayRequest = Body(default_factory=OverlayRequest)) -> dict[str, Any]:
        options = request.domain()
        try:
            captured = service.capture_snapshot(options)
            overlay_image = render_overlay(
                captured.frame.image,
                source_sequence=captured.frame.sequence,
                detection=captured.detection,
                options=options,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            capture_dir = Path(storage.save_capture(captured, overlay_image))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="storage operation failed") from exc
        return {
            "capture": _safe_capture_contract(storage, capture_dir),
            "files": _capture_response_files(capture_dir),
        }

    @app.get("/api/preview.mjpg")
    async def preview(
        overlay: bool = Query(True),
        detection: bool = Query(True),
        show_board_outline: bool = Query(True),
        show_corners: bool = Query(True),
        show_center: bool = Query(True),
        show_crosshair: bool = Query(True),
        show_detection_text: bool = Query(True),
        show_center_roi: bool = Query(True),
        max_width: int | None = Query(None, ge=1, le=1280),
    ) -> StreamingResponse:
        options = OverlayOptions(
            enabled=overlay,
            show_board_outline=show_board_outline,
            show_corners=show_corners,
            show_center=show_center,
            show_crosshair=show_crosshair,
            show_detection_text=show_detection_text,
            show_center_roi=show_center_roi,
        )
        try:
            service.set_detection_enabled(detection)
        except (AttributeError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async def frames() -> AsyncIterator[bytes]:
            while True:
                frame = service.latest_frame()
                if frame is None:
                    await asyncio.sleep(interval_s)
                    continue
                image = frame.image
                if options.enabled:
                    image = render_overlay(
                        image,
                        source_sequence=frame.sequence,
                        detection=service.latest_detection(),
                        options=options,
                    )
                image = _resize_max_width(image, max_width)
                jpeg = _encode_jpeg(image)
                service.record_preview_frame()
                yield _multipart_frame(jpeg)
                await asyncio.sleep(interval_s)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    return app


__all__ = ["create_camera_tuning_app"]
