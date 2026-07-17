from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ev_vision.config import DetectionConfig
from ev_vision.models import Frame
from ev_vision.tuning.models import (
    DetectionCandidateSnapshot,
    DetectionDebugSnapshot,
    DetectionSnapshot,
    EditableCameraParameters,
    RuntimeSnapshot,
)
from ev_vision.tuning.service import StaleDetectionFrameError
from ev_vision.web.camera_tuning_app import create_camera_tuning_app


DEFAULTS = EditableCameraParameters(
    exposure_us=800.0,
    gain_db=6.0,
    acquisition_fps=120.0,
    auto_exposure=False,
    auto_gain=False,
    auto_white_balance=False,
)


def fake_candidate_payloads() -> list[dict[str, Any]]:
    return [
        {
            "xyxy_px": [90.0, 70.0, 530.0, 704.0],
            "accepted": True,
            "model_confidence": 0.91,
            "geometry_score": 0.88,
            "edge_support_score": 0.81,
            "structure_score": 0.72,
            "temporal_score": 0.75,
            "combined_score": 0.86,
            "failure_reason": None,
        },
        {
            "xyxy_px": [10.0, 20.0, 100.0, 140.0],
            "accepted": False,
            "model_confidence": 0.62,
            "geometry_score": 0.31,
            "edge_support_score": 0.22,
            "structure_score": 0.20,
            "temporal_score": 0.10,
            "combined_score": 0.37,
            "failure_reason": "LOW_GEOMETRY_SCORE",
        },
    ]


def detection_snapshot() -> DetectionSnapshot:
    candidates = tuple(DetectionCandidateSnapshot(**item) for item in fake_candidate_payloads())
    return DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=42,
        target_valid=True,
        tracking_state="TRACKING",
        model_state="READY",
        model_backend="onnx",
        model_path="models/target-board.onnx",
        candidate_count=2,
        model_confidence=0.91,
        geometry_score=0.88,
        edge_support_score=0.81,
        structure_score=0.72,
        temporal_score=0.75,
        combined_score=0.86,
        confirmation_count=3,
        miss_count=0,
        failure_reason=None,
        inference_ms=18.0,
        geometry_ms=4.0,
        total_ms=22.0,
        result_age_ms=12.0,
        homography_valid=True,
        target_x_mm=0.0,
        target_y_mm=0.0,
        corners_px=((100.0, 80.0), (520.0, 100.0), (500.0, 694.0), (120.0, 674.0)),
        center_px=(310.0, 387.0),
        candidates=candidates,
    )


def valid_detection_payload() -> dict[str, Any]:
    config = DetectionConfig(backend="hybrid")
    return {
        "backend": "hybrid",
        "model": {"confidence_threshold": 0.45, "max_candidates": 3},
        "roi_geometry": {
            "padding_fraction": 0.08,
            "canny_low": 60,
            "canny_high": 180,
            "min_edge_support": 0.45,
            "min_geometry_score": 0.55,
            "expected_aspect_ratio": 0.707,
            "aspect_ratio_tolerance": 0.35,
            "minimum_side_px": 40.0,
            "minimum_area_fraction": 0.25,
            "maximum_area_fraction": 1.15,
        },
        "candidate_scoring": {
            "model_weight": 0.45,
            "geometry_weight": 0.30,
            "structure_weight": 0.15,
            "temporal_weight": 0.10,
            "ambiguity_margin": 0.08,
        },
        "normalization": {field: getattr(config.normalization, field) for field in config.normalization.__dataclass_fields__},
        "white_board": {field: getattr(config.white_board, field) for field in config.white_board.__dataclass_fields__},
        "rings": {field: list(getattr(config.rings, field)) if field == "expected_radius_ratios" else getattr(config.rings, field) for field in config.rings.__dataclass_fields__},
        "classical_scoring": {field: getattr(config.classical_scoring, field) for field in config.classical_scoring.__dataclass_fields__},
        "tracking": {field: getattr(config.tracking, field) for field in config.tracking.__dataclass_fields__},
    }


class FakeService:
    def __init__(self) -> None:
        self.config = DetectionConfig(backend="hybrid")
        self.snapshot = detection_snapshot()
        self.apply_calls: list[EditableCameraParameters] = []
        self.detection_apply_calls: list[DetectionConfig] = []
        self.reload_calls = 0
        self.reload_error: BaseException | None = None
        self.stop_calls = 0
        self.debug_calls: list[int | None] = []
        self.frame_calls: list[int | None] = []
        self.debug_images = {
            "model-candidates": np.full((20, 30, 3), 10, dtype=np.uint8),
            "roi": np.full((10, 15, 3), 20, dtype=np.uint8),
            "roi-edges": np.full((10, 15), 255, dtype=np.uint8),
            "roi-geometry": np.full((10, 15, 3), 30, dtype=np.uint8),
        }
        self.frame = Frame(42, 123, np.full((20, 30, 3), 40, dtype=np.uint8))

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_calls += 1

    def runtime_snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(state="Streaming", frame_count=42)

    def applied_parameters(self) -> EditableCameraParameters:
        return DEFAULTS

    def latest_detection(self) -> DetectionSnapshot:
        return self.snapshot

    def detection_config(self) -> DetectionConfig:
        return self.config

    def apply_detection_config(self, config: DetectionConfig) -> DetectionConfig:
        self.detection_apply_calls.append(config)
        self.config = config
        return config

    def detection_debug_for_latest(
        self, *, expected_sequence: int | None = None
    ) -> DetectionDebugSnapshot | None:
        self.debug_calls.append(expected_sequence)
        if expected_sequence is not None and expected_sequence != self.frame.sequence:
            raise StaleDetectionFrameError(expected_sequence, self.frame.sequence)
        return DetectionDebugSnapshot(self.frame.sequence, self.debug_images)

    def detection_frame_for_latest(
        self, *, expected_sequence: int | None = None
    ) -> tuple[Frame, DetectionSnapshot] | None:
        self.frame_calls.append(expected_sequence)
        if expected_sequence is not None and expected_sequence != self.frame.sequence:
            raise StaleDetectionFrameError(expected_sequence, self.frame.sequence)
        return self.frame, self.snapshot

    def reload_detection_model(self) -> None:
        self.reload_calls += 1
        if self.reload_error is not None:
            raise self.reload_error


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def client(service: FakeService) -> TestClient:
    app = create_camera_tuning_app(service, object(), DEFAULTS, preview_fps=1000.0)
    with TestClient(app) as test_client:
        yield test_client


def test_detection_status_exposes_scores_tracker_and_model(client: TestClient) -> None:
    response = client.get("/api/detection/status")
    assert response.status_code == 200
    assert response.json() == detection_snapshot_payload(detection_snapshot())


def test_detection_status_matches_legacy_status_detection_key(client: TestClient) -> None:
    legacy = client.get("/api/status")
    detection = client.get("/api/detection/status")
    assert legacy.status_code == detection.status_code == 200
    assert legacy.json()["detection"] == detection.json()
    assert "observation" in detection.json()
    assert "error" in detection.json()
    assert {"camera", "fixed_format", "runtime", "applied", "overlay", "detection"} <= legacy.json().keys()


def test_detection_config_update_is_separate_from_camera_parameters(
    client: TestClient, service: FakeService
) -> None:
    payload = client.get("/api/detection/config").json()
    assert payload == valid_detection_payload()
    payload["model"]["confidence_threshold"] = 0.50
    response = client.put("/api/detection/config", json=payload)
    assert response.status_code == 200
    assert response.json()["model"]["confidence_threshold"] == 0.50
    assert service.detection_apply_calls[-1].model.confidence_threshold == 0.50
    assert service.apply_calls == []


@pytest.mark.parametrize(
    "unsafe_patch",
    [
        {"backend": "classical"},
        {"model": {"path": "C:/secret/model.engine"}},
        {"model": {"fallback_path": "C:/secret/model.onnx"}},
        {"model": {"input_width": 320}},
        {"model": {"input_height": 320}},
        {"model": {"device": 1}},
    ],
)
def test_detection_config_rejects_unsafe_hot_changes(
    client: TestClient, service: FakeService, unsafe_patch: dict[str, Any]
) -> None:
    payload = valid_detection_payload()
    for key, value in unsafe_patch.items():
        if key == "model":
            payload["model"].update(value)
        else:
            payload[key] = value
    response = client.put("/api/detection/config", json=payload)
    assert response.status_code == 422
    assert service.detection_apply_calls == []


def test_invalid_detection_config_returns_422(
    client: TestClient, service: FakeService
) -> None:
    payload = valid_detection_payload()
    payload["roi_geometry"]["canny_low"] = 200
    payload["roi_geometry"]["canny_high"] = 100
    response = client.put("/api/detection/config", json=payload)
    assert response.status_code == 422
    assert "canny_low" in str(response.json()["detail"])
    assert service.detection_apply_calls == []


def test_detection_config_rejects_non_finite_values(client: TestClient) -> None:
    payload = valid_detection_payload()
    payload["candidate_scoring"]["model_weight"] = "NaN"
    response = client.put("/api/detection/config", json=payload)
    assert response.status_code == 422


def test_debug_image_rejects_unknown_name_and_stale_sequence(client: TestClient) -> None:
    assert client.get("/api/detection/debug", params={"image_name": "not-a-view"}).status_code == 404
    response = client.get(
        "/api/detection/debug",
        params={"image_name": "roi-edges", "sequence": 1},
    )
    assert response.status_code == 409
    assert "latest" in response.json()["detail"]


def test_canonical_debug_image_alias_returns_matching_view(client: TestClient) -> None:
    response = client.get(
        "/api/detection/debug/roi-geometry",
        params={"sequence": 42},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"


def test_canonical_model_reload_alias_preserves_legacy_route(
    client: TestClient, service: FakeService
) -> None:
    canonical = client.post("/api/detection/model/reload")
    legacy = client.post("/api/detection/reload")
    assert canonical.status_code == 200
    assert legacy.status_code == 200
    assert service.reload_calls == 2


def test_debug_request_returns_no_store_jpeg_without_advancing_tracker(
    client: TestClient, service: FakeService
) -> None:
    before = client.get("/api/detection/status").json()
    response = client.get(
        "/api/detection/debug",
        params={"image_name": "roi-geometry", "sequence": before["source_sequence"]},
    )
    after = client.get("/api/detection/status").json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content.startswith(b"\xff\xd8")
    assert service.debug_calls == [42]
    assert after["confirmation_count"] == before["confirmation_count"]
    assert after["miss_count"] == before["miss_count"]


def test_final_overlay_uses_matching_retained_frame_and_snapshot(
    client: TestClient, service: FakeService
) -> None:
    response = client.get(
        "/api/detection/debug",
        params={"image_name": "final-overlay", "sequence": 42},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert service.frame_calls == [42]
    assert service.debug_calls == []


def test_unavailable_debug_image_returns_404(client: TestClient, service: FakeService) -> None:
    service.debug_images.pop("roi")
    response = client.get(
        "/api/detection/debug",
        params={"image_name": "roi", "sequence": 42},
    )
    assert response.status_code == 404


def test_model_reload_failure_is_diagnostic_and_preview_safe(
    client: TestClient, service: FakeService
) -> None:
    service.reload_error = RuntimeError("bad engine")
    response = client.post("/api/detection/reload")
    assert response.status_code == 409
    assert response.json()["detail"] == "model reload failed: bad engine"
    assert service.reload_calls == 1
    assert service.stop_calls == 0
    assert client.get("/api/status").status_code == 200


def test_model_reload_success_returns_current_detection_status(
    client: TestClient, service: FakeService
) -> None:
    response = client.post("/api/detection/reload")
    assert response.status_code == 200
    assert response.json() == detection_snapshot_payload(service.snapshot)
    assert service.reload_calls == 1


def detection_snapshot_payload(snapshot: DetectionSnapshot) -> dict[str, Any]:
    return {
        "enabled": snapshot.enabled,
        "detected": snapshot.detected,
        "source_sequence": snapshot.source_sequence,
        "observation": None,
        "observation_source": snapshot.observation_source,
        "confidence": snapshot.confidence,
        "scale_px_per_mm": snapshot.scale_px_per_mm,
        "velocity_px_s": list(snapshot.velocity_px_s) if snapshot.velocity_px_s is not None else None,
        "predicted_frames": snapshot.predicted_frames,
        "source_age_us": snapshot.source_age_us,
        "near_image_edge": snapshot.near_image_edge,
        "partially_outside": snapshot.partially_outside,
        "rejection_reasons": list(snapshot.rejection_reasons),
        "error": snapshot.error,
        "target_valid": snapshot.target_valid,
        "tracking_state": snapshot.tracking_state,
        "model_state": snapshot.model_state,
        "model_backend": snapshot.model_backend,
        "model_path": snapshot.model_path,
        "candidate_count": snapshot.candidate_count,
        "model_confidence": snapshot.model_confidence,
        "geometry_score": snapshot.geometry_score,
        "edge_support_score": snapshot.edge_support_score,
        "structure_score": snapshot.structure_score,
        "temporal_score": snapshot.temporal_score,
        "combined_score": snapshot.combined_score,
        "confirmation_count": snapshot.confirmation_count,
        "miss_count": snapshot.miss_count,
        "failure_reason": snapshot.failure_reason,
        "inference_ms": snapshot.inference_ms,
        "geometry_ms": snapshot.geometry_ms,
        "total_ms": snapshot.total_ms,
        "result_age_ms": snapshot.result_age_ms,
        "homography_valid": snapshot.homography_valid,
        "target_x_mm": snapshot.target_x_mm,
        "target_y_mm": snapshot.target_y_mm,
        "corners_px": [list(point) for point in snapshot.corners_px],
        "center_px": list(snapshot.center_px) if snapshot.center_px is not None else None,
        "candidates": fake_candidate_payloads(),
    }


def classical_detection_payload() -> dict[str, Any]:
    return {
        "backend": "classical",
        "normalization": {
            "gaussian_kernel": 3, "clahe_clip_limit": 2.4,
            "clahe_grid_size": 8, "illumination_kernel": 81,
            "white_percentile": 72.0, "white_local_offset": 10.0,
            "saturation_threshold": 250, "canny_low": 40, "canny_high": 120,
        },
        "white_board": {
            "expected_aspect_ratio": 210.0 / 297.0,
            "aspect_ratio_tolerance": 0.24, "min_area_fraction": 0.015,
            "max_area_fraction": 0.92, "min_white_occupancy": 0.55,
            "max_texture_std": 58.0, "min_convexity": 0.90,
            "min_side_px": 45.0, "border_band_fraction": 0.045,
        },
        "rings": {
            "expected_radius_ratios": [1, 2, 3, 4, 5],
            "ratio_tolerance": 0.20, "center_tolerance_fraction": 0.08,
            "min_arc_coverage": 0.18, "min_multiple_arcs": 2,
            "max_single_arc_frames": 2, "saturation_mask_radius_px": 12,
        },
        "classical_scoring": {
            "white_weight": 0.24, "geometry_weight": 0.22,
            "ring_weight": 0.30, "border_weight": 0.08,
            "temporal_weight": 0.16, "acquisition_threshold": 0.66,
            "tracking_threshold": 0.50, "ambiguity_margin": 0.08,
            "max_texture_penalty": 0.20,
        },
        "tracking": {
            "confirm_frames": 3, "predict_frames": 2,
            "predict_max_frames": 3, "predict_max_ms": 150.0,
            "lost_frames": 4, "max_single_arc_frames": 2,
            "max_center_jump_px": 160.0, "max_scale_jump_fraction": 0.30,
            "max_velocity_px_s": 5000.0,
            "max_acceleration_px_s2": 30000.0,
            "max_result_age_ms": 100.0,
        },
    }


def test_config_exposes_classical_sections(client: TestClient, service: FakeService) -> None:
    service.config = DetectionConfig(backend="classical")
    payload = client.get("/api/detection/config").json()
    assert payload["backend"] == "classical"
    assert payload["normalization"]["clahe_clip_limit"] == 2.0
    assert payload["rings"]["expected_radius_ratios"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert "red_threshold" not in str(payload).lower()


def test_put_classical_config_preserves_hybrid_sections_and_does_not_restart_camera(
    client: TestClient, service: FakeService
) -> None:
    service.config = DetectionConfig(backend="classical")
    service.camera_generation = 7
    response = client.put("/api/detection/config", json=classical_detection_payload())
    assert response.status_code == 200
    assert service.camera_generation == 7
    changed = service.detection_apply_calls[-1]
    assert changed.backend == "classical"
    assert changed.model == DetectionConfig().model
    assert changed.roi_geometry == DetectionConfig().roi_geometry
    assert changed.normalization.clahe_clip_limit == 2.4
    assert changed.white_board.min_white_occupancy == 0.55
    assert changed.rings.ratio_tolerance == 0.20
    assert changed.classical_scoring.tracking_threshold == 0.50


def test_status_reports_classical_source_and_prediction(client: TestClient, service: FakeService) -> None:
    service.snapshot = DetectionSnapshot(
        enabled=True, detected=True, source_sequence=42, target_valid=True,
        tracking_state="PREDICTING", observation_source="PREDICTED",
        confidence=0.77, source_age_us=45000, predicted_frames=2,
        model_state="READY", model_backend="classical",
    )
    payload = client.get("/api/detection/status").json()
    assert payload["observation_source"] == "PREDICTED"
    assert payload["confidence"] == pytest.approx(0.77)
    assert payload["source_age_us"] == 45000
    assert payload["predicted_frames"] == 2


@pytest.mark.parametrize("name", [
    "normalized-gray", "white-mask", "edge-mask",
    "ring-arcs", "candidate-scores",
])
def test_classical_debug_route_returns_png(
    client: TestClient, service: FakeService, name: str
) -> None:
    service.debug_images[name] = np.full((10, 15), 127, dtype=np.uint8)
    response = client.get(f"/api/detection/debug/{name}", params={"sequence": 42})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_classical_reload_route_is_ready_no_op(
    client: TestClient, service: FakeService
) -> None:
    service.config = DetectionConfig(backend="classical")
    response = client.post("/api/detection/model/reload")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready", "backend": "classical", "reloaded": False
    }
    assert service.reload_calls == 0
