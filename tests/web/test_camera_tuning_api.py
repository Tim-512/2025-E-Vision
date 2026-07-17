from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ev_vision.config import CameraConfig
from ev_vision.models import BoardObservation, Frame
from ev_vision.tuning.models import (
    CameraIdentity,
    CaptureSnapshot,
    DetectionSnapshot,
    EditableCameraParameters,
    ImageDiagnostics,
    OverlayOptions,
    RuntimeSnapshot,
)
from ev_vision.tuning.service import ParameterApplyError
from ev_vision.tuning.storage import NamedProfile
from ev_vision.web.camera_tuning_app import create_camera_tuning_app


PARAMETERS = {
    "exposure_us": 1200.0,
    "gain_db": 4.5,
    "acquisition_fps": 60.0,
    "auto_exposure": False,
    "auto_gain": False,
    "auto_white_balance": True,
}
DEFAULTS = EditableCameraParameters(
    exposure_us=800.0,
    gain_db=6.0,
    acquisition_fps=120.0,
    auto_exposure=False,
    auto_gain=False,
    auto_white_balance=False,
)


def parameters(**changes: Any) -> EditableCameraParameters:
    return EditableCameraParameters(**{**PARAMETERS, **changes})


def diagnostics() -> ImageDiagnostics:
    histogram = tuple(np.int64(index) for index in range(256))
    return ImageDiagnostics(
        source_sequence=np.int64(7),
        computed_ns=np.int64(456),
        gray_histogram=histogram,
        blue_histogram=histogram,
        green_histogram=histogram,
        red_histogram=histogram,
        dark_percent=np.float32(1.25),
        bright_percent=np.float64(2.5),
        focus_score=np.float32(33.75),
        roi_px=tuple(np.int32(value) for value in (3, 2, 6, 4)),
    )


def detection() -> DetectionSnapshot:
    return DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=np.int64(7),
        observation=BoardObservation(
            captured_ns=np.int64(123),
            corners_px=np.array(
                [[2.0, 2.0], [17.0, 2.0], [17.0, 17.0], [2.0, 17.0]],
                dtype=np.float32,
            ),
            center_px=np.array([9.5, 9.5], dtype=np.float64),
            confidence=np.float32(0.9),
            homography_valid=np.bool_(True),
        ),
        result_age_ms=np.float32(3.0),
    )


def snapshot() -> CaptureSnapshot:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :15] = (10, 20, 30)
    return CaptureSnapshot(
        frame=Frame(sequence=7, captured_ns=123, image=image),
        parameters=parameters(),
        runtime=RuntimeSnapshot(state="Streaming", frame_count=7),
        diagnostics=diagnostics(),
        detection=detection(),
        overlay_options=OverlayOptions(),
        camera_config=CameraConfig(),
        camera_identity=CameraIdentity(model="MV-CA013-21UC", serial="00G02809155"),
    )


class FakeService:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.apply_calls: list[EditableCameraParameters] = []
        self.preview_calls = 0
        self.detection_enabled_calls: list[bool] = []
        self.applied = parameters()
        self.runtime = RuntimeSnapshot(
            state="Streaming",
            acquisition_fps=np.float32(119.5),
            preview_fps=np.float64(19.0),
            detection_fps=np.float32(14.5),
            diagnostics_fps=np.float64(9.5),
            frame_count=np.int64(42),
            timeout_count=np.int32(2),
            sequence_gap_count=np.int32(1),
            frame_age_ms=np.float32(4.25),
            last_error=None,
        )
        self.identity = CameraIdentity(model="MV-CA013-21UC", serial="00G02809155")
        self.diagnostic = diagnostics()
        self.detect = detection()
        self.current_frame: Frame | None = snapshot().frame
        self.capture = snapshot()
        self.apply_error: BaseException | None = None
        self.start_error: BaseException | None = None

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stop_calls += 1

    def apply_parameters(self, candidate: EditableCameraParameters) -> EditableCameraParameters:
        self.apply_calls.append(candidate)
        if self.apply_error is not None:
            raise self.apply_error
        self.applied = candidate
        return candidate

    def applied_parameters(self) -> EditableCameraParameters:
        return self.applied

    def runtime_snapshot(self) -> RuntimeSnapshot:
        return self.runtime

    def latest_diagnostics(self) -> ImageDiagnostics | None:
        return self.diagnostic

    def latest_detection(self) -> DetectionSnapshot:
        return self.detect

    def latest_frame(self) -> Frame | None:
        return self.current_frame

    def set_detection_enabled(self, enabled: bool) -> None:
        self.detection_enabled_calls.append(enabled)
        self.detect = DetectionSnapshot(enabled=enabled, detected=False)

    def capture_snapshot(self, overlay_options: OverlayOptions | None = None) -> CaptureSnapshot:
        if overlay_options is None:
            return self.capture
        return CaptureSnapshot(
            frame=self.capture.frame,
            parameters=self.capture.parameters,
            runtime=self.capture.runtime,
            diagnostics=self.capture.diagnostics,
            detection=self.capture.detection,
            overlay_options=overlay_options,
            camera_config=self.capture.camera_config,
            camera_identity=self.capture.camera_identity,
        )

    def record_preview_frame(self) -> None:
        self.preview_calls += 1


class FakeStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.profiles: dict[str, NamedProfile] = {}
        self.saved_profiles: list[NamedProfile] = []
        self.deleted: list[str] = []
        self.saved_capture: tuple[CaptureSnapshot, np.ndarray] | None = None
        self.failure: BaseException | None = None

    def _maybe_fail(self) -> None:
        if self.failure is not None:
            raise self.failure

    def list_profiles(self) -> list[str]:
        self._maybe_fail()
        return sorted(self.profiles)

    def load_profile(self, name: str) -> NamedProfile:
        self._maybe_fail()
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise FileNotFoundError(name) from exc

    def save_profile(
        self,
        name: str,
        candidate: EditableCameraParameters,
        *,
        display_name: str | None = None,
    ) -> NamedProfile:
        self._maybe_fail()
        if "/" in name or ".." in name:
            raise ValueError("unsafe profile name")
        profile = NamedProfile(name=name, display_name=display_name, parameters=candidate)
        self.profiles[name] = profile
        self.saved_profiles.append(profile)
        return profile

    def delete_profile(self, name: str) -> None:
        self._maybe_fail()
        if name not in self.profiles:
            raise FileNotFoundError(name)
        del self.profiles[name]
        self.deleted.append(name)

    def save_capture(self, captured: CaptureSnapshot, overlay_image: np.ndarray) -> Path:
        self._maybe_fail()
        self.saved_capture = (captured, overlay_image.copy())
        capture_dir = self.root / "captures" / "20260715_010203_456"
        capture_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "original.png",
            "overlay.png",
            "metadata.yaml",
            "model-candidates.png",
            "geometry-accepted.png",
            "normalized-gray.png",
            "white-mask.png",
            "edge-mask.png",
            "ring-arcs.png",
            "candidate-scores.png",
        ):
            (capture_dir / name).write_bytes(b"capture")
        (capture_dir / "subdirectory").mkdir(exist_ok=True)
        (capture_dir / "unexpected.txt").write_text("ignore", encoding="utf-8")
        return capture_dir


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def storage(tmp_path: Path) -> FakeStorage:
    return FakeStorage(tmp_path)


@pytest.fixture
def app(service: FakeService, storage: FakeStorage):
    return create_camera_tuning_app(service, storage, DEFAULTS, preview_fps=1000.0)


def test_lifespan_starts_and_stops_service_exactly_once(app, service: FakeService) -> None:
    with TestClient(app) as client:
        assert client.get("/api/status").status_code == 200
        assert service.start_calls == 1
        assert service.stop_calls == 0
    assert service.start_calls == 1
    assert service.stop_calls == 1


def test_lifespan_stops_service_when_start_raises(service: FakeService, storage: FakeStorage) -> None:
    service.start_error = RuntimeError("camera unavailable")
    app = create_camera_tuning_app(service, storage, DEFAULTS)
    with pytest.raises(RuntimeError, match="camera unavailable"):
        with TestClient(app):
            pass
    assert service.start_calls == 1
    assert service.stop_calls == 1


def test_root_returns_minimal_page_and_static_mount_can_serve_task6_assets(
    app, service: FakeService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Camera Tuning" in response.text
        assert "/static/camera-tuning.css" in response.text

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "probe.txt").write_text("static-ok", encoding="utf-8")
    monkeypatch.setattr("ev_vision.web.camera_tuning_app._package_static_dir", lambda: static_dir)
    mounted = create_camera_tuning_app(service, FakeStorage(tmp_path), DEFAULTS)
    with TestClient(mounted) as client:
        assert client.get("/static/probe.txt").text == "static-ok"


def test_status_has_fixed_format_identity_runtime_applied_overlay_and_detection_without_controls(app) -> None:
    with TestClient(app) as client:
        payload = client.get("/api/status").json()
    assert payload["camera"] == {"model": "MV-CA013-21UC", "serial": "00G02809155"}
    assert payload["fixed_format"] == {
        "width": 1280,
        "height": 1024,
        "pixel_format": "BayerRG8",
        "buffer_size": 2,
    }
    assert payload["runtime"]["state"] == "Streaming"
    assert payload["runtime"]["frame_count"] == 42
    assert payload["applied"] == PARAMETERS
    assert payload["overlay"] == OverlayOptions().__dict__
    assert payload["detection"]["enabled"] is True
    lowered = str(payload).lower()
    assert "laser" not in lowered
    assert "gimbal" not in lowered



def test_route_surface_contains_only_camera_tuning_endpoints(app) -> None:
    paths = {getattr(route, "path", None) for route in app.routes}
    assert {
        "/",
        "/static",
        "/api/status",
        "/api/parameters",
        "/api/diagnostics",
        "/api/preview.mjpg",
        "/api/profiles",
        "/api/profiles/{name}",
        "/api/captures",
    } <= paths
    application_paths = " ".join(path for path in paths if path and not path.startswith("/openapi") and path not in {"/docs", "/docs/oauth2-redirect", "/redoc"}).lower()
    assert "laser" not in application_paths
    assert "gimbal" not in application_paths
def test_parameters_get_and_complete_put_apply_candidate(app, service: FakeService) -> None:
    with TestClient(app) as client:
        before = client.get("/api/parameters")
        response = client.put("/api/parameters", json=PARAMETERS)
    assert before.json()["project_defaults"] == DEFAULTS.to_dict()
    assert before.json()["bounds"] == {
        "exposure_min_us": 20.0,
        "exposure_max_us": 1_000_000.0,
        "gain_min_db": 0.0,
        "gain_max_db": 24.0,
        "acquisition_fps_min": 1.0,
        "acquisition_fps_max": 120.0,
    }
    assert response.status_code == 200
    assert response.json() == {"applied": PARAMETERS}
    assert service.apply_calls == [parameters()]


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in PARAMETERS.items() if key != "gain_db"},
        {**PARAMETERS, "exposure_us": True},
        {**PARAMETERS, "gain_db": False},
        {**PARAMETERS, "acquisition_fps": True},
        {**PARAMETERS, "auto_gain": 1},
        {**PARAMETERS, "gain_db": 1000.0},
    ],
)
def test_invalid_parameter_put_is_422_without_apply(app, service: FakeService, payload: dict[str, Any]) -> None:
    with TestClient(app) as client:
        response = client.put("/api/parameters", json=payload)
    assert response.status_code == 422
    assert service.apply_calls == []


def test_apply_value_error_is_422(app, service: FakeService) -> None:
    service.apply_error = ValueError("candidate rejected")
    with TestClient(app) as client:
        response = client.put("/api/parameters", json=PARAMETERS)
    assert response.status_code == 422
    assert response.json()["detail"] == "candidate rejected"
    assert service.apply_calls == [parameters()]


def test_apply_runtime_conflict_is_409_with_actionable_detail(app, service: FakeService) -> None:
    service.apply_error = RuntimeError("camera tuning service is not running")
    with TestClient(app) as client:
        response = client.put("/api/parameters", json=PARAMETERS)
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "camera tuning service is not running",
        "apply_error": "camera tuning service is not running",
        "rollback_error": None,
    }

def test_apply_failure_with_successful_rollback_reports_null_rollback_detail(app, service: FakeService) -> None:
    service.apply_error = ParameterApplyError(
        "apply failed; previous parameters restored",
        apply_error=RuntimeError("native apply failed"),
    )
    with TestClient(app) as client:
        response = client.put("/api/parameters", json=PARAMETERS)
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "apply failed; previous parameters restored",
        "apply_error": "native apply failed",
        "rollback_error": None,
    }

def test_apply_failure_is_409_with_rollback_details(app, service: FakeService) -> None:
    service.apply_error = ParameterApplyError(
        "apply failed and rollback failed",
        apply_error=RuntimeError("native apply failed"),
        rollback_error=RuntimeError("native rollback failed"),
    )
    with TestClient(app) as client:
        response = client.put("/api/parameters", json=PARAMETERS)
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "apply failed and rollback failed",
        "apply_error": "native apply failed",
        "rollback_error": "native rollback failed",
    }


def test_diagnostics_explicitly_serializes_dataclasses_tuples_and_numpy(app) -> None:
    with TestClient(app) as client:
        payload = client.get("/api/diagnostics").json()
    assert payload["diagnostics"]["gray_histogram"][7] == 7
    assert payload["diagnostics"]["roi_px"] == [3, 2, 6, 4]
    assert payload["detection"]["observation"]["corners_px"][0] == [2.0, 2.0]
    assert payload["detection"]["observation"]["homography_valid"] is True


def test_profile_crud_returns_draft_without_applying_or_touching_defaults(
    app, service: FakeService, storage: FakeStorage, tmp_path: Path
) -> None:
    default_yaml = tmp_path / "config" / "default.yaml"
    default_yaml.parent.mkdir()
    default_yaml.write_text("sentinel: unchanged\n", encoding="utf-8")
    body = {"display_name": "Indoor", "parameters": PARAMETERS}
    with TestClient(app) as client:
        saved = client.put("/api/profiles/indoor-normal", json=body)
        listed = client.get("/api/profiles")
        loaded = client.get("/api/profiles/indoor-normal")
        deleted = client.delete("/api/profiles/indoor-normal")
    assert saved.status_code == 200
    assert listed.json() == {"profiles": ["indoor-normal"]}
    assert loaded.json() == {"name": "indoor-normal", "display_name": "Indoor", "draft": PARAMETERS}
    assert deleted.json() == {"deleted": "indoor-normal"}
    assert service.apply_calls == []
    assert default_yaml.read_text(encoding="utf-8") == "sentinel: unchanged\n"


def test_profile_errors_map_to_422_404_and_500(app, storage: FakeStorage) -> None:
    with TestClient(app) as client:
        unsafe = client.put("/api/profiles/..", json={"parameters": PARAMETERS})
        missing = client.get("/api/profiles/missing")
        storage.failure = OSError("disk unavailable")
        failed = client.get("/api/profiles")
    assert unsafe.status_code in {404, 422}
    assert missing.status_code == 404
    assert failed.status_code == 500
    assert failed.json()["detail"] == "storage operation failed"


def test_capture_renders_overlay_saves_snapshot_and_returns_safe_relative_contract(app, storage: FakeStorage) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/captures",
            json={"overlay": True, "show_crosshair": False, "show_center_roi": False},
        )
    assert response.status_code == 200
    assert response.json() == {
        "capture": "captures/20260715_010203_456",
        "files": [
            "candidate-scores.png",
            "edge-mask.png",
            "geometry-accepted.png",
            "metadata.yaml",
            "model-candidates.png",
            "normalized-gray.png",
            "original.png",
            "overlay.png",
            "ring-arcs.png",
            "white-mask.png",
        ],
    }
    assert storage.saved_capture is not None
    saved_snapshot, overlay = storage.saved_capture
    assert saved_snapshot.overlay_options.show_crosshair is False
    assert saved_snapshot.overlay_options.show_center_roi is False
    assert overlay.shape == saved_snapshot.frame.image.shape
    assert np.any(overlay != saved_snapshot.frame.image)


def test_capture_storage_failure_is_500(app, storage: FakeStorage) -> None:
    storage.failure = OSError("read-only filesystem")
    with TestClient(app) as client:
        response = client.post("/api/captures", json={})
    assert response.status_code == 500
    assert response.json()["detail"] == "storage operation failed"


def _first_preview_chunk(app, **changes: Any):
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/preview.mjpg"
    )
    options = {
        "overlay": True,
        "detection": True,
        "show_board_outline": True,
        "show_corners": True,
        "show_center": True,
        "show_crosshair": True,
        "show_detection_text": True,
        "show_center_roi": True,
        "max_width": None,
        **changes,
    }

    async def read_first():
        response = await endpoint(**options)
        try:
            first = await anext(response.body_iterator)
        finally:
            await response.body_iterator.aclose()
        return response, first

    return asyncio.run(read_first())


def test_preview_first_chunk_is_latest_only_overlay_resized_jpeg_and_records_counter(app, service: FakeService) -> None:
    response, first = _first_preview_chunk(
        app, overlay=False, detection=False, max_width=10
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace; boundary=frame")
    assert first.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    jpeg = first.split(b"\r\n\r\n", 1)[1].split(b"\r\n", 1)[0]
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 10
    assert service.preview_calls == 1
    assert service.detection_enabled_calls == [False]


def test_preview_no_frame_retries_until_a_frame_arrives(app, service: FakeService, monkeypatch: pytest.MonkeyPatch) -> None:
    service.current_frame = None
    calls = 0

    def latest_frame() -> Frame | None:
        nonlocal calls
        calls += 1
        if calls < 2:
            return None
        return snapshot().frame

    monkeypatch.setattr(service, "latest_frame", latest_frame)
    _, first = _first_preview_chunk(app)
    assert first.startswith(b"--frame")
    assert calls == 2


def test_dashboard_contains_required_camera_tuning_controls_without_actuator_controls(app) -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    required_ids = {
        "preview",
        "exposure-us",
        "gain-db",
        "acquisition-fps",
        "auto-exposure",
        "auto-gain",
        "auto-white-balance",
        "apply-parameters",
        "revert-draft",
        "restore-defaults",
        "detection-enabled",
        "pause-preview",
        "capture-button",
        "profile-name",
        "profile-select",
        "load-profile",
        "save-profile",
        "delete-profile",
        "gray-histogram",
        "rgb-histogram",
        "center-roi",
        "status-message",
    }
    for element_id in required_ids:
        assert f'id="{element_id}"' in html
    assert 'role="status"' in html
    assert "/static/camera-tuning.css" in html
    assert "/static/camera-tuning.js" in html
    lowered = html.lower()
    assert "laser safety" in lowered
    assert "hardware-always-on" in lowered
    assert "jetson/v2 cannot control" in lowered
    assert "software cannot make it safe" in lowered
    assert "gimbal" not in lowered
