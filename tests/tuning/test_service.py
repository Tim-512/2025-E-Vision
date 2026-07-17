from __future__ import annotations

from collections import deque
from dataclasses import replace
import threading
import time
from typing import Iterable

import numpy as np
import pytest

from ev_vision.config import CameraConfig, DetectionConfig, RingGeometryConfig
from ev_vision.detection.contracts import ClassicalBoardResult, ObservationSource
from ev_vision.detection.failures import (
    CandidateEvaluation,
    DetectionFailure,
    HybridBoardResult,
)
from ev_vision.detection.roi_board_geometry import GeometryResult
from ev_vision.detection.yolo_board import BoardSearchResult
from ev_vision.models import BoardObservation, Frame
from ev_vision.tuning.models import CameraIdentity, EditableCameraParameters, OverlayOptions
from ev_vision.tuning.service import CameraTuningService, ParameterApplyError


class FakeCamera:
    def __init__(self, items: Iterable[Frame | BaseException] = (), *, open_error: BaseException | None = None, idle_sleep_s: float = 0.001) -> None:
        self.items = deque(items)
        self.open_error = open_error
        self.idle_sleep_s = idle_sleep_s
        self.open_count = 0
        self.close_count = 0
        self.read_count = 0
        self.opened = False
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            self.open_count += 1
            if self.open_error is not None:
                raise self.open_error
            self.opened = True

    def close(self) -> None:
        with self._lock:
            self.close_count += 1
            self.opened = False

    def read(self, *, timeout_ms: int = 100) -> Frame:
        with self._lock:
            if not self.opened:
                raise RuntimeError("read on closed camera")
            self.read_count += 1
            item = self.items.popleft() if self.items else None
        if item is None:
            time.sleep(min(self.idle_sleep_s, timeout_ms / 1000.0))
            raise TimeoutError("fake timeout")
        if isinstance(item, BaseException):
            raise item
        return item


class ReconfigurableFakeCamera(FakeCamera):
    def __init__(
        self,
        items: Iterable[Frame | BaseException] = (),
        *,
        reconfigure_results: Iterable[BaseException | None] = (),
    ) -> None:
        super().__init__(items)
        self.reconfigure_results = deque(reconfigure_results)
        self.reconfigure_calls: list[CameraConfig] = []

    def reconfigure(self, camera_config: CameraConfig) -> None:
        self.reconfigure_calls.append(camera_config)
        result = self.reconfigure_results.popleft() if self.reconfigure_results else None
        if result is not None:
            raise result
        self.config = camera_config
        self.items.append(frame(100 + len(self.reconfigure_calls)))


class FakeFactory:
    def __init__(self, cameras: Iterable[FakeCamera]) -> None:
        self.cameras = deque(cameras)
        self.configs: list[CameraConfig] = []
        self.created: list[FakeCamera] = []
        self._lock = threading.Lock()

    def __call__(self, config: CameraConfig) -> FakeCamera:
        with self._lock:
            self.configs.append(config)
            if not self.cameras:
                raise AssertionError("unexpected camera factory call")
            camera = self.cameras.popleft()
            self.created.append(camera)
            return camera


class FakeDetector:
    def __init__(self, results: Iterable[BoardObservation | None | BaseException]) -> None:
        self.results = deque(results)
        self.calls: list[tuple[np.ndarray, int]] = []

    def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
        self.calls.append((image.copy(), captured_ns))
        result = self.results.popleft() if self.results else None
        if isinstance(result, BaseException):
            raise result
        return result


class HybridDetectorFake:
    def __init__(self, results: Iterable[HybridBoardResult | BaseException]) -> None:
        self.results = deque(results)
        self.calls: list[dict[str, object]] = []
        self.reset_calls = 0
        self.config: DetectionConfig | None = None
        self.reload_calls = 0

    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
        update_tracker: bool = True,
    ) -> HybridBoardResult:
        self.calls.append({
            "image": image.copy(),
            "captured_ns": captured_ns,
            "source_sequence": source_sequence,
            "include_debug": include_debug,
            "update_tracker": update_tracker,
        })
        result = self.results.popleft()
        if isinstance(result, BaseException):
            raise result
        return replace(
            result, timestamp_ns=captured_ns, source_sequence=source_sequence,
            debug_images={"edges": np.full((3, 4), 7, dtype=np.uint8)} if include_debug else {},
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def apply_config(self, config: DetectionConfig) -> None:
        self.config = config

    def reload_model(self) -> None:
        self.reload_calls += 1


class RepeatingHybridDetector(HybridDetectorFake):
    def __init__(self, result: HybridBoardResult) -> None:
        super().__init__([result])
        self.result = result

    def detect(self, image: np.ndarray, *, captured_ns: int, source_sequence: int, include_debug: bool = False, update_tracker: bool = True) -> HybridBoardResult:
        if not self.results:
            self.results.append(self.result)
        return super().detect(
            image, captured_ns=captured_ns, source_sequence=source_sequence,
            include_debug=include_debug, update_tracker=update_tracker,
        )


class FailingReloadHybridDetector(RepeatingHybridDetector):
    def __init__(self, result: HybridBoardResult) -> None:
        super().__init__(result)
        self.model_state = "READY"
        self.model_backend = "onnx"
        self.model_path = "target.onnx"
        self.model_errors: tuple[str, ...] = ()

    def reload_model(self) -> None:
        self.reload_calls += 1
        self.model_state = "UNAVAILABLE"
        self.model_backend = "classical-diagnostic"
        self.model_path = None
        self.model_errors = ("replacement.onnx: load failed",)
        raise RuntimeError(self.model_errors[0])


class BlockingOpenCamera(FakeCamera):
    def __init__(self, items: Iterable[Frame | BaseException]) -> None:
        super().__init__(items)
        self.open_started = threading.Event()
        self.allow_open = threading.Event()

    def open(self) -> None:
        self.open_started.set()
        assert self.allow_open.wait(1.0)
        super().open()


class QueuedStartOperationLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_waiting = threading.Event()
        self.allow_start = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "queued-start":
            self.start_waiting.set()
            assert self.allow_start.wait(1.0)
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._lock.release()

    def acquire(self, *, timeout: float = -1.0) -> bool:
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()


def frame(sequence: int, value: int | None = None) -> Frame:
    pixel = sequence if value is None else value
    return Frame(sequence=sequence, captured_ns=time.monotonic_ns(), image=np.full((8, 10, 3), pixel, dtype=np.uint8))


def observation(captured_ns: int) -> BoardObservation:
    return BoardObservation(captured_ns=captured_ns, corners_px=((1.0, 1.0), (8.0, 1.0), (8.0, 6.0), (1.0, 6.0)), center_px=(4.5, 3.5), confidence=0.9, homography_valid=True)


def config() -> CameraConfig:
    return CameraConfig(width=10, height=8, acquisition_fps=60, exposure_us=800, gain_db=6.0)


def parameters(base: CameraConfig | None = None, **changes: object) -> EditableCameraParameters:
    return replace(EditableCameraParameters.from_camera_config(base or config()), **changes)


def make_service(factory: FakeFactory, *, detector: FakeDetector | None = None, diagnostics_fps: float = 100.0, detection_fps: float = 100.0, confirm_timeout_s: float = 0.05) -> CameraTuningService:
    return CameraTuningService(camera_factory=factory, base_config=config(), detector=detector, camera_identity=CameraIdentity(model="fake", serial="serial-1"), read_timeout_ms=2, confirm_timeout_s=confirm_timeout_s, diagnostics_fps=diagnostics_fps, detection_fps=detection_fps, disconnect_timeout_threshold=1000)


def hybrid_result(sequence: int = 9) -> HybridBoardResult:
    model = BoardSearchResult(xyxy_px=(1.0, 2.0, 9.0, 7.0), confidence=0.91)
    geometry = GeometryResult(
        accepted=True, corners_px=((1.0, 1.0), (8.0, 1.0), (8.0, 6.0), (1.0, 6.0)),
        center_px=(4.5, 3.5), geometry_score=0.82, edge_support_score=0.73,
        structure_score=0.64, roi_xyxy_px=(0, 0, 10, 8), failure_reason=None,
    )
    candidate = CandidateEvaluation(model=model, geometry=geometry, temporal_score=0.72, combined_score=0.83)
    return HybridBoardResult(
        timestamp_ns=123, source_sequence=sequence, detected=True, target_valid=True,
        tracking_state="TRACKING", model_state="READY", model_backend="onnx",
        model_path="target.onnx", model_confidence=0.91, geometry_score=0.82,
        edge_support_score=0.73, structure_score=0.64, temporal_score=0.72,
        combined_score=0.83, candidate_count=1, corners_px=geometry.corners_px,
        center_px=geometry.center_px, failure_reason=None, inference_ms=2.5,
        geometry_ms=1.25, total_ms=4.0, homography_valid=True, target_x_mm=12.5,
        target_y_mm=-7.0, candidates=(candidate,),
    )


def updated_detection_config() -> DetectionConfig:
    current = DetectionConfig()
    return replace(current, tracking=replace(current.tracking, confirm_frames=current.tracking.confirm_frames + 1))


def wait_until(predicate, *, timeout_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before timeout")


def test_acquisition_keeps_only_latest_frame_counts_gaps_and_timeouts_and_copies_images() -> None:
    camera = FakeCamera([frame(1), TimeoutError("miss"), frame(4)])
    service = make_service(FakeFactory([camera]))
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().frame_count >= 2)
        latest = service.latest_frame()
        assert latest is not None and latest.sequence == 4 and np.all(latest.image == 4)
        latest.image[:] = 99
        assert np.all(service.latest_frame().image == 4)
        runtime = service.runtime_snapshot()
        assert runtime.sequence_gap_count == 2
        assert runtime.timeout_count >= 1
        assert runtime.state == "Connected"
    finally:
        service.stop()


def test_detection_disable_clears_result_and_reenable_uses_latest_frame() -> None:
    first = frame(1)
    detected = observation(first.captured_ns)
    detector = FakeDetector([detected, detected])
    service = make_service(FakeFactory([FakeCamera([first, frame(2), frame(3)])]), detector=detector)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().detected)
        service.set_detection_enabled(False)
        cleared = service.latest_detection()
        assert cleared.enabled is False and cleared.detected is False
        assert cleared.source_sequence is None and cleared.observation is None and cleared.error is None
        calls_when_disabled = len(detector.calls)
        time.sleep(0.03)
        assert len(detector.calls) == calls_when_disabled
        service.set_detection_enabled(True)
        wait_until(lambda: len(detector.calls) > calls_when_disabled)
        assert service.latest_detection().enabled is True
    finally:
        service.stop()


def test_detector_error_is_reported_without_stopping_acquisition() -> None:
    first, second = frame(1), frame(2)
    detector = FakeDetector([RuntimeError("detector boom")])
    camera = FakeCamera([first])
    service = make_service(FakeFactory([camera]), detector=detector, diagnostics_fps=10.0, detection_fps=10.0)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().error is not None)
        with camera._lock:
            camera.items.append(second)
        detector.results.append(observation(second.captured_ns))
        wait_until(lambda: service.latest_detection().detected)
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()


def test_service_publishes_hybrid_snapshot_and_candidate_geometry() -> None:
    detector = RepeatingHybridDetector(hybrid_result())
    service = make_service(FakeFactory([FakeCamera([frame(9)])]), detector=detector, detection_fps=1000.0)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().source_sequence == 9)
        snapshot = service.latest_detection()
        assert snapshot.target_valid is True
        assert snapshot.tracking_state == "TRACKING"
        assert snapshot.model_backend == "onnx"
        assert snapshot.combined_score == pytest.approx(0.83)
        assert snapshot.temporal_score == pytest.approx(0.72)
        assert snapshot.homography_valid is True
        assert snapshot.target_x_mm == pytest.approx(12.5)
        assert snapshot.target_y_mm == pytest.approx(-7.0)
        assert snapshot.corners_px == ((1.0, 1.0), (8.0, 1.0), (8.0, 6.0), (1.0, 6.0))
        assert snapshot.center_px == (4.5, 3.5)
        assert snapshot.observation is not None and snapshot.observation.confidence == pytest.approx(0.83)
        assert snapshot.candidates[0].xyxy_px == (1.0, 2.0, 9.0, 7.0)
        assert snapshot.candidates[0].accepted is True
        assert snapshot.candidates[0].model_confidence == pytest.approx(0.91)
        assert detector.calls[0]["include_debug"] is False
        assert detector.calls[0]["update_tracker"] is True
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()


def test_detection_frame_and_debug_are_latest_only_cached_and_tracker_safe() -> None:
    detector = RepeatingHybridDetector(hybrid_result())
    service = make_service(FakeFactory([FakeCamera([frame(9)])]), detector=detector, detection_fps=1000.0)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().source_sequence == 9)
        pair = service.detection_frame_for_latest(expected_sequence=9)
        assert pair is not None and pair[0].sequence == pair[1].source_sequence == 9
        pair[0].image[:] = 0
        assert np.all(service.detection_frame_for_latest(expected_sequence=9)[0].image == 9)
        debug = service.detection_debug_for_latest(expected_sequence=9)
        assert debug is not None and np.all(debug.images["edges"] == 7)
        assert detector.calls[-1]["include_debug"] is True
        assert detector.calls[-1]["update_tracker"] is False
        calls_after_debug = len(detector.calls)
        assert service.detection_debug_for_latest(expected_sequence=9) is debug
        assert len(detector.calls) == calls_after_debug
        with pytest.raises(RuntimeError) as caught:
            service.detection_frame_for_latest(expected_sequence=8)
        assert caught.value.expected == 8 and caught.value.actual == 9
    finally:
        service.stop()


def test_capture_snapshot_inspects_newer_captured_frame_not_stale_detection_frame() -> None:
    detector = RepeatingHybridDetector(hybrid_result())
    camera = FakeCamera([frame(9)])
    service = make_service(
        FakeFactory([camera]), detector=detector, detection_fps=1000.0
    )
    service.start()
    try:
        wait_until(lambda: service.latest_detection().source_sequence == 9)
        assert service._stop_analysis_workers()
        with camera._lock:
            camera.items.append(frame(10))
        wait_until(lambda: service.latest_frame().sequence == 10)
        calls_before_capture = len(detector.calls)

        snapshot = service.capture_snapshot()

        assert snapshot.frame.sequence == 10
        assert snapshot.detection.source_sequence == 9
        assert snapshot.detection_debug is not None
        assert snapshot.detection_debug.source_sequence == 10
        assert len(detector.calls) == calls_before_capture + 1
        assert detector.calls[-1]["source_sequence"] == 10
        assert detector.calls[-1]["include_debug"] is True
        assert detector.calls[-1]["update_tracker"] is False
    finally:
        service.stop()


def test_capture_snapshot_inspects_its_copied_frame_for_tracker_safe_debug() -> None:
    detector = RepeatingHybridDetector(hybrid_result())
    service = make_service(
        FakeFactory([FakeCamera([frame(9)])]), detector=detector, detection_fps=1000.0
    )
    service.start()
    try:
        wait_until(lambda: service.latest_detection().source_sequence == 9)
        calls_before_capture = len(detector.calls)

        snapshot = service.capture_snapshot()

        assert snapshot.frame.sequence == 9
        assert snapshot.detection_debug is not None
        assert snapshot.detection_debug.source_sequence == snapshot.frame.sequence
        assert np.all(snapshot.detection_debug.images["edges"] == 7)
        assert len(detector.calls) == calls_before_capture + 1
        assert detector.calls[-1]["source_sequence"] == snapshot.frame.sequence
        assert detector.calls[-1]["include_debug"] is True
        assert detector.calls[-1]["update_tracker"] is False
        assert all(call["include_debug"] is False for call in detector.calls[:-1])
    finally:
        service.stop()


def test_detector_exception_publishes_model_error_and_camera_keeps_streaming() -> None:
    detector = HybridDetectorFake([RuntimeError("detector boom")])
    service = make_service(FakeFactory([FakeCamera([frame(1), frame(2)])]), detector=detector, detection_fps=1000.0)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().failure_reason == "MODEL_ERROR")
        snapshot = service.latest_detection()
        assert snapshot.target_valid is False and snapshot.detected is False
        assert snapshot.model_state == "ERROR" and snapshot.error == "detector boom"
        assert service.runtime_snapshot().frame_count > 0
        assert service.runtime_snapshot().state == "Connected"
        assert service.detection_frame_for_latest() is not None
    finally:
        service.stop()


def test_detection_config_update_and_model_reload_do_not_reopen_camera() -> None:
    camera = FakeCamera([frame(1)])
    detector = RepeatingHybridDetector(hybrid_result(sequence=1))
    initial = DetectionConfig()
    service = CameraTuningService(
        camera_factory=FakeFactory([camera]), base_config=config(), detector=detector,
        detection_config=initial, read_timeout_ms=2, confirm_timeout_s=0.05,
        diagnostics_fps=100.0, detection_fps=100.0, disconnect_timeout_threshold=1000,
    )
    service.start()
    try:
        updated = updated_detection_config()
        assert service.detection_config() == initial
        assert service.apply_detection_config(updated) == updated
        service.reload_detection_model()
        assert camera.close_count == 0
        assert detector.config == updated and detector.reload_calls == 1
    finally:
        service.stop()


def test_failed_model_reload_immediately_clears_published_target_and_retained_debug() -> None:
    camera = FakeCamera([frame(1)])
    detector = FailingReloadHybridDetector(hybrid_result(sequence=1))
    service = make_service(FakeFactory([camera]), detector=detector, detection_fps=1000.0)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().target_valid)
        assert service.detection_frame_for_latest() is not None
        assert service.detection_debug_for_latest() is not None

        with pytest.raises(RuntimeError, match="replacement.onnx"):
            service.reload_detection_model()

        snapshot = service.latest_detection()
        assert snapshot.enabled is True
        assert snapshot.detected is False
        assert snapshot.target_valid is False
        assert snapshot.source_sequence is None
        assert snapshot.observation is None
        assert snapshot.corners_px == ()
        assert snapshot.center_px is None
        assert snapshot.homography_valid is False
        assert snapshot.target_x_mm is None
        assert snapshot.target_y_mm is None
        assert snapshot.model_state == "UNAVAILABLE"
        assert snapshot.model_backend == "classical-diagnostic"
        assert snapshot.model_path is None
        assert snapshot.failure_reason == "MODEL_UNAVAILABLE"
        assert service.detection_frame_for_latest() is None
        assert service.detection_debug_for_latest() is None
        assert detector.model_errors == ("replacement.onnx: load failed",)
        assert camera.close_count == 0
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()


def test_model_reload_gate_blocks_old_model_worker_and_recovers_after_failure() -> None:
    reload_entered = threading.Event()
    release_reload = threading.Event()
    stale_detect_started = threading.Event()
    release_stale_detect = threading.Event()

    class ReloadWindowDetector(RepeatingHybridDetector):
        def detect(
            self,
            image: np.ndarray,
            *,
            captured_ns: int,
            source_sequence: int,
            include_debug: bool = False,
            update_tracker: bool = True,
        ) -> HybridBoardResult:
            if reload_entered.is_set() and not release_reload.is_set() and update_tracker:
                stale_detect_started.set()
                assert release_stale_detect.wait(2.0)
            return super().detect(
                image,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                include_debug=include_debug,
                update_tracker=update_tracker,
            )

        def reload_model(self) -> None:
            self.reload_calls += 1
            reload_entered.set()
            assert release_reload.wait(2.0)
            raise RuntimeError("replacement model failed")

    camera = FakeCamera([frame(1)])
    detector = ReloadWindowDetector(hybrid_result(sequence=1))
    service = make_service(
        FakeFactory([camera]), detector=detector, detection_fps=1000.0
    )
    reload_errors: list[BaseException] = []
    reload_thread: threading.Thread | None = None

    def reload() -> None:
        try:
            service.reload_detection_model()
        except BaseException as exc:
            reload_errors.append(exc)

    service.start()
    try:
        wait_until(lambda: service.latest_detection().target_valid)
        reload_thread = threading.Thread(target=reload)
        reload_thread.start()
        assert reload_entered.wait(0.5)

        # The old implementation schedules a new-generation worker task in the
        # window before the detector begins its own reload/epoch transition.
        if stale_detect_started.wait(0.5):
            release_stale_detect.set()
            wait_until(lambda: service.latest_detection().target_valid)

        snapshot = service.latest_detection()
        assert snapshot.detected is False
        assert snapshot.target_valid is False
        assert snapshot.source_sequence is None
        assert snapshot.observation is None
        assert service.detection_frame_for_latest() is None
        assert service.detection_debug_for_latest() is None
        assert camera.close_count == 0
        assert service.runtime_snapshot().state == "Connected"

        release_reload.set()
        reload_thread.join(1.0)
        assert not reload_thread.is_alive()
        assert len(reload_errors) == 1
        assert isinstance(reload_errors[0], RuntimeError)
        assert str(reload_errors[0]) == "replacement model failed"
        assert service._detection_reload_in_progress is False

        wait_until(lambda: service.latest_detection().target_valid)
        assert service.detection_frame_for_latest() is not None
        assert camera.close_count == 0
        assert service.runtime_snapshot().state == "Connected"
    finally:
        release_stale_detect.set()
        release_reload.set()
        if reload_thread is not None:
            reload_thread.join(1.0)
        service.stop()

def test_disabling_detection_camera_replacement_disconnect_and_shutdown_reset_tracker() -> None:
    detector = RepeatingHybridDetector(hybrid_result())
    service = make_service(FakeFactory([FakeCamera([frame(1)]), FakeCamera([frame(2)])]), detector=detector)
    service.start()
    try:
        before_disable = detector.reset_calls
        service.set_detection_enabled(False)
        assert detector.reset_calls == before_disable + 1
        before_replace = detector.reset_calls
        service.apply_parameters(parameters(exposure_us=1200.0))
        assert detector.reset_calls >= before_replace + 1
    finally:
        before_shutdown = detector.reset_calls
        service.stop()
        assert detector.reset_calls >= before_shutdown + 1


def test_camera_disconnect_resets_tracker() -> None:
    detector = RepeatingHybridDetector(hybrid_result(sequence=1))
    service = CameraTuningService(
        camera_factory=FakeFactory([FakeCamera([frame(1), TimeoutError("lost")])]),
        base_config=config(), detector=detector, read_timeout_ms=2, confirm_timeout_s=0.05,
        diagnostics_fps=100.0, detection_fps=100.0, disconnect_timeout_threshold=1,
    )
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().state == "Disconnected")
        assert detector.reset_calls >= 1
    finally:
        service.stop()


def test_stale_diagnostics_failure_does_not_pollute_newer_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ev_vision.tuning.service as service_module

    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[int] = []
    real_compute = service_module.compute_diagnostics

    def controlled_compute(image, *, source_sequence, computed_ns):
        calls.append(source_sequence)
        if source_sequence == 1:
            first_started.set()
            assert release_first.wait(1.0)
            raise RuntimeError("old diagnostics boom")
        return real_compute(
            image,
            source_sequence=source_sequence,
            computed_ns=computed_ns,
        )

    monkeypatch.setattr(service_module, "compute_diagnostics", controlled_compute)
    camera = FakeCamera([frame(1)])
    service = make_service(FakeFactory([camera]), diagnostics_fps=100.0)
    service.start()
    try:
        assert first_started.wait(0.5)
        with camera._lock:
            camera.items.append(frame(2))
        wait_until(
            lambda: service.latest_frame() is not None
            and service.latest_frame().sequence == 2
        )
        release_first.set()
        wait_until(
            lambda: service.latest_diagnostics() is not None
            and service.latest_diagnostics().source_sequence == 2
        )
        assert calls[:2] == [1, 2]
        assert service.runtime_snapshot().last_error is None
    finally:
        release_first.set()
        service.stop()


def test_slow_detection_does_not_block_diagnostics_or_acquisition() -> None:
    class SlowDetector(FakeDetector):
        def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
            time.sleep(0.08)
            return super().detect(image, captured_ns=captured_ns)

    service = make_service(FakeFactory([FakeCamera([frame(index) for index in range(1, 8)])]), detector=SlowDetector([None]))
    service.start()
    try:
        wait_until(lambda: service.latest_diagnostics() is not None, timeout_s=0.05)
        wait_until(lambda: service.runtime_snapshot().frame_count >= 7)
    finally:
        service.stop()


def test_invalid_candidate_is_rejected_before_camera_is_closed() -> None:
    camera = FakeCamera([frame(1)])
    factory = FakeFactory([camera])
    service = make_service(factory)
    service.start()
    try:
        with pytest.raises(ValueError, match="gain_db"):
            service.apply_parameters(parameters(gain_db=999.0))
        assert camera.close_count == 0
        assert len(factory.created) == 1
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()


def test_successful_apply_confirms_frame_before_publishing_candidate() -> None:
    original = FakeCamera([frame(1)])
    candidate_camera = FakeCamera([frame(100, value=42)])
    factory = FakeFactory([original, candidate_camera])
    service = make_service(factory)
    candidate = parameters(exposure_us=1500.0, gain_db=7.5, acquisition_fps=45.0)
    service.start()
    try:
        service.apply_parameters(candidate)
        assert service.applied_parameters() == candidate
        assert factory.configs[-1] == candidate.to_camera_config(config())
        assert service.latest_frame().sequence == 100
        assert service.runtime_snapshot().state == "Connected"
        assert service.runtime_snapshot().last_error is None
        assert original.close_count == 1 and candidate_camera.close_count == 0
    finally:
        service.stop()


def test_apply_failure_rolls_back_and_keeps_last_good_parameters() -> None:
    original = FakeCamera([frame(1)])
    failed_candidate = FakeCamera(open_error=RuntimeError("candidate rejected"))
    rollback = FakeCamera([frame(10)])
    service = make_service(FakeFactory([original, failed_candidate, rollback]))
    last_good = service.applied_parameters()
    service.start()
    try:
        with pytest.raises(ParameterApplyError, match="candidate rejected") as caught:
            service.apply_parameters(parameters(exposure_us=1600.0))
        assert caught.value.apply_error is not None and caught.value.rollback_error is None
        assert service.applied_parameters() == last_good
        runtime = service.runtime_snapshot()
        assert runtime.state == "Connected" and "rolled back" in runtime.last_error
        assert service.latest_frame().sequence == 10
        assert original.close_count == 1 and failed_candidate.close_count == 1
    finally:
        service.stop()


def test_rollback_failure_preserves_both_errors_and_disconnects() -> None:
    original = FakeCamera([frame(1)])
    failed_candidate = FakeCamera(open_error=RuntimeError("apply exploded"))
    failed_rollback = FakeCamera(open_error=RuntimeError("rollback exploded"))
    service = make_service(FakeFactory([original, failed_candidate, failed_rollback]))
    service.start()
    with pytest.raises(ParameterApplyError) as caught:
        service.apply_parameters(parameters(exposure_us=1700.0))
    error = caught.value
    assert "apply exploded" in str(error) and "rollback exploded" in str(error)
    assert str(error.apply_error) == "apply exploded" and str(error.rollback_error) == "rollback exploded"
    runtime = service.runtime_snapshot()
    assert runtime.state == "Disconnected"
    assert "apply exploded" in runtime.last_error and "rollback exploded" in runtime.last_error
    service.stop()


def test_snapshot_methods_are_isolated_and_preview_rate_is_recorded() -> None:
    service = make_service(FakeFactory([FakeCamera([frame(1, value=12)])]))
    service.start()
    try:
        wait_until(lambda: service.latest_diagnostics() is not None)
        service.record_preview_frame()
        time.sleep(0.002)
        service.record_preview_frame()
        snapshot = service.capture_snapshot(OverlayOptions(show_crosshair=False))
        snapshot.frame.image[:] = 77
        assert np.all(service.latest_frame().image == 12)
        assert snapshot.parameters == service.applied_parameters()
        assert snapshot.camera_config == service.applied_parameters().to_camera_config(config())
        assert snapshot.camera_identity.serial == "serial-1"
        assert snapshot.overlay_options.show_crosshair is False
        assert service.runtime_snapshot().preview_fps > 0.0
    finally:
        service.stop()


def test_stop_is_idempotent_closes_once_and_service_can_restart() -> None:
    first, second = FakeCamera([frame(1)]), FakeCamera([frame(2)])
    service = make_service(FakeFactory([first, second]))
    service.start()
    service.stop()
    service.stop()
    assert first.close_count == 1 and service.runtime_snapshot().state == "Stopped"
    service.start()
    try:
        assert service.latest_frame().sequence == 2
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()
    assert second.close_count == 1


def test_queued_start_cannot_consume_a_newer_completed_shutdown_intent() -> None:
    camera = FakeCamera([frame(7)])
    factory = FakeFactory([camera])
    service = make_service(factory)
    service.stop()
    initial_generation = service._shutdown_generation
    assert service._shutdown_requested

    operation_lock = QueuedStartOperationLock()
    service._operation_lock = operation_lock  # type: ignore[assignment]
    start_errors: list[BaseException] = []

    def queued_start() -> None:
        try:
            service.start()
        except BaseException as exc:
            start_errors.append(exc)

    start_thread = threading.Thread(
        target=queued_start, name="queued-start", daemon=True
    )
    start_thread.start()
    assert operation_lock.start_waiting.wait(0.5)

    service.stop()
    newer_generation = service._shutdown_generation
    assert newer_generation == initial_generation + 1
    assert service._shutdown_requested

    operation_lock.allow_start.set()
    start_thread.join(1.0)
    assert not start_thread.is_alive()
    assert len(start_errors) == 1
    assert isinstance(start_errors[0], RuntimeError)
    assert "cancelled by shutdown" in str(start_errors[0])
    assert factory.created == []
    assert service.runtime_snapshot().state == "Stopped"
    assert service._shutdown_requested
    assert service._shutdown_generation == newer_generation

    service.start()
    try:
        assert service.latest_frame() is not None
        assert service.latest_frame().sequence == 7
    finally:
        service.stop()
    assert camera.close_count == 1


def test_stop_is_bounded_and_cancels_apply_blocked_in_candidate_open() -> None:
    original = FakeCamera([frame(1)])
    candidate_camera = BlockingOpenCamera([frame(20)])
    replacement = FakeCamera([frame(30)])
    factory = FakeFactory([original, candidate_camera, replacement])
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.03,
        disconnect_timeout_threshold=1000,
    )
    service.start()
    apply_error: list[BaseException] = []

    def apply() -> None:
        try:
            service.apply_parameters(parameters(exposure_us=1800.0))
        except BaseException as exc:
            apply_error.append(exc)

    apply_thread = threading.Thread(target=apply, daemon=True)
    apply_thread.start()
    assert candidate_camera.open_started.wait(0.5)

    stop_returned = threading.Event()
    stop_started = time.monotonic()
    stop_thread = threading.Thread(
        target=lambda: (service.stop(), stop_returned.set()), daemon=True
    )
    stop_thread.start()
    try:
        assert stop_returned.wait(0.12), "stop waited without bound for operation lock"
        assert time.monotonic() - stop_started < 0.12
        assert apply_thread.is_alive()
        runtime = service.runtime_snapshot()
        assert runtime.state == "Disconnected"
        assert runtime.last_error is not None and "shutdown" in runtime.last_error
        shutdown_generation = service._shutdown_generation
        assert service._shutdown_requested
        with pytest.raises(RuntimeError, match="shutdown|cleanup|camera"):
            service.start()

        remaining = 0.16 - (time.monotonic() - stop_started)
        if remaining > 0:
            time.sleep(remaining)
    finally:
        candidate_camera.allow_open.set()

    apply_thread.join(1.0)
    stop_thread.join(1.0)
    assert not apply_thread.is_alive() and not stop_thread.is_alive()
    assert len(apply_error) == 1
    assert isinstance(apply_error[0], ParameterApplyError)
    assert candidate_camera.open_count == 1
    assert candidate_camera.read_count == 0
    assert candidate_camera.close_count == 1
    assert original.close_count == 1
    assert service._shutdown_requested
    assert service._shutdown_generation == shutdown_generation
    wait_until(
        lambda: service._camera is None
        and service._acquisition_thread is None
        and service._diagnostics_thread is None
        and service._detection_thread is None
        and service.runtime_snapshot().state == "Stopped"
    )

    service.start()
    try:
        assert service.latest_frame() is not None
        assert service.latest_frame().sequence == 30
    finally:
        service.stop()
    assert replacement.close_count == 1


def test_start_installed_before_stop_timeout_finishes_shutdown_without_second_stop() -> None:
    installed = FakeCamera([frame(10)])
    replacement = FakeCamera([frame(20)])
    factory = FakeFactory([installed, replacement])
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.03,
        disconnect_timeout_threshold=1000,
    )
    entered_after_install = threading.Event()
    release_install = threading.Event()
    original_install = service._install_session

    def gated_install(
        camera, confirming_frame, *, parameters, error, mutation_generation
    ):
        result = original_install(
            camera,
            confirming_frame,
            parameters=parameters,
            error=error,
            mutation_generation=mutation_generation,
        )
        entered_after_install.set()
        assert release_install.wait(1.0)
        return result

    service._install_session = gated_install  # type: ignore[method-assign]
    start_errors: list[BaseException] = []

    def start() -> None:
        try:
            service.start()
        except BaseException as exc:
            start_errors.append(exc)

    start_thread = threading.Thread(target=start, daemon=True)
    start_thread.start()
    assert entered_after_install.wait(0.5)

    started = time.monotonic()
    service.stop()
    assert time.monotonic() - started < 0.12
    assert service.runtime_snapshot().state == "Disconnected"

    release_install.set()
    start_thread.join(1.0)
    assert not start_thread.is_alive()
    assert len(start_errors) == 1
    assert isinstance(start_errors[0], RuntimeError)
    assert "shutdown" in str(start_errors[0])
    wait_until(
        lambda: service._camera is None
        and service._acquisition_thread is None
        and service._diagnostics_thread is None
        and service._detection_thread is None
        and service.runtime_snapshot().state == "Stopped",
        timeout_s=1.0,
    )
    assert installed.close_count == 1

    service.start()
    try:
        assert service.latest_frame() is not None
        assert service.latest_frame().sequence == 20
    finally:
        service.stop()
    assert replacement.close_count == 1


def test_apply_installed_before_stop_timeout_finishes_shutdown_without_second_stop() -> None:
    original = FakeCamera([frame(1)])
    candidate_camera = FakeCamera([frame(10)])
    replacement = FakeCamera([frame(20)])
    factory = FakeFactory([original, candidate_camera, replacement])
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.03,
        disconnect_timeout_threshold=1000,
    )
    service.start()
    entered_after_install = threading.Event()
    release_install = threading.Event()
    original_install = service._install_session

    def gated_install(
        camera, confirming_frame, *, parameters, error, mutation_generation
    ):
        result = original_install(
            camera,
            confirming_frame,
            parameters=parameters,
            error=error,
            mutation_generation=mutation_generation,
        )
        if camera is candidate_camera:
            entered_after_install.set()
            assert release_install.wait(1.0)
        return result

    service._install_session = gated_install  # type: ignore[method-assign]
    apply_errors: list[BaseException] = []

    def apply() -> None:
        try:
            service.apply_parameters(parameters(exposure_us=1700.0))
        except BaseException as exc:
            apply_errors.append(exc)

    apply_thread = threading.Thread(target=apply, daemon=True)
    apply_thread.start()
    assert entered_after_install.wait(0.5)

    started = time.monotonic()
    service.stop()
    assert time.monotonic() - started < 0.12
    assert service.runtime_snapshot().state == "Disconnected"

    release_install.set()
    apply_thread.join(1.0)
    assert not apply_thread.is_alive()
    assert len(apply_errors) == 1
    assert isinstance(apply_errors[0], ParameterApplyError)
    assert "shutdown" in str(apply_errors[0])
    wait_until(
        lambda: service._camera is None
        and service._acquisition_thread is None
        and service._diagnostics_thread is None
        and service._detection_thread is None
        and service.runtime_snapshot().state == "Stopped",
        timeout_s=1.0,
    )
    assert original.close_count == 1
    assert candidate_camera.close_count == 1

    service.start()
    try:
        assert service.latest_frame() is not None
        assert service.latest_frame().sequence == 20
    finally:
        service.stop()
    assert replacement.close_count == 1


def test_service_is_exported_from_tuning_package() -> None:
    from ev_vision.tuning import CameraTuningService as ExportedService
    from ev_vision.tuning import ParameterApplyError as ExportedApplyError

    assert ExportedService is CameraTuningService
    assert ExportedApplyError is ParameterApplyError


def test_blocked_detection_cannot_deadlock_stop_or_publish_into_restarted_session() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    class SessionDetector:
        def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                assert release_first.wait(1.0)
            return observation(captured_ns)

    first = FakeCamera([frame(1)])
    second = FakeCamera([frame(20)])
    service = make_service(FakeFactory([first, second]), detector=SessionDetector())
    service.start()
    assert first_started.wait(0.5)

    stop_thread = threading.Thread(target=service.stop)
    stop_thread.start()
    stop_thread.join(0.4)
    assert not stop_thread.is_alive()
    assert first.close_count == 1

    with pytest.raises(RuntimeError, match="analysis|shutdown|cleanup"):
        service.start()
    release_first.set()
    wait_until(lambda: service._detection_thread is None or not service._detection_thread.is_alive())

    service.start()
    try:
        wait_until(lambda: service.latest_detection().source_sequence == 20)
        time.sleep(0.03)
        assert service.latest_detection().source_sequence == 20
    finally:
        service.stop()


def test_service_can_restart_last_good_after_rollback_failure() -> None:
    original = FakeCamera([frame(1)])
    failed_candidate = FakeCamera(open_error=RuntimeError("apply exploded"))
    failed_rollback = FakeCamera(open_error=RuntimeError("rollback exploded"))
    recovered = FakeCamera([frame(30)])
    service = make_service(FakeFactory([original, failed_candidate, failed_rollback, recovered]))
    service.start()

    with pytest.raises(ParameterApplyError):
        service.apply_parameters(parameters(exposure_us=1700.0))
    assert service.runtime_snapshot().state == "Disconnected"

    service.start()
    try:
        assert service.runtime_snapshot().state == "Connected"
        assert service.latest_frame().sequence == 30
        assert service.applied_parameters() == parameters()
    finally:
        service.stop()


def test_capture_snapshot_is_atomic_across_frame_and_runtime() -> None:
    class SlowReadCamera(FakeCamera):
        def read(self, *, timeout_ms: int = 100) -> Frame:
            time.sleep(0.002)
            return super().read(timeout_ms=timeout_ms)

    camera = SlowReadCamera([frame(index) for index in range(1, 200)])
    service = make_service(FakeFactory([camera]))
    original_runtime_snapshot = service.runtime_snapshot

    def delayed_runtime_snapshot():
        time.sleep(0.02)
        return original_runtime_snapshot()

    service.runtime_snapshot = delayed_runtime_snapshot  # type: ignore[method-assign]
    service.start()
    try:
        snapshot = service.capture_snapshot()
        assert snapshot.frame.sequence == snapshot.runtime.frame_count
    finally:
        service.stop()


def test_non_increasing_camera_sequences_do_not_replace_latest_frame() -> None:
    service = make_service(FakeFactory([FakeCamera([frame(5), frame(4)])]))
    service.start()
    try:
        time.sleep(0.03)
        latest = service.latest_frame()
        assert latest is not None and latest.sequence == 5
        assert service.runtime_snapshot().frame_count == 1
    finally:
        service.stop()

class BlockingReadCamera:
    """Camera fake that exposes SDK-call overlap instead of serializing it away."""

    def __init__(self, confirming_frame: Frame) -> None:
        self.confirming_frame = confirming_frame
        self.open_count = 0
        self.close_count = 0
        self.read_count = 0
        self.opened = False
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self._confirm_pending = True
        self._calls_lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.close_during_read = False

    def _enter(self) -> None:
        with self._calls_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)

    def _exit(self) -> None:
        with self._calls_lock:
            self.active_calls -= 1

    def open(self) -> None:
        self._enter()
        try:
            self.open_count += 1
            self.opened = True
        finally:
            self._exit()

    def close(self) -> None:
        self._enter()
        try:
            self.close_count += 1
            if self.read_started.is_set() and not self.release_read.is_set():
                self.close_during_read = True
            self.opened = False
        finally:
            self._exit()

    def read(self, *, timeout_ms: int = 100) -> Frame:
        self._enter()
        try:
            self.read_count += 1
            if self._confirm_pending:
                self._confirm_pending = False
                return self.confirming_frame
            self.read_started.set()
            assert self.release_read.wait(2.0)
            raise TimeoutError("released blocked read")
        finally:
            self._exit()


class CloseFailCamera(FakeCamera):
    def __init__(self, items: Iterable[Frame | BaseException]) -> None:
        super().__init__(items)
        self.fail_close = True

    def close(self) -> None:
        self.close_count += 1
        if self.fail_close:
            raise RuntimeError("close exploded")
        self.opened = False


def test_blocked_read_makes_stop_bounded_without_concurrent_close_and_apply_is_rejected() -> None:
    blocked = BlockingReadCamera(frame(1))
    spare = FakeCamera([frame(2)])
    factory = FakeFactory([blocked, spare])  # type: ignore[list-item]
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.03,
    )
    service.start()
    assert blocked.read_started.wait(0.5)

    started = time.monotonic()
    service.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    runtime = service.runtime_snapshot()
    assert runtime.state == "Disconnected"
    assert runtime.last_error is not None and "did not stop" in runtime.last_error
    assert blocked.close_count == 0
    assert blocked.close_during_read is False
    assert blocked.max_active_calls == 1
    with pytest.raises(RuntimeError):
        service.apply_parameters(parameters(exposure_us=1500.0))
    assert len(factory.created) == 1

    blocked.release_read.set()
    wait_until(lambda: blocked.close_count == 1)
    assert blocked.close_during_read is False
    assert blocked.max_active_calls == 1


def test_apply_fails_boundedly_without_reopen_when_read_owner_is_stuck() -> None:
    blocked = BlockingReadCamera(frame(1))
    factory = FakeFactory([blocked, FakeCamera([frame(2)])])  # type: ignore[list-item]
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.03,
    )
    service.start()
    assert blocked.read_started.wait(0.5)
    try:
        started = time.monotonic()
        with pytest.raises(ParameterApplyError) as caught:
            service.apply_parameters(parameters(exposure_us=1500.0))
        assert time.monotonic() - started < 0.2
        assert caught.value.rollback_error is None
        assert service.runtime_snapshot().state == "Disconnected"
        assert len(factory.created) == 1
        assert blocked.close_count == 0
    finally:
        blocked.release_read.set()
        wait_until(lambda: blocked.close_count == 1)
        service.stop()


def test_close_failure_retains_handle_disconnects_and_prevents_reopen() -> None:
    camera = CloseFailCamera([frame(1)])
    factory = FakeFactory([camera, FakeCamera([frame(2)])])
    service = make_service(factory)
    service.start()
    service.stop()

    runtime = service.runtime_snapshot()
    assert runtime.state == "Disconnected"
    assert runtime.last_error is not None and "close exploded" in runtime.last_error
    assert camera.close_count == 1
    with pytest.raises(RuntimeError, match="camera"):
        service.start()
    assert len(factory.created) == 1


def test_analysis_workers_do_not_accumulate_or_run_detector_concurrently_across_applies() -> None:
    release = threading.Event()
    first_started = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum = 0

    class BlockingDetector:
        def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                first_started.set()
            try:
                assert release.wait(2.0)
                return observation(captured_ns)
            finally:
                with lock:
                    active -= 1

    before = sum(t.name.startswith("camera-tuning-d") for t in threading.enumerate())
    cameras = [FakeCamera([frame(index)]) for index in (1, 10, 20, 30)]
    service = make_service(FakeFactory(cameras), detector=BlockingDetector())
    service.start()
    assert first_started.wait(0.5)
    try:
        service.apply_parameters(parameters(exposure_us=1200.0))
        service.apply_parameters(parameters(exposure_us=1400.0))
        service.apply_parameters(parameters(exposure_us=1600.0))
        time.sleep(0.08)
        after = sum(t.name.startswith("camera-tuning-d") for t in threading.enumerate())
        assert maximum == 1
        assert after - before <= 2
    finally:
        release.set()
        service.stop()


def test_session_switch_atomically_clears_diagnostics_and_detection_for_same_sequence() -> None:
    detector = FakeDetector([observation(1), observation(2)])
    service = make_service(
        FakeFactory([FakeCamera([frame(7, 10)]), FakeCamera([frame(7, 20)])]),
        detector=detector,
        diagnostics_fps=1.0,
        detection_fps=1.0,
    )
    service.start()
    try:
        wait_until(lambda: service.latest_diagnostics() is not None)
        service.set_detection_enabled(False)
        service._diagnostics_period_s = 10.0
        service._stop_analysis_workers()
        service.apply_parameters(parameters(exposure_us=1200.0))
        assert service.latest_diagnostics() is None
        detection = service.latest_detection()
        assert detection.enabled is False
        assert detection.source_sequence is None
        assert detection.result_age_ms is None
    finally:
        service.stop()


def test_repeated_timeouts_disconnect_and_a_new_frame_recovers_state_and_error() -> None:
    camera = FakeCamera([frame(1), TimeoutError("one"), TimeoutError("two")])
    service = CameraTuningService(
        camera_factory=FakeFactory([camera]),
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        disconnect_timeout_threshold=2,
    )
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().state == "Disconnected")
        disconnected = service.runtime_snapshot()
        assert disconnected.timeout_count >= 2
        assert disconnected.last_error is not None and "timeouts" in disconnected.last_error
        with camera._lock:
            camera.items.append(frame(2))
        wait_until(lambda: service.runtime_snapshot().state == "Connected")
        assert service.runtime_snapshot().last_error is None
    finally:
        service.stop()


def test_transient_read_exception_is_cleared_by_next_successful_frame() -> None:
    camera = FakeCamera([frame(1), RuntimeError("temporary read failure")])
    service = make_service(FakeFactory([camera]))
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().state == "Disconnected")
        with camera._lock:
            camera.items.append(frame(2))
        wait_until(lambda: service.latest_frame() is not None and service.latest_frame().sequence == 2)
        runtime = service.runtime_snapshot()
        assert runtime.state == "Connected"
        assert runtime.last_error is None
    finally:
        service.stop()


def test_32bit_sequence_wrap_is_forward_and_counts_modular_gaps() -> None:
    camera = FakeCamera([frame(0xFFFFFFFE, 254), frame(1), frame(0), frame(1)])
    service = make_service(FakeFactory([camera]))
    service.start()
    try:
        wait_until(lambda: service.latest_frame() is not None and service.latest_frame().sequence == 1)
        runtime = service.runtime_snapshot()
        assert runtime.frame_count == 2
        assert runtime.sequence_gap_count == 2
    finally:
        service.stop()


def test_each_camera_frame_is_frozen_only_once_before_publication() -> None:
    service = make_service(FakeFactory([FakeCamera([frame(1), frame(2)])]))
    original = service._freeze_frame
    freeze_count = 0

    def counted(candidate: Frame) -> Frame:
        nonlocal freeze_count
        freeze_count += 1
        return original(candidate)

    service._freeze_frame = counted  # type: ignore[method-assign]
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().frame_count == 2)
        assert freeze_count == 2
    finally:
        service.stop()




class SharedMethodTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def exit(self) -> None:
        with self.lock:
            self.active -= 1


class TrackedCamera(FakeCamera):
    def __init__(self, tracker: SharedMethodTracker, items: Iterable[Frame | BaseException]) -> None:
        super().__init__(items)
        self.tracker = tracker

    def open(self) -> None:
        self.tracker.enter()
        try:
            super().open()
        finally:
            self.tracker.exit()

    def close(self) -> None:
        self.tracker.enter()
        try:
            super().close()
        finally:
            self.tracker.exit()

    def read(self, *, timeout_ms: int = 100) -> Frame:
        self.tracker.enter()
        try:
            return super().read(timeout_ms=timeout_ms)
        finally:
            self.tracker.exit()


def test_camera_sdk_methods_never_overlap_across_apply_sessions() -> None:
    tracker = SharedMethodTracker()
    original = TrackedCamera(tracker, [frame(1)])
    candidate = TrackedCamera(tracker, [frame(10)])
    service = make_service(FakeFactory([original, candidate]))
    service.start()
    try:
        service.apply_parameters(parameters(exposure_us=1300.0))
    finally:
        service.stop()
    assert tracker.maximum == 1
    assert original.open_count == original.close_count == 1
    assert candidate.open_count == candidate.close_count == 1


def test_disabled_detection_does_not_copy_latest_frame_for_detector_worker() -> None:
    detector = FakeDetector([])
    service = make_service(FakeFactory([FakeCamera([frame(1)])]), detector=detector)
    service.set_detection_enabled(False)
    original = service._copy_frame
    copying_threads: list[str] = []

    def tracked_copy(candidate: Frame) -> Frame:
        copying_threads.append(threading.current_thread().name)
        return original(candidate)

    service._copy_frame = tracked_copy  # type: ignore[method-assign]
    service.start()
    try:
        wait_until(lambda: service.latest_diagnostics() is not None)
        time.sleep(0.03)
        assert "camera-tuning-detection" not in copying_threads
        assert detector.calls == []
    finally:
        service.stop()


class TargetReturnPauseThread(threading.Thread):
    """Pause a real acquisition thread after its target has returned."""

    pause_next_acquisition = False
    target_returned = threading.Event()
    allow_thread_exit = threading.Event()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pause_after_target = (
            self.name == "camera-tuning-acquisition"
            and type(self).pause_next_acquisition
        )
        if self._pause_after_target:
            type(self).pause_next_acquisition = False

    def run(self) -> None:
        super().run()
        if self._pause_after_target:
            type(self).target_returned.set()
            assert type(self).allow_thread_exit.wait(2.0)


def test_deferred_cleanup_closes_after_real_thread_target_return_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_thread = threading.Thread
    TargetReturnPauseThread.pause_next_acquisition = True
    TargetReturnPauseThread.target_returned = threading.Event()
    TargetReturnPauseThread.allow_thread_exit = threading.Event()
    monkeypatch.setattr(threading, "Thread", TargetReturnPauseThread)
    blocked = BlockingReadCamera(frame(1))
    replacement = FakeCamera([frame(2)])
    service = CameraTuningService(
        camera_factory=FakeFactory([blocked, replacement]),
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.01,
    )
    service.start()
    assert blocked.read_started.wait(0.5)
    acquisition = service._acquisition_thread
    assert isinstance(acquisition, original_thread)

    assert service._camera_stop is not None
    service._camera_stop.set()
    blocked.release_read.set()
    assert TargetReturnPauseThread.target_returned.wait(0.5)
    assert acquisition.is_alive()

    started = time.monotonic()
    service.stop()
    assert time.monotonic() - started < 0.2
    assert blocked.close_count == 0

    TargetReturnPauseThread.allow_thread_exit.set()
    try:
        wait_until(lambda: blocked.close_count == 1)
        wait_until(
            lambda: service._camera is None
            and service._acquisition_thread is None
            and service._camera_stop is None
        )
        service.start()
        service.stop()
        wait_until(lambda: replacement.close_count == 1)
    finally:
        TargetReturnPauseThread.allow_thread_exit.set()
        service.stop()


class BlockingCloseCamera(BlockingReadCamera):
    def __init__(self, confirming_frame: Frame) -> None:
        super().__init__(confirming_frame)
        self.close_entered = threading.Event()
        self.release_close = threading.Event()
        self.active_close = 0
        self.max_active_close = 0

    def close(self) -> None:
        with self._calls_lock:
            self.close_count += 1
            self.active_calls += 1
            self.active_close += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.max_active_close = max(self.max_active_close, self.active_close)
        self.close_entered.set()
        try:
            assert self.release_close.wait(2.0)
            self.opened = False
        finally:
            with self._calls_lock:
                self.active_calls -= 1
                self.active_close -= 1


def test_stop_returns_within_deadline_while_normal_camera_close_is_blocked() -> None:
    blocked = BlockingCloseCamera(frame(1))
    replacement = FakeCamera([frame(2)])
    factory = FakeFactory([blocked, replacement])  # type: ignore[list-item]
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.02,
    )
    service.start()
    assert blocked.read_started.wait(0.5)
    assert service._camera_stop is not None
    service._camera_stop.set()
    blocked.release_read.set()
    wait_until(
        lambda: service._acquisition_thread is not None
        and not service._acquisition_thread.is_alive()
    )

    stop_returned = threading.Event()
    stop_thread = threading.Thread(
        target=lambda: (service.stop(), stop_returned.set()), daemon=True
    )
    stop_thread.start()
    assert blocked.close_entered.wait(0.5)
    try:
        assert stop_returned.wait(0.2), "stop blocked in camera.close()"
        stop_thread.join(0.1)
        assert not stop_thread.is_alive()
        runtime = service.runtime_snapshot()
        assert runtime.state == "Disconnected"
        assert runtime.last_error is not None and "close" in runtime.last_error
        assert blocked.close_count == 1
        assert service._camera is blocked
        with pytest.raises(RuntimeError, match="cleanup|camera"):
            service.start()
        with pytest.raises(RuntimeError):
            service.apply_parameters(parameters(exposure_us=1500.0))
        assert len(factory.created) == 1
    finally:
        blocked.release_close.set()
        stop_thread.join(1.0)

    wait_until(
        lambda: service._camera is None
        and service._camera_close_owner is None
        and service.runtime_snapshot().state == "Stopped"
    )
    assert blocked.close_count == 1
    service.start()
    service.stop()
    wait_until(lambda: replacement.close_count == 1)


def test_failed_camera_close_is_terminal_and_never_retried_by_concurrent_stop() -> None:
    camera = CloseFailCamera([frame(1)])
    replacement = FakeCamera([frame(2)])
    factory = FakeFactory([camera, replacement])
    service = make_service(factory)
    service.start()
    service.stop()

    first_error = service.runtime_snapshot().last_error
    assert camera.close_count == 1
    assert service._camera is camera
    assert service._camera_close_owner is camera
    assert first_error is not None and "close exploded" in first_error

    callers = [threading.Thread(target=service.stop, daemon=True) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(0.2)
        assert not caller.is_alive(), "repeated stop was not bounded"

    assert camera.close_count == 1
    assert service.runtime_snapshot().state == "Disconnected"
    assert service.runtime_snapshot().last_error == first_error
    assert service._camera is camera
    assert service._camera_close_owner is camera
    with pytest.raises(RuntimeError, match="camera"):
        service.start()
    with pytest.raises(RuntimeError):
        service.apply_parameters(parameters(exposure_us=1500.0))
    assert len(factory.created) == 1


def test_second_stop_does_not_duplicate_deferred_close_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    TargetReturnPauseThread.pause_next_acquisition = True
    TargetReturnPauseThread.target_returned = threading.Event()
    TargetReturnPauseThread.allow_thread_exit = threading.Event()
    monkeypatch.setattr(threading, "Thread", TargetReturnPauseThread)
    blocked = BlockingCloseCamera(frame(1))
    replacement = FakeCamera([frame(2)])
    service = CameraTuningService(
        camera_factory=FakeFactory([blocked, replacement]),
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        shutdown_timeout_s=0.01,
    )
    service.start()
    assert blocked.read_started.wait(0.5)
    assert service._camera_stop is not None
    service._camera_stop.set()
    blocked.release_read.set()
    assert TargetReturnPauseThread.target_returned.wait(0.5)

    service.stop()
    TargetReturnPauseThread.allow_thread_exit.set()
    assert blocked.close_entered.wait(0.5)
    assert blocked.close_count == 1

    started = time.monotonic()
    service.stop()
    assert time.monotonic() - started < 0.2
    assert blocked.close_count == 1
    assert blocked.max_active_close == 1

    blocked.release_close.set()
    try:
        wait_until(
            lambda: service._camera is None
            and service._camera_close_owner is None
            and service._acquisition_thread is None
            and service._camera_stop is None
        )
        assert blocked.close_count == 1
        assert blocked.max_active_close == 1
        service.start()
        service.stop()
        wait_until(lambda: replacement.close_count == 1)
    finally:
        TargetReturnPauseThread.allow_thread_exit.set()
        blocked.release_close.set()
        service.stop()


def analysis_thread_count() -> int:
    return sum(
        thread.name in {"camera-tuning-diagnostics", "camera-tuning-detection"}
        for thread in threading.enumerate()
    )


def test_stop_terminates_analysis_workers_and_restart_creates_only_one_pair() -> None:
    before = analysis_thread_count()
    service = make_service(FakeFactory([FakeCamera([frame(1)]), FakeCamera([frame(2)])]))
    service.start()
    wait_until(lambda: analysis_thread_count() == before + 2)
    service.stop()
    wait_until(lambda: analysis_thread_count() == before)

    service.start()
    try:
        wait_until(lambda: analysis_thread_count() == before + 2)
        assert analysis_thread_count() == before + 2
    finally:
        service.stop()
    wait_until(lambda: analysis_thread_count() == before)


def test_blocked_detector_worker_is_not_replaced_or_run_concurrently_on_restart() -> None:
    started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum = 0

    class BlockingDetector:
        def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                started.set()
            try:
                assert release.wait(2.0)
                return observation(captured_ns)
            finally:
                with lock:
                    active -= 1

    factory = FakeFactory([FakeCamera([frame(1)]), FakeCamera([frame(2)])])
    service = make_service(factory, detector=BlockingDetector())
    service.start()
    assert started.wait(0.5)
    service.stop()
    assert service.runtime_snapshot().state == "Stopped"
    with pytest.raises(RuntimeError, match="analysis|shutdown|cleanup"):
        service.start()
    assert len(factory.created) == 1
    assert maximum == 1

    old_diagnostics = service._diagnostics_thread
    old_detection = service._detection_thread
    assert old_diagnostics is not None and old_detection is not None
    release.set()
    wait_until(lambda: not old_diagnostics.is_alive() and not old_detection.is_alive())
    service.start()
    try:
        time.sleep(0.03)
        assert maximum == 1
    finally:
        service.stop()


def test_diagnostics_wakeup_is_not_lost_between_empty_check_and_wait() -> None:
    camera = BlockingOpenCamera([frame(1)])
    service = CameraTuningService(
        camera_factory=FakeFactory([camera]),
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=0.01,
        detection_fps=0.01,
        shutdown_timeout_s=0.05,
    )
    window_reached = threading.Event()
    allow_wait = threading.Event()
    original_wait = service._wait_for_analysis

    def pause_before_wait(
        stop_event: threading.Event,
        wakeup: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if (
            threading.current_thread().name == "camera-tuning-diagnostics"
            and not window_reached.is_set()
        ):
            window_reached.set()
            assert allow_wait.wait(2.0)
        original_wait(stop_event, wakeup, *args, **kwargs)

    service._wait_for_analysis = pause_before_wait  # type: ignore[method-assign]
    start_errors: list[BaseException] = []

    def start_service() -> None:
        try:
            service.start()
        except BaseException as exc:
            start_errors.append(exc)

    starter = threading.Thread(target=start_service)
    starter.start()
    try:
        assert window_reached.wait(0.5)
        camera.allow_open.set()
        starter.join(0.5)
        assert not starter.is_alive()
        assert start_errors == []
        allow_wait.set()
        wait_until(lambda: service.latest_diagnostics() is not None, timeout_s=0.2)
    finally:
        camera.allow_open.set()
        allow_wait.set()
        starter.join(0.5)
        service.stop()


def test_detection_wakeup_is_not_lost_between_empty_check_and_wait() -> None:
    camera = BlockingOpenCamera([frame(1)])
    detector = FakeDetector([observation(time.monotonic_ns())])
    service = CameraTuningService(
        camera_factory=FakeFactory([camera]),
        base_config=config(),
        detector=detector,
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=0.01,
        detection_fps=0.01,
        shutdown_timeout_s=0.05,
    )
    window_reached = threading.Event()
    allow_wait = threading.Event()
    original_wait = service._wait_for_analysis

    def pause_before_wait(
        stop_event: threading.Event,
        wakeup: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if (
            threading.current_thread().name == "camera-tuning-detection"
            and not window_reached.is_set()
        ):
            window_reached.set()
            assert allow_wait.wait(2.0)
        original_wait(stop_event, wakeup, *args, **kwargs)

    service._wait_for_analysis = pause_before_wait  # type: ignore[method-assign]
    start_errors: list[BaseException] = []

    def start_service() -> None:
        try:
            service.start()
        except BaseException as exc:
            start_errors.append(exc)

    starter = threading.Thread(target=start_service)
    starter.start()
    try:
        assert window_reached.wait(0.5)
        camera.allow_open.set()
        starter.join(0.5)
        assert not starter.is_alive()
        assert start_errors == []
        allow_wait.set()
        wait_until(
            lambda: service.latest_detection().source_sequence == 1,
            timeout_s=0.2,
        )
    finally:
        camera.allow_open.set()
        allow_wait.set()
        starter.join(0.5)
        service.stop()


def test_low_fps_analysis_workers_exit_immediately_after_blocked_detector_returns() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingDetector:
        def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
            started.set()
            assert release.wait(2.0)
            return observation(captured_ns)

    factory = FakeFactory([FakeCamera([frame(1)]), FakeCamera([frame(2)])])
    service = CameraTuningService(
        camera_factory=factory,
        base_config=config(),
        detector=BlockingDetector(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=0.01,
        detection_fps=0.01,
        shutdown_timeout_s=0.01,
        disconnect_timeout_threshold=1000,
    )
    service.start()
    assert started.wait(0.5)
    old_diagnostics = service._diagnostics_thread
    old_detection = service._detection_thread
    assert old_diagnostics is not None and old_detection is not None

    service.stop()
    assert old_detection.is_alive()
    release.set()
    wait_until(
        lambda: not old_diagnostics.is_alive()
        and not old_detection.is_alive()
        and service._camera is None
        and service._camera_close_owner is None
        and service._acquisition_thread is None
        and service.runtime_snapshot().state == "Stopped",
        timeout_s=0.2,
    )

    service.start()
    try:
        assert service._diagnostics_thread is not old_diagnostics
        assert service._detection_thread is not old_detection
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()


def test_stale_detection_exception_after_disable_does_not_publish_error() -> None:
    started = threading.Event()
    release = threading.Event()
    detector_returned = threading.Event()

    class FailingDetector:
        def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
            started.set()
            try:
                assert release.wait(2.0)
                raise RuntimeError("stale detector failure")
            finally:
                detector_returned.set()

    service = make_service(
        FakeFactory([FakeCamera([frame(1)])]), detector=FailingDetector()
    )
    service.start()
    assert started.wait(0.5)
    service.set_detection_enabled(False)
    release.set()
    try:
        assert detector_returned.wait(0.5)
        detection = service.latest_detection()
        assert detection.enabled is False
        assert detection.detected is False
        assert detection.source_sequence is None
        assert detection.error is None
        assert service.runtime_snapshot().last_error is None
    finally:
        service.stop()


def test_timeout_streak_does_not_carry_into_successfully_applied_camera() -> None:
    original = FakeCamera([frame(1), TimeoutError("old timeout")])
    candidate_camera = FakeCamera([frame(10), TimeoutError("new timeout")])
    service = CameraTuningService(
        camera_factory=FakeFactory([original, candidate_camera]),
        base_config=config(),
        read_timeout_ms=2,
        confirm_timeout_s=0.05,
        diagnostics_fps=100.0,
        detection_fps=100.0,
        disconnect_timeout_threshold=2,
    )
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().timeout_count >= 1)
        service.apply_parameters(parameters(exposure_us=1200.0))
        wait_until(lambda: service.runtime_snapshot().timeout_count >= 2)
        assert service.runtime_snapshot().state == "Connected"
        assert service.runtime_snapshot().last_error is None
    finally:
        service.stop()


def test_first_rollback_session_frame_does_not_clear_rollback_message() -> None:
    original = FakeCamera([frame(1), RuntimeError("old read fault")])
    failed_candidate = FakeCamera(open_error=RuntimeError("candidate rejected"))
    rollback = FakeCamera([frame(10), frame(11)])
    service = make_service(FakeFactory([original, failed_candidate, rollback]))
    service.start()
    wait_until(lambda: service.runtime_snapshot().state == "Disconnected")

    with pytest.raises(ParameterApplyError):
        service.apply_parameters(parameters(exposure_us=1400.0))
    wait_until(lambda: service.latest_frame() is not None and service.latest_frame().sequence == 11)
    runtime = service.runtime_snapshot()
    assert runtime.state == "Connected"
    assert runtime.last_error is not None and "apply failed and rolled back" in runtime.last_error
    service.stop()


def test_apply_commits_candidate_parameters_and_frame_atomically() -> None:
    original = FakeCamera([frame(1, 1)])
    candidate_camera = FakeCamera([frame(20, 20)])
    service = make_service(FakeFactory([original, candidate_camera]))
    candidate = parameters(exposure_us=1500.0)
    service.start()

    entered_install = threading.Event()
    allow_install = threading.Event()
    original_install = service._install_session

    def gated_install(
        camera, confirming_frame, *, parameters, error, mutation_generation
    ):
        entered_install.set()
        assert allow_install.wait(1.0)
        return original_install(
            camera,
            confirming_frame,
            parameters=parameters,
            error=error,
            mutation_generation=mutation_generation,
        )

    service._install_session = gated_install  # type: ignore[method-assign]
    apply_errors: list[BaseException] = []

    def apply() -> None:
        try:
            service.apply_parameters(candidate)
        except BaseException as exc:
            apply_errors.append(exc)

    apply_thread = threading.Thread(target=apply)
    apply_thread.start()
    assert entered_install.wait(0.5)
    observed = service.capture_snapshot()
    allow_install.set()
    apply_thread.join(1.0)
    try:
        assert apply_errors == []
        assert not (
            observed.parameters == candidate and observed.frame.sequence == 1
        )
        final = service.capture_snapshot()
        assert final.parameters == candidate
        assert final.frame.sequence == 20
    finally:
        service.stop()


def test_reconfigurable_camera_applies_repeated_parameters_without_closing_handle() -> None:
    camera = ReconfigurableFakeCamera([frame(1)])
    factory = FakeFactory([camera])
    service = make_service(factory)
    service.start()
    try:
        for exposure_us in (800.0, 1000.0, 1200.0, 1500.0, 2000.0, 3000.0):
            applied = service.apply_parameters(parameters(exposure_us=exposure_us))
            assert applied.exposure_us == exposure_us

        assert len(factory.created) == 1
        assert camera.close_count == 0
        assert [item.exposure_us for item in camera.reconfigure_calls] == [
            800.0,
            1000.0,
            1200.0,
            1500.0,
            2000.0,
            3000.0,
        ]
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()
    assert camera.close_count == 1


def test_reconfigurable_camera_rolls_back_in_place_after_candidate_failure() -> None:
    candidate_error = RuntimeError("candidate node rejected")
    camera = ReconfigurableFakeCamera(
        [frame(1)],
        reconfigure_results=[candidate_error, None],
    )
    factory = FakeFactory([camera])
    service = make_service(factory)
    service.start()
    try:
        with pytest.raises(ParameterApplyError, match="rolled back") as captured:
            service.apply_parameters(parameters(exposure_us=1600.0))

        assert captured.value.apply_error is candidate_error
        assert captured.value.rollback_error is None
        assert len(factory.created) == 1
        assert camera.close_count == 0
        assert [item.exposure_us for item in camera.reconfigure_calls] == [1600.0, 800.0]
        assert service.applied_parameters().exposure_us == 800.0
        assert service.runtime_snapshot().state == "Connected"
    finally:
        service.stop()


class ClassicalDetectorFake:
    model_state = "READY"
    model_backend = "classical"
    model_path = None

    def __init__(self) -> None:
        self.config = DetectionConfig(backend="classical")
        self.reload_calls = 0
        self.reset_calls = 0

    def detect(
        self, image: np.ndarray, *, captured_ns: int, source_sequence: int,
        include_debug: bool = False, update_tracker: bool = True,
    ) -> ClassicalBoardResult:
        return ClassicalBoardResult(
            timestamp_ns=captured_ns, source_sequence=source_sequence,
            detected=False, target_valid=False, tracking_state="SEARCHING",
            observation_source=ObservationSource.NONE, confidence=0.0,
            center_px=None, debug_images={
                "normalized-gray": np.zeros(image.shape[:2], dtype=np.uint8)
            } if include_debug else {},
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def apply_config(self, config: DetectionConfig) -> None:
        self.config = config

    def reload_model(self) -> None:
        self.reload_calls += 1


def test_detector_config_update_does_not_replace_camera() -> None:
    camera = FakeCamera([frame(index) for index in range(1, 200)])
    detector = ClassicalDetectorFake()
    service = CameraTuningService(
        camera_factory=FakeFactory([camera]), base_config=config(),
        detector=detector, detection_config=detector.config,
        read_timeout_ms=2, confirm_timeout_s=0.05,
        diagnostics_fps=100.0, detection_fps=100.0,
        disconnect_timeout_threshold=1000,
    )
    service.start()
    try:
        wait_until(lambda: service.runtime_snapshot().frame_count >= 1)
        original_camera = service.active_camera
        original_count = service.runtime_snapshot().frame_count
        changed = replace(
            service.detection_config(),
            rings=RingGeometryConfig(ratio_tolerance=0.20),
        )
        service.apply_detection_config(changed)
        assert service.active_camera is original_camera
        wait_until(lambda: service.runtime_snapshot().frame_count > original_count)
        assert service.runtime_snapshot().state == "Connected"
        assert camera.close_count == 0
    finally:
        service.stop()


def test_classical_reload_is_clean_no_op() -> None:
    detector = ClassicalDetectorFake()
    service = make_service(
        FakeFactory([FakeCamera([frame(1)])]), detector=detector  # type: ignore[arg-type]
    )
    service.reload_detection_model()
    status = service.latest_detection()
    assert status.model_state == "READY"
    assert status.model_backend == "classical"
    assert detector.reload_calls == 0


def test_classical_result_maps_generic_snapshot_and_debug() -> None:
    source = frame(42)
    result = ClassicalBoardResult(
        timestamp_ns=source.captured_ns, source_sequence=source.sequence,
        detected=True, target_valid=True, tracking_state="PREDICTING",
        observation_source=ObservationSource.PREDICTED, confidence=0.73,
        center_px=(5.0, 4.0), scale_px_per_mm=3.2,
        velocity_px_s=(12.0, -4.0), predicted_frames=2, source_age_us=8000,
        near_image_edge=True, partially_outside=True,
        debug_images={"normalized-gray": np.zeros((8, 10), dtype=np.uint8)},
    )
    snapshot = CameraTuningService._snapshot_for_result(result, source)
    assert snapshot.observation_source == "PREDICTED"
    assert snapshot.confidence == pytest.approx(0.73)
    assert snapshot.scale_px_per_mm == pytest.approx(3.2)
    assert snapshot.velocity_px_s == (12.0, -4.0)
    assert snapshot.predicted_frames == 2
    assert snapshot.source_age_us == 8000
    assert snapshot.near_image_edge is True
    assert snapshot.partially_outside is True
    assert snapshot.rejection_reasons == ()
    debug = CameraTuningService._debug_snapshot_for_result(result, source)
    assert debug is not None
    assert debug.source_sequence == 42
    assert set(debug.images) == {"normalized-gray"}
