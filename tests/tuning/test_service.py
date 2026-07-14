from __future__ import annotations

from collections import deque
from dataclasses import replace
import threading
import time
from typing import Iterable

import numpy as np
import pytest

from ev_vision.config import CameraConfig
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


class BlockingOpenCamera(FakeCamera):
    def __init__(self, items: Iterable[Frame | BaseException]) -> None:
        super().__init__(items)
        self.open_started = threading.Event()
        self.allow_open = threading.Event()

    def open(self) -> None:
        self.open_started.set()
        assert self.allow_open.wait(1.0)
        super().open()


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
    return CameraTuningService(camera_factory=factory, base_config=config(), detector=detector, camera_identity=CameraIdentity(model="fake", serial="serial-1"), read_timeout_ms=2, confirm_timeout_s=confirm_timeout_s, diagnostics_fps=diagnostics_fps, detection_fps=detection_fps)


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
    detector = FakeDetector([RuntimeError("detector boom"), observation(second.captured_ns)])
    service = make_service(FakeFactory([FakeCamera([first, second, frame(3)])]), detector=detector)
    service.start()
    try:
        wait_until(lambda: service.latest_detection().error is not None)
        wait_until(lambda: service.latest_detection().detected)
        assert service.runtime_snapshot().frame_count >= 3
        assert service.runtime_snapshot().state == "Connected"
    finally:
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


def test_stop_waits_for_apply_then_closes_new_camera_without_deadlock() -> None:
    original = FakeCamera([frame(1)])
    candidate_camera = BlockingOpenCamera([frame(20)])
    service = make_service(FakeFactory([original, candidate_camera]))
    service.start()
    apply_error: list[BaseException] = []

    def apply() -> None:
        try:
            service.apply_parameters(parameters(exposure_us=1800.0))
        except BaseException as exc:
            apply_error.append(exc)

    apply_thread = threading.Thread(target=apply)
    apply_thread.start()
    assert candidate_camera.open_started.wait(0.5)
    stop_thread = threading.Thread(target=service.stop)
    stop_thread.start()
    time.sleep(0.02)
    assert stop_thread.is_alive()
    candidate_camera.allow_open.set()
    apply_thread.join(1.0)
    stop_thread.join(1.0)
    assert not apply_thread.is_alive() and not stop_thread.is_alive()
    assert apply_error == []
    assert candidate_camera.close_count == 1
    assert service.runtime_snapshot().state == "Stopped"



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

    service.start()
    try:
        wait_until(lambda: service.latest_detection().source_sequence == 20)
        release_first.set()
        time.sleep(0.03)
        assert service.latest_detection().source_sequence == 20
    finally:
        release_first.set()
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
