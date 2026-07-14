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
    return CameraTuningService(camera_factory=factory, base_config=config(), detector=detector, camera_identity=CameraIdentity(model="fake", serial="serial-1"), read_timeout_ms=2, confirm_timeout_s=confirm_timeout_s, diagnostics_fps=diagnostics_fps, detection_fps=detection_fps, disconnect_timeout_threshold=1000)


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

    with pytest.raises(RuntimeError, match="analysis"):
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
    with pytest.raises(RuntimeError, match="analysis"):
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
        lambda: not old_diagnostics.is_alive() and not old_detection.is_alive(),
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

    def gated_install(camera, confirming_frame, *, parameters, error):
        entered_install.set()
        assert allow_install.wait(1.0)
        return original_install(
            camera, confirming_frame, parameters=parameters, error=error
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
