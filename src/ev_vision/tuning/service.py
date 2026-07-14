from __future__ import annotations

from collections import deque
from dataclasses import replace
import threading
import time
from typing import Callable, Protocol

import numpy as np

from ev_vision.config import CameraConfig
from ev_vision.models import BoardObservation, Frame
from ev_vision.tuning.diagnostics import compute_diagnostics
from ev_vision.tuning.models import (
    CameraIdentity,
    CaptureSnapshot,
    DetectionSnapshot,
    EditableCameraParameters,
    ImageDiagnostics,
    OverlayOptions,
    ParameterBounds,
    RuntimeSnapshot,
)


class CameraPort(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def read(self, *, timeout_ms: int = 100) -> Frame: ...


CameraFactory = Callable[[CameraConfig], CameraPort]


class DetectorPort(Protocol):
    def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None: ...


class ParameterApplyError(RuntimeError):
    """A candidate camera configuration failed, optionally along with rollback."""

    def __init__(
        self,
        message: str,
        *,
        apply_error: BaseException,
        rollback_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.apply_error = apply_error
        self.rollback_error = rollback_error


class _RateWindow:
    def __init__(self, *, window_s: float = 1.0) -> None:
        self._window_ns = int(window_s * 1_000_000_000)
        self._events: deque[int] = deque()

    def record(self, now_ns: int) -> None:
        self._events.append(now_ns)
        self._prune(now_ns)

    def value(self, now_ns: int) -> float:
        self._prune(now_ns)
        if not self._events:
            return 0.0
        if len(self._events) == 1:
            return 1.0
        elapsed_s = max((self._events[-1] - self._events[0]) / 1_000_000_000.0, 1e-9)
        return (len(self._events) - 1) / elapsed_s

    def _prune(self, now_ns: int) -> None:
        cutoff = now_ns - self._window_ns
        while self._events and self._events[0] < cutoff:
            self._events.popleft()


class CameraTuningService:
    """Own the camera and expose copied, latest-only tuning snapshots."""

    def __init__(
        self,
        *,
        camera_factory: CameraFactory,
        base_config: CameraConfig,
        detector: DetectorPort | None = None,
        bounds: ParameterBounds | None = None,
        camera_identity: CameraIdentity | None = None,
        read_timeout_ms: int = 100,
        confirm_timeout_s: float = 1.0,
        diagnostics_fps: float = 10.0,
        detection_fps: float = 15.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if read_timeout_ms <= 0:
            raise ValueError("read_timeout_ms must be positive")
        if confirm_timeout_s <= 0:
            raise ValueError("confirm_timeout_s must be positive")
        if diagnostics_fps <= 0 or detection_fps <= 0:
            raise ValueError("analysis rates must be positive")

        self._camera_factory = camera_factory
        self._base_config = base_config
        self._detector = detector
        self._bounds = bounds or ParameterBounds()
        self._camera_identity = camera_identity or CameraIdentity(
            model="", serial=""
        )
        self._read_timeout_ms = int(read_timeout_ms)
        self._confirm_timeout_s = float(confirm_timeout_s)
        self._diagnostics_period_s = 1.0 / float(diagnostics_fps)
        self._detection_period_s = 1.0 / float(detection_fps)
        self._clock_ns = clock_ns

        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._camera: CameraPort | None = None
        self._camera_stop: threading.Event | None = None
        self._acquisition_thread: threading.Thread | None = None
        self._diagnostics_thread: threading.Thread | None = None
        self._detection_thread: threading.Thread | None = None
        self._detection_wakeup: threading.Event | None = None
        self._session_generation = 0
        self._running = False

        self._state = "Stopped"
        self._last_error: str | None = None
        self._applied = EditableCameraParameters.from_camera_config(base_config)
        self._latest_frame: Frame | None = None
        self._latest_frame_received_ns: int | None = None
        self._latest_diagnostics: ImageDiagnostics | None = None
        self._latest_detection = DetectionSnapshot(
            enabled=detector is not None, detected=False
        )
        self._detection_enabled = detector is not None
        self._detection_generation = 0
        self._detection_computed_ns: int | None = None

        self._frame_count = 0
        self._timeout_count = 0
        self._sequence_gap_count = 0
        self._last_sequence: int | None = None
        self._acquisition_rate = _RateWindow()
        self._preview_rate = _RateWindow()
        self._diagnostics_rate = _RateWindow()
        self._detection_rate = _RateWindow()

    def start(self) -> None:
        with self._operation_lock:
            with self._lock:
                if self._running:
                    return
                self._state = "Starting"
                self._last_error = None

            try:
                camera, confirming_frame = self._open_and_confirm(
                    self._applied.to_camera_config(self._base_config)
                )
            except BaseException as exc:
                self._set_state("Disconnected", str(exc))
                raise

            with self._lock:
                self._camera = camera
                self._running = True
                self._publish_frame_locked(confirming_frame, reset_sequence=True)
                self._state = "Connected"
                self._last_error = None
            self._start_camera_threads(camera)

    def stop(self) -> None:
        with self._operation_lock:
            with self._lock:
                if self._state == "Stopped" and self._camera is None:
                    self._running = False
                    return
                self._running = False
            self._stop_camera_threads()
            close_error = self._close_camera()
            error = (
                None
                if close_error is None
                else f"camera close failed: {close_error}"
            )
            self._set_state("Stopped", error)

    def apply_parameters(
        self, candidate: EditableCameraParameters
    ) -> EditableCameraParameters:
        candidate = self._bounds.validate(candidate)
        with self._operation_lock:
            with self._lock:
                if not self._running:
                    raise RuntimeError("camera tuning service is not running")
                last_good = self._applied
                self._state = "Applying"
                self._last_error = None

            self._stop_camera_threads()
            close_error = self._close_camera()
            apply_error: BaseException | None = close_error
            if apply_error is None:
                try:
                    camera, confirming_frame = self._open_and_confirm(
                        candidate.to_camera_config(self._base_config)
                    )
                except BaseException as exc:
                    apply_error = exc

            if apply_error is not None:
                self._set_state("Recovering", str(apply_error))
                try:
                    rollback_camera, rollback_frame = self._open_and_confirm(
                        last_good.to_camera_config(self._base_config)
                    )
                except BaseException as rollback_error:
                    message = f"apply failed: {apply_error}; rollback failed: {rollback_error}"
                    with self._lock:
                        self._running = False
                        self._state = "Disconnected"
                        self._last_error = message
                    raise ParameterApplyError(
                        message,
                        apply_error=apply_error,
                        rollback_error=rollback_error,
                    ) from apply_error

                with self._lock:
                    self._camera = rollback_camera
                    self._publish_frame_locked(rollback_frame, reset_sequence=True)
                    self._state = "Connected"
                    self._last_error = f"apply failed and rolled back: {apply_error}"
                self._start_camera_threads(rollback_camera)
                raise ParameterApplyError(
                    f"apply failed and rolled back: {apply_error}",
                    apply_error=apply_error,
                ) from apply_error

            with self._lock:
                self._camera = camera
                self._applied = candidate
                self._publish_frame_locked(confirming_frame, reset_sequence=True)
                self._state = "Connected"
                self._last_error = None
            self._start_camera_threads(camera)
            return candidate

    def applied_parameters(self) -> EditableCameraParameters:
        with self._lock:
            return self._applied

    def runtime_snapshot(self) -> RuntimeSnapshot:
        now_ns = self._clock_ns()
        with self._lock:
            return self._runtime_snapshot_locked(now_ns)

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return None if self._latest_frame is None else self._copy_frame(self._latest_frame)

    def latest_diagnostics(self) -> ImageDiagnostics | None:
        with self._lock:
            return self._latest_diagnostics

    def latest_detection(self) -> DetectionSnapshot:
        now_ns = self._clock_ns()
        with self._lock:
            return self._detection_snapshot_locked(now_ns)

    def set_detection_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        if enabled and self._detector is None:
            raise RuntimeError("no detector is configured")
        with self._lock:
            if self._detection_enabled == enabled:
                return
            self._detection_enabled = enabled
            self._detection_generation += 1
            self._detection_computed_ns = None
            self._latest_detection = DetectionSnapshot(enabled=enabled, detected=False)
            wakeup = self._detection_wakeup
        if wakeup is not None:
            wakeup.set()

    def capture_snapshot(self, overlay_options: OverlayOptions | None = None) -> CaptureSnapshot:
        now_ns = self._clock_ns()
        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("no camera frame is available")
            parameters = self._applied
            return CaptureSnapshot(
                frame=self._copy_frame(self._latest_frame),
                parameters=parameters,
                runtime=self._runtime_snapshot_locked(now_ns),
                diagnostics=self._latest_diagnostics,
                detection=self._detection_snapshot_locked(now_ns),
                overlay_options=overlay_options or OverlayOptions(),
                camera_config=parameters.to_camera_config(self._base_config),
                camera_identity=self._camera_identity,
            )

    def record_preview_frame(self) -> None:
        now_ns = self._clock_ns()
        with self._lock:
            self._preview_rate.record(now_ns)

    def _open_and_confirm(self, config: CameraConfig) -> tuple[CameraPort, Frame]:
        camera = self._camera_factory(config)
        try:
            camera.open()
            deadline_ns = self._clock_ns() + int(self._confirm_timeout_s * 1_000_000_000)
            while True:
                try:
                    candidate = camera.read(timeout_ms=self._read_timeout_ms)
                    return camera, self._freeze_frame(candidate)
                except TimeoutError:
                    self._record_timeout()
                    if self._clock_ns() >= deadline_ns:
                        raise TimeoutError("timed out waiting for a confirming camera frame")
        except BaseException:
            try:
                camera.close()
            except BaseException:
                pass
            raise

    def _start_camera_threads(self, camera: CameraPort) -> None:
        stop_event = threading.Event()
        detection_wakeup = threading.Event()
        with self._lock:
            self._session_generation += 1
            session_generation = self._session_generation
        acquisition = threading.Thread(
            target=self._acquisition_loop,
            args=(camera, stop_event, session_generation),
            name="camera-tuning-acquisition",
            daemon=True,
        )
        diagnostics = threading.Thread(
            target=self._diagnostics_loop,
            args=(stop_event, session_generation),
            name="camera-tuning-diagnostics",
            daemon=True,
        )
        detection = threading.Thread(
            target=self._detection_loop,
            args=(stop_event, detection_wakeup, session_generation),
            name="camera-tuning-detection",
            daemon=True,
        )
        with self._lock:
            self._camera_stop = stop_event
            self._detection_wakeup = detection_wakeup
            self._acquisition_thread = acquisition
            self._diagnostics_thread = diagnostics
            self._detection_thread = detection
        acquisition.start()
        diagnostics.start()
        detection.start()

    def _stop_camera_threads(self) -> None:
        with self._lock:
            stop_event = self._camera_stop
            detection_wakeup = self._detection_wakeup
            acquisition = self._acquisition_thread
            analysis_threads = (self._diagnostics_thread, self._detection_thread)
            self._session_generation += 1
        if stop_event is not None:
            stop_event.set()
        if detection_wakeup is not None:
            detection_wakeup.set()
        current = threading.current_thread()
        if acquisition is not None and acquisition is not current:
            acquisition.join()
        for thread in analysis_threads:
            if thread is not None and thread is not current:
                thread.join(timeout=0.05)
        with self._lock:
            if self._camera_stop is stop_event:
                self._camera_stop = None
                self._detection_wakeup = None
                self._acquisition_thread = None
                self._diagnostics_thread = None
                self._detection_thread = None

    def _close_camera(self) -> BaseException | None:
        with self._lock:
            camera = self._camera
            self._camera = None
        if camera is None:
            return None
        try:
            camera.close()
        except BaseException as exc:
            return exc
        return None

    def _acquisition_loop(
        self, camera: CameraPort, stop_event: threading.Event, session_generation: int
    ) -> None:
        while not stop_event.is_set():
            try:
                camera_frame = camera.read(timeout_ms=self._read_timeout_ms)
            except TimeoutError:
                self._record_timeout()
                continue
            except BaseException as exc:
                if stop_event.is_set():
                    break
                self._record_worker_error(
                    "camera read", exc, disconnected=True, session_generation=session_generation
                )
                stop_event.wait(0.01)
                continue
            frozen = self._freeze_frame(camera_frame)
            with self._lock:
                if stop_event.is_set() or session_generation != self._session_generation:
                    break
                self._publish_frame_locked(frozen)

    def _diagnostics_loop(
        self, stop_event: threading.Event, session_generation: int
    ) -> None:
        last_sequence: int | None = None
        while not stop_event.is_set():
            started_ns = self._clock_ns()
            frame = self._analysis_frame(last_sequence)
            if frame is not None:
                try:
                    result = compute_diagnostics(
                        frame.image,
                        source_sequence=frame.sequence,
                        computed_ns=self._clock_ns(),
                    )
                except BaseException as exc:
                    self._record_worker_error(
                        "diagnostics", exc, session_generation=session_generation
                    )
                else:
                    now_ns = self._clock_ns()
                    with self._lock:
                        if session_generation != self._session_generation:
                            break
                        self._latest_diagnostics = result
                        self._diagnostics_rate.record(now_ns)
                    last_sequence = frame.sequence
            elapsed_s = (self._clock_ns() - started_ns) / 1_000_000_000.0
            stop_event.wait(max(0.0, self._diagnostics_period_s - elapsed_s))

    def _detection_loop(
        self,
        stop_event: threading.Event,
        detection_wakeup: threading.Event,
        session_generation: int,
    ) -> None:
        last_sequence: int | None = None
        last_generation = -1
        while not stop_event.is_set():
            started_ns = self._clock_ns()
            with self._lock:
                enabled = self._detection_enabled
                generation = self._detection_generation
                detector = self._detector
            frame = self._analysis_frame(None if generation != last_generation else last_sequence)
            if enabled and detector is not None and frame is not None:
                try:
                    result = detector.detect(frame.image, captured_ns=frame.captured_ns)
                except BaseException as exc:
                    now_ns = self._clock_ns()
                    with self._lock:
                        if (
                            session_generation == self._session_generation
                            and self._detection_enabled
                            and self._detection_generation == generation
                        ):
                            self._latest_detection = DetectionSnapshot(
                                enabled=True,
                                detected=False,
                                source_sequence=frame.sequence,
                                error=str(exc),
                            )
                            self._detection_computed_ns = now_ns
                    self._record_worker_error(
                        "detection", exc, session_generation=session_generation
                    )
                else:
                    now_ns = self._clock_ns()
                    with self._lock:
                        if (
                            session_generation == self._session_generation
                            and self._detection_enabled
                            and self._detection_generation == generation
                        ):
                            self._latest_detection = DetectionSnapshot(
                                enabled=True,
                                detected=result is not None,
                                source_sequence=frame.sequence,
                                observation=result,
                            )
                            self._detection_computed_ns = now_ns
                            self._detection_rate.record(now_ns)
                            last_sequence = frame.sequence
                            last_generation = generation
            detection_wakeup.clear()
            elapsed_s = (self._clock_ns() - started_ns) / 1_000_000_000.0
            detection_wakeup.wait(max(0.0, self._detection_period_s - elapsed_s))

    def _analysis_frame(self, after_sequence: int | None) -> Frame | None:
        with self._lock:
            if self._latest_frame is None or self._latest_frame.sequence == after_sequence:
                return None
            return self._copy_frame(self._latest_frame)

    def _publish_frame_locked(self, frame: Frame, *, reset_sequence: bool = False) -> None:
        frozen = self._freeze_frame(frame)
        now_ns = self._clock_ns()
        if reset_sequence:
            self._last_sequence = None
        if self._last_sequence is not None:
            if frozen.sequence <= self._last_sequence:
                return
            if frozen.sequence > self._last_sequence + 1:
                self._sequence_gap_count += frozen.sequence - self._last_sequence - 1
        self._last_sequence = frozen.sequence
        self._latest_frame = frozen
        self._latest_frame_received_ns = now_ns
        self._frame_count += 1
        self._acquisition_rate.record(now_ns)

    def _runtime_snapshot_locked(self, now_ns: int) -> RuntimeSnapshot:
        frame_age_ms = None
        if self._latest_frame_received_ns is not None:
            frame_age_ms = max(
                0.0, (now_ns - self._latest_frame_received_ns) / 1_000_000.0
            )
        return RuntimeSnapshot(
            state=self._state,
            acquisition_fps=self._acquisition_rate.value(now_ns),
            preview_fps=self._preview_rate.value(now_ns),
            detection_fps=self._detection_rate.value(now_ns),
            diagnostics_fps=self._diagnostics_rate.value(now_ns),
            frame_count=self._frame_count,
            timeout_count=self._timeout_count,
            sequence_gap_count=self._sequence_gap_count,
            frame_age_ms=frame_age_ms,
            last_error=self._last_error,
        )

    def _detection_snapshot_locked(self, now_ns: int) -> DetectionSnapshot:
        result = self._latest_detection
        age_ms = None
        if (
            self._detection_computed_ns is not None
            and result.source_sequence is not None
        ):
            age_ms = max(
                0.0, (now_ns - self._detection_computed_ns) / 1_000_000.0
            )
        return replace(result, result_age_ms=age_ms)

    def _record_timeout(self) -> None:
        with self._lock:
            self._timeout_count += 1

    def _record_worker_error(
        self,
        worker: str,
        error: BaseException,
        *,
        disconnected: bool = False,
        session_generation: int | None = None,
    ) -> None:
        with self._lock:
            if (
                session_generation is not None
                and session_generation != self._session_generation
            ):
                return
            if disconnected:
                self._state = "Disconnected"
            self._last_error = f"{worker} failed: {error}"

    def _set_state(self, state: str, error: str | None) -> None:
        with self._lock:
            self._state = state
            self._last_error = error

    @staticmethod
    def _freeze_frame(frame: Frame) -> Frame:
        image = np.array(frame.image, copy=True)
        image.setflags(write=False)
        return Frame(
            sequence=int(frame.sequence),
            captured_ns=int(frame.captured_ns),
            image=image,
        )

    @staticmethod
    def _copy_frame(frame: Frame) -> Frame:
        return Frame(
            sequence=frame.sequence,
            captured_ns=frame.captured_ns,
            image=np.array(frame.image, copy=True),
        )


TuningService = CameraTuningService


__all__ = [
    "CameraFactory",
    "CameraPort",
    "CameraTuningService",
    "DetectorPort",
    "ParameterApplyError",
    "TuningService",
]
