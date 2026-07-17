from __future__ import annotations

from collections import deque
from dataclasses import replace
import inspect
import threading
import time
from typing import Callable, Mapping, Protocol

import numpy as np

from ev_vision.config import CameraConfig, DetectionConfig
from ev_vision.detection.contracts import ClassicalBoardResult, ClassicalCandidateEvaluation
from ev_vision.detection.failures import CandidateEvaluation, HybridBoardResult
from ev_vision.models import BoardObservation, Frame
from ev_vision.tuning.diagnostics import compute_diagnostics
from ev_vision.tuning.models import (
    CameraIdentity,
    CaptureSnapshot,
    DetectionCandidateSnapshot,
    DetectionDebugSnapshot,
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
    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
        update_tracker: bool = True,
    ) -> HybridBoardResult | ClassicalBoardResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def apply_config(self, config: DetectionConfig) -> None:
        raise NotImplementedError

    def reload_model(self) -> None:
        raise NotImplementedError


class StaleDetectionFrameError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"requested detection sequence {expected}, latest is {actual}")
        self.expected = expected
        self.actual = actual


def copy_frame(frame: Frame | None) -> Frame | None:
    if frame is None:
        return None
    return Frame(frame.sequence, frame.captured_ns, frame.image.copy())


def copy_debug_images(images: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return {name: image.copy() for name, image in images.items()}


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


class _MutationCancelled(RuntimeError):
    """An in-flight start/apply was invalidated by a shutdown request."""


class _AnalysisWakeup:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.generation = 0


class CameraTuningService:
    """Own a latest-only camera session with transactional parameter changes."""

    def __init__(
        self,
        *,
        camera_factory: CameraFactory,
        base_config: CameraConfig,
        detector: DetectorPort | None = None,
        detection_config: DetectionConfig | None = None,
        bounds: ParameterBounds | None = None,
        camera_identity: CameraIdentity | None = None,
        read_timeout_ms: int = 100,
        confirm_timeout_s: float = 1.0,
        diagnostics_fps: float = 10.0,
        detection_fps: float = 15.0,
        shutdown_timeout_s: float = 0.25,
        disconnect_timeout_threshold: int = 3,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if read_timeout_ms <= 0:
            raise ValueError("read_timeout_ms must be positive")
        if confirm_timeout_s <= 0:
            raise ValueError("confirm_timeout_s must be positive")
        if diagnostics_fps <= 0 or detection_fps <= 0:
            raise ValueError("analysis rates must be positive")
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if disconnect_timeout_threshold <= 0:
            raise ValueError("disconnect_timeout_threshold must be positive")

        self._camera_factory = camera_factory
        self._base_config = base_config
        self._detector = detector
        self._detection_config = detection_config or DetectionConfig()
        self._bounds = bounds or ParameterBounds()
        self._camera_identity = camera_identity or CameraIdentity(model="", serial="")
        self._read_timeout_ms = int(read_timeout_ms)
        self._confirm_timeout_s = float(confirm_timeout_s)
        self._diagnostics_period_s = 1.0 / float(diagnostics_fps)
        self._detection_period_s = 1.0 / float(detection_fps)
        self._shutdown_timeout_s = float(shutdown_timeout_s)
        self._disconnect_timeout_threshold = int(disconnect_timeout_threshold)
        self._clock_ns = clock_ns

        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._detector_lock = threading.RLock()
        self._camera: CameraPort | None = None
        # A claimed camera handle is closed exactly once by its dedicated worker.
        # A failed operation deliberately retains its owner/result as a terminal
        # marker so repeated stop calls cannot retry an uncertain native handle.
        self._camera_close_owner: CameraPort | None = None
        self._camera_close_done: threading.Event | None = None
        self._camera_close_thread: threading.Thread | None = None
        self._camera_close_error: BaseException | None = None
        self._camera_stop: threading.Event | None = None
        self._acquisition_thread: threading.Thread | None = None
        self._deferred_cleanup = False
        self._cleanup_claimed = False
        self._acquisition_target_done: threading.Event | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._session_generation = 0
        self._session_active = False
        self._running = False
        self._shutdown_requested = False
        self._shutdown_generation = 0
        self._active_mutations = 0

        # Analysis workers are one bounded pair per running lifecycle. A blocked
        # worker prevents replacement, so detector calls cannot overlap on restart.
        self._diagnostics_wakeup = _AnalysisWakeup()
        self._detection_wakeup = _AnalysisWakeup()
        self._analysis_stop = threading.Event()
        self._diagnostics_thread: threading.Thread | None = None
        self._detection_thread: threading.Thread | None = None

        self._state = "Stopped"
        self._last_error: str | None = None
        self._applied = EditableCameraParameters.from_camera_config(base_config)
        self._latest_frame: Frame | None = None
        self._latest_frame_received_ns: int | None = None
        self._latest_diagnostics: ImageDiagnostics | None = None
        self._detection_enabled = detector is not None
        self._detection_generation = 0
        self._detection_reload_in_progress = False
        self._latest_detection = DetectionSnapshot(
            enabled=self._detection_enabled, detected=False
        )
        self._detection_computed_ns: int | None = None
        self._latest_detection_frame: Frame | None = None
        self._latest_detection_debug: DetectionDebugSnapshot | None = None

        self._frame_count = 0
        self._timeout_count = 0
        self._consecutive_timeouts = 0
        self._camera_fault_active = False
        self._sequence_gap_count = 0
        self._last_sequence: int | None = None
        self._acquisition_rate = _RateWindow()
        self._preview_rate = _RateWindow()
        self._diagnostics_rate = _RateWindow()
        self._detection_rate = _RateWindow()

    def start(self) -> None:
        with self._lock:
            invocation_generation = self._shutdown_generation
            invocation_saw_shutdown = self._shutdown_requested
            if invocation_saw_shutdown and (
                self._active_mutations > 0
                or not self._shutdown_cleanup_complete_locked()
            ):
                raise RuntimeError("camera shutdown or cleanup is still in progress")
        with self._operation_lock:
            mutation_generation: int | None = None
            mutation_registered = False
            try:
                with self._lock:
                    if invocation_generation != self._shutdown_generation:
                        raise RuntimeError("camera start was cancelled by shutdown")
                    if self._shutdown_requested:
                        if (
                            not invocation_saw_shutdown
                            or not self._shutdown_cleanup_complete_locked()
                        ):
                            raise RuntimeError(
                                "camera shutdown or cleanup is still in progress"
                            )
                        self._shutdown_requested = False
                        self._shutdown_generation += 1
                    if self._running:
                        return
                    if self._camera is not None or self._acquisition_thread is not None:
                        raise RuntimeError("camera handle is still pending cleanup")
                    self._active_mutations += 1
                    mutation_registered = True
                    mutation_generation = self._shutdown_generation
                    self._state = "Starting"
                    self._last_error = None
                self._ensure_analysis_workers()

                try:
                    camera, confirming_frame = self._open_and_confirm(
                        self._applied.to_camera_config(self._base_config),
                        mutation_generation=mutation_generation,
                    )
                    self._install_session(
                        camera,
                        confirming_frame,
                        parameters=self._applied,
                        error=None,
                        mutation_generation=mutation_generation,
                    )
                    self._raise_if_mutation_cancelled(mutation_generation)
                except BaseException as exc:
                    if not isinstance(exc, _MutationCancelled):
                        with self._lock:
                            cancelled = self._mutation_cancelled_locked(
                                mutation_generation
                            )
                        if not cancelled:
                            self._set_state("Disconnected", str(exc))
                    raise
            finally:
                if mutation_registered:
                    with self._lock:
                        self._active_mutations -= 1
                self._finish_shutdown_after_mutation()

    def stop(self) -> None:
        deadline = time.monotonic() + self._shutdown_timeout_s
        self._publish_shutdown_intent()
        if not self._operation_lock.acquire(timeout=self._remaining(deadline)):
            return
        try:
            with self._lock:
                already_stopped = (
                    self._state == "Stopped"
                    and self._camera is None
                    and self._acquisition_thread is None
                )
            if already_stopped:
                self._stop_analysis_workers(timeout_s=self._remaining(deadline))
                return

            acquisition_stopped = self._stop_acquisition(
                "camera acquisition did not stop before shutdown timeout",
                timeout_s=self._remaining(deadline),
            )
            self._stop_analysis_workers(timeout_s=self._remaining(deadline))
            if not acquisition_stopped:
                return

            close_completed, close_error = self._close_camera(
                deadline=deadline,
                pending_error="camera close cleanup is still in progress",
            )
            if not close_completed:
                with self._lock:
                    done = self._camera_close_done
                    if self._camera is not None and (done is None or not done.is_set()):
                        self._state = "Disconnected"
                        self._last_error = "camera close cleanup is still in progress"
                return
            if close_error is not None:
                return
            with self._lock:
                if self._camera is None:
                    self._state = "Stopped"
                    self._last_error = None
        finally:
            self._operation_lock.release()

    def apply_parameters(
        self, candidate: EditableCameraParameters
    ) -> EditableCameraParameters:
        candidate = self._bounds.validate(candidate)
        with self._lock:
            mutation_generation = self._shutdown_generation
        with self._operation_lock:
            mutation_registered = False
            try:
                with self._lock:
                    if self._mutation_cancelled_locked(mutation_generation):
                        raise RuntimeError("camera mutation was cancelled by shutdown")
                    self._active_mutations += 1
                    mutation_registered = True
                    if self._mutation_cancelled_locked(mutation_generation):
                        raise RuntimeError("camera mutation was cancelled by shutdown")
                    if not self._running or not self._session_active:
                        raise RuntimeError("camera tuning service is not running")
                    if self._deferred_cleanup:
                        raise RuntimeError("camera session is pending cleanup")
                    last_good = self._applied
                    self._state = "Applying"
                    self._last_error = None

                shutdown_error = "camera acquisition did not stop before apply timeout"
                if not self._stop_acquisition(shutdown_error):
                    error = RuntimeError(shutdown_error)
                    raise ParameterApplyError(
                        f"apply failed: {error}", apply_error=error
                    ) from error

                with self._lock:
                    active_camera = self._camera
                reconfigure = (
                    getattr(active_camera, "reconfigure", None)
                    if active_camera is not None
                    else None
                )
                if callable(reconfigure):
                    return self._apply_parameters_in_place(
                        active_camera,
                        candidate,
                        last_good=last_good,
                        mutation_generation=mutation_generation,
                    )

                close_deadline = time.monotonic() + self._shutdown_timeout_s
                close_completed, close_error = self._close_camera(
                    deadline=close_deadline,
                    pending_error="camera close cleanup is still in progress",
                )
                if not close_completed:
                    error = RuntimeError("camera close did not finish before apply timeout")
                    message = f"apply failed: {error}"
                    with self._lock:
                        self._running = False
                        self._state = "Disconnected"
                        self._last_error = message
                    raise ParameterApplyError(message, apply_error=error) from error
                if close_error is not None:
                    message = f"apply failed: camera close failed: {close_error}"
                    with self._lock:
                        self._running = False
                        self._state = "Disconnected"
                        self._last_error = message
                    raise ParameterApplyError(
                        message, apply_error=close_error
                    ) from close_error

                self._raise_if_mutation_cancelled(mutation_generation)
                apply_error: BaseException | None = None
                try:
                    camera, confirming_frame = self._open_and_confirm(
                        candidate.to_camera_config(self._base_config),
                        mutation_generation=mutation_generation,
                    )
                except BaseException as exc:
                    apply_error = exc

                if apply_error is not None:
                    with self._lock:
                        cancelled = self._mutation_cancelled_locked(mutation_generation)
                        retained_handle = self._camera is not None
                    if cancelled:
                        error = RuntimeError("camera apply was cancelled by shutdown")
                        raise ParameterApplyError(
                            f"apply failed: {error}", apply_error=error
                        ) from apply_error
                    if retained_handle:
                        message = f"apply failed: {apply_error}; camera cleanup failed"
                        with self._lock:
                            self._running = False
                            self._state = "Disconnected"
                            self._last_error = message
                        raise ParameterApplyError(
                            message, apply_error=apply_error
                        ) from apply_error

                    self._set_state("Recovering", str(apply_error))
                    self._raise_if_mutation_cancelled(mutation_generation)
                    try:
                        rollback_camera, rollback_frame = self._open_and_confirm(
                            last_good.to_camera_config(self._base_config),
                            mutation_generation=mutation_generation,
                        )
                    except BaseException as rollback_error:
                        with self._lock:
                            cancelled = self._mutation_cancelled_locked(
                                mutation_generation
                            )
                        if cancelled:
                            error = RuntimeError(
                                "camera rollback was cancelled by shutdown"
                            )
                            raise ParameterApplyError(
                                f"apply failed: {error}",
                                apply_error=apply_error,
                            ) from rollback_error
                        message = (
                            f"apply failed: {apply_error}; "
                            f"rollback failed: {rollback_error}"
                        )
                        with self._lock:
                            self._running = False
                            self._session_active = False
                            self._state = "Disconnected"
                            self._last_error = message
                        raise ParameterApplyError(
                            message,
                            apply_error=apply_error,
                            rollback_error=rollback_error,
                        ) from apply_error

                    try:
                        self._install_session(
                            rollback_camera,
                            rollback_frame,
                            parameters=last_good,
                            error=f"apply failed and rolled back: {apply_error}",
                            mutation_generation=mutation_generation,
                        )
                    except _MutationCancelled as cancelled:
                        self._close_uninstalled_camera(rollback_camera, cancelled)
                        error = RuntimeError("camera rollback was cancelled by shutdown")
                        raise ParameterApplyError(
                            f"apply failed: {error}", apply_error=apply_error
                        ) from cancelled
                    raise ParameterApplyError(
                        f"apply failed and rolled back: {apply_error}",
                        apply_error=apply_error,
                    ) from apply_error

                installed = False
                try:
                    self._install_session(
                        camera,
                        confirming_frame,
                        parameters=candidate,
                        error=None,
                        mutation_generation=mutation_generation,
                    )
                    installed = True
                    self._raise_if_mutation_cancelled(mutation_generation)
                except _MutationCancelled as cancelled:
                    if not installed:
                        self._close_uninstalled_camera(camera, cancelled)
                    error = RuntimeError("camera apply was cancelled by shutdown")
                    raise ParameterApplyError(
                        f"apply failed: {error}", apply_error=error
                    ) from cancelled
                return candidate
            finally:
                if mutation_registered:
                    with self._lock:
                        self._active_mutations -= 1
                self._finish_shutdown_after_mutation()

    @property
    def active_camera(self) -> CameraPort | None:
        with self._lock:
            return self._camera

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

    def latest_frame_with_detection(
        self,
    ) -> tuple[Frame, DetectionSnapshot] | None:
        now_ns = self._clock_ns()
        with self._lock:
            frame = copy_frame(self._latest_frame)
            if frame is None:
                return None
            detection = self._detection_snapshot_locked(now_ns)
            if detection.source_sequence != frame.sequence:
                detection = self._unavailable_detection_for_frame_locked(frame.sequence)
            return frame, detection

    def latest_diagnostics(self) -> ImageDiagnostics | None:
        with self._lock:
            return self._latest_diagnostics

    def latest_detection(self) -> DetectionSnapshot:
        now_ns = self._clock_ns()
        with self._lock:
            return self._detection_snapshot_locked(now_ns)

    def detection_config(self) -> DetectionConfig:
        with self._lock:
            return self._detection_config

    def apply_detection_config(self, config: DetectionConfig) -> DetectionConfig:
        detector = self._detector
        if detector is None:
            raise RuntimeError("no detector is configured")
        apply_config = getattr(detector, "apply_config", None)
        if not callable(apply_config):
            raise RuntimeError("configured detector does not support detection config updates")
        with self._detector_lock:
            apply_config(config)
        with self._lock:
            self._detection_config = config
            self._detection_generation += 1
            self._latest_detection_frame = None
            self._latest_detection_debug = None
        self._notify_analysis_worker(self._detection_wakeup)
        return config

    def _unavailable_detection_for_frame_locked(
        self, source_sequence: int
    ) -> DetectionSnapshot:
        detector = self._detector
        return DetectionSnapshot(
            enabled=self._detection_enabled,
            detected=False,
            source_sequence=source_sequence,
            error="matching detection unavailable",
            target_valid=False,
            tracking_state="SEARCHING",
            observation_source="NONE",
            model_state=str(getattr(detector, "model_state", "UNAVAILABLE")),
            model_backend=str(getattr(detector, "model_backend", "none")),
            model_path=getattr(detector, "model_path", None),
            failure_reason="DETECTION_UNAVAILABLE",
        )

    def detection_frame_for_latest(
        self,
        *,
        expected_sequence: int | None = None,
    ) -> tuple[Frame, DetectionSnapshot] | None:
        with self._lock:
            frame = copy_frame(self._latest_detection_frame)
            snapshot = self._detection_snapshot_locked(self._clock_ns())
        if frame is None or snapshot.source_sequence != frame.sequence:
            return None
        if expected_sequence is not None and frame.sequence != expected_sequence:
            raise StaleDetectionFrameError(expected_sequence, frame.sequence)
        return frame, snapshot

    def detection_debug_for_latest(
        self,
        *,
        expected_sequence: int | None = None,
    ) -> DetectionDebugSnapshot | None:
        """Inspect the retained detection input frame without advancing the tracker."""
        pair = self.detection_frame_for_latest(expected_sequence=expected_sequence)
        if pair is None:
            return None
        frame, _snapshot = pair
        with self._lock:
            cached = self._latest_detection_debug
        if cached is not None and cached.source_sequence == frame.sequence:
            return cached
        snapshot = self._inspect_detection_debug(frame)
        if snapshot is None:
            return None
        with self._lock:
            if (
                self._latest_detection_frame is not None
                and self._latest_detection_frame.sequence == frame.sequence
            ):
                self._latest_detection_debug = snapshot
        return snapshot

    def _inspect_detection_debug(
        self, frame: Frame
    ) -> DetectionDebugSnapshot | None:
        detector = self._detector
        if detector is None:
            return None
        result = self._detect_board(
            detector,
            frame,
            include_debug=True,
            update_tracker=False,
        )
        return self._debug_snapshot_for_result(result, frame)

    def reload_detection_model(self) -> None:
        detector = self._detector
        if detector is None:
            raise RuntimeError("no detector is configured")
        if str(getattr(detector, "model_backend", "none")) == "classical":
            with self._lock:
                self._invalidate_detection_for_reload_locked(detector)
            self._notify_analysis_worker(self._detection_wakeup)
            return
        reload_model = getattr(detector, "reload_model", None)
        if not callable(reload_model):
            raise RuntimeError("configured detector does not support model reload")
        with self._lock:
            self._detection_reload_in_progress = True
            self._invalidate_detection_for_reload_locked(detector)
        self._notify_analysis_worker(self._detection_wakeup)
        try:
            with self._detector_lock:
                reload_model()
        finally:
            with self._lock:
                self._invalidate_detection_for_reload_locked(detector)
                self._detection_reload_in_progress = False
            self._notify_analysis_worker(self._detection_wakeup)

    def _invalidate_detection_for_reload_locked(self, detector: DetectorPort) -> None:
        model_state = str(getattr(detector, "model_state", "UNAVAILABLE"))
        model_backend = str(getattr(detector, "model_backend", "none"))
        model_path = getattr(detector, "model_path", None)
        failure_reason = (
            "MODEL_UNAVAILABLE" if model_state.upper() == "UNAVAILABLE" else None
        )
        self._detection_generation += 1
        self._detection_computed_ns = None
        self._latest_detection_frame = None
        self._latest_detection_debug = None
        self._latest_detection = DetectionSnapshot(
            enabled=self._detection_enabled,
            detected=False,
            target_valid=False,
            tracking_state="SEARCHING",
            model_state=model_state,
            model_backend=model_backend,
            model_path=None if model_path is None else str(model_path),
            failure_reason=failure_reason,
        )

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
            self._latest_detection_frame = None
            self._latest_detection_debug = None
            self._latest_detection = DetectionSnapshot(enabled=enabled, detected=False)
        if not enabled:
            self._reset_detector()
        self._notify_analysis_worker(self._detection_wakeup)

    def capture_snapshot(
        self, overlay_options: OverlayOptions | None = None
    ) -> CaptureSnapshot:
        now_ns = self._clock_ns()
        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("no camera frame is available")
            parameters = self._applied
            frame = self._copy_frame(self._latest_frame)
            runtime = self._runtime_snapshot_locked(now_ns)
            diagnostics = self._latest_diagnostics
            detection = self._detection_snapshot_locked(now_ns)
            detection_matches = detection.source_sequence == frame.sequence
            if not detection_matches:
                detection = self._unavailable_detection_for_frame_locked(frame.sequence)
            cached_debug = self._latest_detection_debug
            camera_config = parameters.to_camera_config(self._base_config)
            camera_identity = self._camera_identity
        detection_debug = None
        if detection_matches:
            if cached_debug is not None and cached_debug.source_sequence == frame.sequence:
                detection_debug = DetectionDebugSnapshot(
                    cached_debug.source_sequence, copy_debug_images(cached_debug.images)
                )
            else:
                try:
                    detection_debug = self._inspect_detection_debug(frame)
                except RuntimeError:
                    detection_debug = None
        return CaptureSnapshot(
            frame=frame,
            parameters=parameters,
            runtime=runtime,
            diagnostics=diagnostics,
            detection=detection,
            overlay_options=overlay_options or OverlayOptions(),
            camera_config=camera_config,
            camera_identity=camera_identity,
            detection_debug=detection_debug,
        )

    def record_preview_frame(self) -> None:
        now_ns = self._clock_ns()
        with self._lock:
            self._preview_rate.record(now_ns)

    def _ensure_analysis_workers(self) -> None:
        with self._lock:
            existing = (self._diagnostics_thread, self._detection_thread)
            alive = [
                thread
                for thread in existing
                if thread is not None and thread.is_alive()
            ]
            if alive:
                if self._analysis_stop.is_set():
                    raise RuntimeError("analysis workers are still shutting down")
                return
            if any(thread is not None for thread in existing):
                self._diagnostics_thread = None
                self._detection_thread = None
            self._analysis_stop = threading.Event()
            diagnostics = threading.Thread(
                target=self._diagnostics_loop,
                args=(self._analysis_stop,),
                name="camera-tuning-diagnostics",
                daemon=True,
            )
            detection = threading.Thread(
                target=self._detection_loop,
                args=(self._analysis_stop,),
                name="camera-tuning-detection",
                daemon=True,
            )
            self._diagnostics_thread = diagnostics
            self._detection_thread = detection
        detection.start()
        diagnostics.start()

    def _stop_analysis_workers(self, *, timeout_s: float | None = None) -> bool:
        with self._lock:
            stop_event = self._analysis_stop
            threads = (self._diagnostics_thread, self._detection_thread)
        stop_event.set()
        self._notify_analysis_worker(self._diagnostics_wakeup)
        self._notify_analysis_worker(self._detection_wakeup)
        current = threading.current_thread()
        deadline = time.monotonic() + (
            self._shutdown_timeout_s if timeout_s is None else max(0.0, timeout_s)
        )
        for thread in threads:
            if thread is not None and thread is not current:
                thread.join(timeout=self._remaining(deadline))
        with self._lock:
            alive = any(thread is not None and thread.is_alive() for thread in threads)
            if not alive:
                if self._diagnostics_thread is threads[0]:
                    self._diagnostics_thread = None
                if self._detection_thread is threads[1]:
                    self._detection_thread = None
            return not alive

    def _apply_parameters_in_place(
        self,
        camera: CameraPort,
        candidate: EditableCameraParameters,
        *,
        last_good: EditableCameraParameters,
        mutation_generation: int,
    ) -> EditableCameraParameters:
        reconfigure = getattr(camera, "reconfigure")
        apply_error: BaseException | None = None
        try:
            reconfigure(candidate.to_camera_config(self._base_config))
            confirming_frame = self._confirm_camera_frame(
                camera,
                mutation_generation=mutation_generation,
            )
        except BaseException as exc:
            apply_error = exc

        if apply_error is None:
            try:
                self._install_session(
                    camera,
                    confirming_frame,
                    parameters=candidate,
                    error=None,
                    mutation_generation=mutation_generation,
                )
            except _MutationCancelled as cancelled:
                error = RuntimeError("camera apply was cancelled by shutdown")
                raise ParameterApplyError(
                    f"apply failed: {error}",
                    apply_error=error,
                ) from cancelled
            return candidate

        with self._lock:
            cancelled = self._mutation_cancelled_locked(mutation_generation)
        if cancelled:
            error = RuntimeError("camera apply was cancelled by shutdown")
            raise ParameterApplyError(
                f"apply failed: {error}",
                apply_error=error,
            ) from apply_error

        self._set_state("Recovering", str(apply_error))
        try:
            reconfigure(last_good.to_camera_config(self._base_config))
            rollback_frame = self._confirm_camera_frame(
                camera,
                mutation_generation=mutation_generation,
            )
            self._install_session(
                camera,
                rollback_frame,
                parameters=last_good,
                error=f"apply failed and rolled back: {apply_error}",
                mutation_generation=mutation_generation,
            )
        except BaseException as rollback_error:
            with self._lock:
                self._running = False
                self._session_active = False
                self._state = "Disconnected"
                self._last_error = (
                    f"apply failed: {apply_error}; rollback failed: {rollback_error}"
                )
            raise ParameterApplyError(
                self._last_error,
                apply_error=apply_error,
                rollback_error=rollback_error,
            ) from apply_error

        raise ParameterApplyError(
            f"apply failed and rolled back: {apply_error}",
            apply_error=apply_error,
        ) from apply_error

    def _confirm_camera_frame(
        self,
        camera: CameraPort,
        *,
        mutation_generation: int,
    ) -> Frame:
        deadline_ns = self._clock_ns() + int(
            self._confirm_timeout_s * 1_000_000_000
        )
        while True:
            try:
                candidate = camera.read(timeout_ms=self._read_timeout_ms)
                self._raise_if_mutation_cancelled(mutation_generation)
                return self._freeze_frame(candidate)
            except TimeoutError:
                with self._lock:
                    self._timeout_count += 1
                self._raise_if_mutation_cancelled(mutation_generation)
                if self._clock_ns() >= deadline_ns:
                    raise TimeoutError(
                        "timed out waiting for a confirming camera frame"
                    )

    def _open_and_confirm(
        self, config: CameraConfig, *, mutation_generation: int
    ) -> tuple[CameraPort, Frame]:
        camera: CameraPort | None = None
        try:
            camera = self._camera_factory(config)
            self._raise_if_mutation_cancelled(mutation_generation)
            camera.open()
            self._raise_if_mutation_cancelled(mutation_generation)
            confirming_frame = self._confirm_camera_frame(
                camera,
                mutation_generation=mutation_generation,
            )
            return camera, confirming_frame
        except BaseException as primary_error:
            if camera is not None:
                self._close_uninstalled_camera(camera, primary_error)
            raise

    def _close_uninstalled_camera(
        self, camera: CameraPort, primary_error: BaseException
    ) -> None:
        with self._lock:
            if self._camera is not None and self._camera is not camera:
                raise RuntimeError(
                    f"{primary_error}; another camera handle is pending cleanup"
                ) from primary_error
            self._camera = camera
        deadline = time.monotonic() + self._shutdown_timeout_s
        close_completed, close_error = self._close_camera(
            deadline=deadline,
            pending_error=f"{primary_error}; camera close cleanup is still in progress",
        )
        if not close_completed:
            return
        if close_error is not None:
            raise RuntimeError(
                f"{primary_error}; camera close failed: {close_error}"
            ) from primary_error

    def _install_session(
        self,
        camera: CameraPort,
        confirming_frame: Frame,
        *,
        parameters: EditableCameraParameters,
        error: str | None,
        mutation_generation: int,
    ) -> None:
        self._reset_detector()
        stop_event = threading.Event()
        target_done = threading.Event()
        with self._lock:
            if self._mutation_cancelled_locked(mutation_generation):
                raise _MutationCancelled("camera session install cancelled by shutdown")
            self._session_generation += 1
            generation = self._session_generation
            acquisition = threading.Thread(
                target=self._acquisition_loop,
                args=(camera, stop_event, generation, target_done),
                name="camera-tuning-acquisition",
                daemon=True,
            )
            self._camera = camera
            self._camera_stop = stop_event
            self._acquisition_thread = acquisition
            self._acquisition_target_done = target_done
            self._cleanup_thread = None
            self._deferred_cleanup = False
            self._cleanup_claimed = False
            self._running = True
            self._session_active = True
            self._consecutive_timeouts = 0
            self._camera_fault_active = False
            self._reset_derived_locked()
            self._applied = parameters
            self._publish_frozen_frame_locked(confirming_frame, reset_sequence=True)
            self._state = "Connected"
            self._last_error = error
            acquisition.start()
        self._wake_analysis_workers()

    def _publish_shutdown_intent(self) -> None:
        self._reset_detector()
        with self._lock:
            self._shutdown_generation += 1
            self._shutdown_requested = True
            self._running = False
            self._session_active = False
            self._session_generation += 1
            self._reset_derived_locked()
            camera_stop = self._camera_stop
            analysis_stop = self._analysis_stop
            close_failed = (
                self._camera_close_owner is not None
                and self._camera_close_done is not None
                and self._camera_close_done.is_set()
                and self._camera_close_error is not None
            )
            if not close_failed and not self._shutdown_cleanup_complete_locked():
                self._state = "Disconnected"
                self._last_error = "camera shutdown is in progress"
        if camera_stop is not None:
            camera_stop.set()
        analysis_stop.set()
        self._wake_analysis_workers()

    def _mutation_cancelled_locked(self, generation: int) -> bool:
        return self._shutdown_requested or generation != self._shutdown_generation

    def _raise_if_mutation_cancelled(self, generation: int) -> None:
        with self._lock:
            if self._mutation_cancelled_locked(generation):
                raise _MutationCancelled("camera mutation cancelled by shutdown")

    def _shutdown_cleanup_complete_locked(self) -> bool:
        if self._active_mutations > 0:
            return False
        analysis_alive = any(
            thread is not None and thread.is_alive()
            for thread in (self._diagnostics_thread, self._detection_thread)
        )
        cleanup_alive = (
            self._cleanup_thread is not None and self._cleanup_thread.is_alive()
        )
        return (
            self._camera is None
            and self._camera_close_owner is None
            and self._acquisition_thread is None
            and not self._deferred_cleanup
            and not cleanup_alive
            and not analysis_alive
        )

    def _finish_shutdown_after_mutation(self) -> None:
        with self._lock:
            requested = self._shutdown_requested
        if not requested:
            return

        deadline = time.monotonic() + self._shutdown_timeout_s
        acquisition_stopped = self._stop_acquisition(
            "camera acquisition did not stop before shutdown timeout",
            timeout_s=self._remaining(deadline),
        )
        self._stop_analysis_workers(timeout_s=self._remaining(deadline))
        if not acquisition_stopped:
            return

        close_completed, close_error = self._close_camera(
            deadline=deadline,
            pending_error="camera close cleanup is still in progress",
        )
        if not close_completed or close_error is not None:
            return
        with self._lock:
            if self._shutdown_cleanup_complete_locked():
                self._state = "Stopped"
                self._last_error = None

    def _stop_acquisition(
        self, timeout_error: str, *, timeout_s: float | None = None
    ) -> bool:
        with self._lock:
            stop_event = self._camera_stop
            acquisition = self._acquisition_thread
            self._session_active = False
            self._session_generation += 1
            self._reset_derived_locked()
        self._wake_analysis_workers()
        if stop_event is not None:
            stop_event.set()
        current = threading.current_thread()
        if acquisition is not None and acquisition is not current:
            acquisition.join(
                timeout=self._shutdown_timeout_s
                if timeout_s is None
                else max(0.0, timeout_s)
            )

        start_cleanup_waiter = False
        timed_out = acquisition is not None and acquisition.is_alive()
        with self._lock:
            if timed_out:
                self._running = False
                self._deferred_cleanup = True
                self._state = "Disconnected"
                self._last_error = timeout_error
                target_done = self._acquisition_target_done
                if target_done is not None and target_done.is_set():
                    if not self._cleanup_claimed:
                        self._cleanup_claimed = True
                        start_cleanup_waiter = True
                elif not self._cleanup_claimed:
                    # The acquisition target still owns cleanup through its finally.
                    pass
                return_after_handoff = True
            else:
                return_after_handoff = False
                if self._acquisition_thread is acquisition:
                    self._acquisition_thread = None
                    self._camera_stop = None
                    self._acquisition_target_done = None
        if start_cleanup_waiter:
            self._start_deferred_cleanup_waiter(acquisition)
        if return_after_handoff:
            return False
        return True

    def _close_camera(
        self, *, deadline: float, pending_error: str
    ) -> tuple[bool, BaseException | None]:
        done = self._claim_camera_close(pending_error)
        if done is None:
            return True, None
        done.wait(timeout=self._remaining(deadline))
        with self._lock:
            if not done.is_set():
                return False, None
            return True, self._camera_close_error

    def _claim_camera_close(self, pending_error: str) -> threading.Event | None:
        start_thread: threading.Thread | None = None
        with self._lock:
            camera = self._camera
            if camera is None:
                return None
            if self._camera_close_owner is None:
                done = threading.Event()
                start_thread = threading.Thread(
                    target=self._camera_close_worker,
                    args=(camera, done),
                    name="camera-tuning-close",
                    daemon=True,
                )
                self._camera_close_owner = camera
                self._camera_close_done = done
                self._camera_close_thread = start_thread
                self._camera_close_error = None
                self._state = "Disconnected"
                self._last_error = pending_error
            elif self._camera_close_owner is camera:
                done = self._camera_close_done
                if done is None:
                    raise RuntimeError("camera close ownership is incomplete")
            else:
                raise RuntimeError("another camera handle owns close cleanup")
        if start_thread is not None:
            start_thread.start()
        return done

    def _camera_close_worker(
        self, camera: CameraPort, done: threading.Event
    ) -> None:
        close_error: BaseException | None = None
        try:
            camera.close()
        except BaseException as exc:
            close_error = exc

        with self._lock:
            if (
                self._camera_close_owner is camera
                and self._camera_close_done is done
            ):
                self._camera_close_error = close_error
                self._camera_close_thread = None
                self._deferred_cleanup = False
                self._cleanup_claimed = False
                if close_error is None:
                    if self._camera is camera:
                        self._camera = None
                    self._camera_close_owner = None
                    self._camera_close_done = None
                    self._camera_close_error = None
                    if not self._running:
                        self._state = "Stopped"
                        self._last_error = None
                else:
                    # Retain both the native handle and terminal owner forever:
                    # retrying an uncertain close would violate exactly-once.
                    self._state = "Disconnected"
                    self._last_error = f"camera close failed: {close_error}"
        done.set()

    def _start_deferred_cleanup_waiter(
        self, acquisition: threading.Thread | None
    ) -> None:
        if acquisition is None:
            self._close_deferred_camera(thread=None)
            return
        cleanup = threading.Thread(
            target=self._deferred_cleanup_waiter,
            args=(acquisition,),
            name="camera-tuning-cleanup",
            daemon=True,
        )
        with self._lock:
            self._cleanup_thread = cleanup
        cleanup.start()

    def _deferred_cleanup_waiter(self, acquisition: threading.Thread) -> None:
        acquisition.join()
        self._close_deferred_camera(thread=acquisition)

    def _close_deferred_camera(self, *, thread: threading.Thread | None) -> None:
        with self._lock:
            if self._acquisition_thread is thread:
                self._acquisition_thread = None
                self._camera_stop = None
                self._acquisition_target_done = None
            if self._cleanup_thread is threading.current_thread():
                self._cleanup_thread = None
        self._claim_camera_close("camera close cleanup is still in progress")

    def _acquisition_loop(
        self,
        camera: CameraPort,
        stop_event: threading.Event,
        generation: int,
        target_done: threading.Event,
    ) -> None:
        try:
            while not stop_event.is_set():
                try:
                    camera_frame = camera.read(timeout_ms=self._read_timeout_ms)
                except TimeoutError:
                    self._record_timeout(generation)
                    continue
                except BaseException as exc:
                    if stop_event.is_set():
                        break
                    self._record_camera_error(exc, generation)
                    stop_event.wait(0.01)
                    continue

                frozen = self._freeze_frame(camera_frame)
                with self._lock:
                    if stop_event.is_set() or generation != self._session_generation:
                        break
                    self._record_camera_success_locked()
                    self._publish_frozen_frame_locked(frozen)
                self._wake_analysis_workers()
        finally:
            claim_close = False
            current = threading.current_thread()
            with self._lock:
                if (
                    self._deferred_cleanup
                    and not self._cleanup_claimed
                    and self._acquisition_thread is current
                    and self._camera is camera
                ):
                    self._cleanup_claimed = True
                    claim_close = True
                elif self._acquisition_thread is current and not self._deferred_cleanup:
                    # Keep the reference for the stop/apply caller to close synchronously.
                    pass
                target_done.set()
            if claim_close:
                self._start_deferred_cleanup_waiter(current)

    def _diagnostics_loop(self, stop_event: threading.Event) -> None:
        last_key: tuple[int, int] | None = None
        while not stop_event.is_set():
            started_ns = self._clock_ns()
            wake_generation = self._analysis_wake_generation(
                self._diagnostics_wakeup
            )
            with self._lock:
                if not self._session_active or self._latest_frame is None:
                    item = None
                else:
                    key = (self._session_generation, self._latest_frame.sequence)
                    item = (
                        None
                        if key == last_key
                        else (key, self._copy_frame(self._latest_frame))
                    )
            if item is not None:
                key, frame = item
                try:
                    result = compute_diagnostics(
                        frame.image,
                        source_sequence=frame.sequence,
                        computed_ns=self._clock_ns(),
                    )
                except BaseException as exc:
                    self._record_analysis_error("diagnostics", exc, key)
                else:
                    now_ns = self._clock_ns()
                    with self._lock:
                        if self._diagnostics_key_is_current_locked(key):
                            self._latest_diagnostics = result
                            self._diagnostics_rate.record(now_ns)
                            if (
                                self._state == "Connected"
                                and self._last_error is not None
                                and self._last_error.startswith("diagnostics failed:")
                            ):
                                self._last_error = None
                            last_key = key
            elapsed_s = (self._clock_ns() - started_ns) / 1_000_000_000.0
            if stop_event.is_set():
                break
            self._wait_for_analysis(
                stop_event,
                self._diagnostics_wakeup,
                wake_generation,
                max(0.0, self._diagnostics_period_s - elapsed_s),
                wake_on_change=item is None,
            )

    def _detection_loop(self, stop_event: threading.Event) -> None:
        last_key: tuple[int, int, int] | None = None
        while not stop_event.is_set():
            started_ns = self._clock_ns()
            wake_generation = self._analysis_wake_generation(
                self._detection_wakeup
            )
            with self._lock:
                detector = self._detector
                if (
                    detector is None
                    or not self._detection_enabled
                    or self._detection_reload_in_progress
                    or not self._session_active
                    or self._latest_frame is None
                ):
                    item = None
                else:
                    key = (
                        self._session_generation,
                        self._detection_generation,
                        self._latest_frame.sequence,
                    )
                    item = (
                        None
                        if key == last_key
                        else (key, self._copy_frame(self._latest_frame))
                    )
            if item is not None:
                key, frame = item
                try:
                    result = self._detect_board(
                        detector,
                        frame,
                        include_debug=False,
                        update_tracker=True,
                    )
                except BaseException as exc:
                    now_ns = self._clock_ns()
                    with self._lock:
                        if self._detection_key_is_current_locked(key):
                            self._latest_detection = DetectionSnapshot(
                                enabled=True,
                                detected=False,
                                source_sequence=frame.sequence,
                                error=str(exc),
                                target_valid=False,
                                model_state="ERROR",
                                failure_reason="MODEL_ERROR",
                            )
                            self._latest_detection_frame = copy_frame(frame)
                            self._latest_detection_debug = None
                            self._detection_computed_ns = now_ns
                            self._detection_rate.record(now_ns)
                            last_key = key
                else:
                    now_ns = self._clock_ns()
                    with self._lock:
                        if self._detection_key_is_current_locked(key):
                            self._latest_detection = self._snapshot_for_result(
                                result, frame
                            )
                            self._latest_detection_frame = copy_frame(frame)
                            self._latest_detection_debug = None
                            self._detection_computed_ns = now_ns
                            self._detection_rate.record(now_ns)
                            last_key = key
            elapsed_s = (self._clock_ns() - started_ns) / 1_000_000_000.0
            if stop_event.is_set():
                break
            self._wait_for_analysis(
                stop_event,
                self._detection_wakeup,
                wake_generation,
                max(0.0, self._detection_period_s - elapsed_s),
                wake_on_change=item is None,
            )

    def _detect_board(
        self,
        detector: DetectorPort,
        frame: Frame,
        *,
        include_debug: bool,
        update_tracker: bool,
    ) -> HybridBoardResult | ClassicalBoardResult | BoardObservation | None:
        with self._detector_lock:
            parameters = inspect.signature(detector.detect).parameters
            supports_structured = all(
                name in parameters
                for name in ("source_sequence", "include_debug", "update_tracker")
            )
            if supports_structured:
                return detector.detect(
                    frame.image,
                    captured_ns=frame.captured_ns,
                    source_sequence=frame.sequence,
                    include_debug=include_debug,
                    update_tracker=update_tracker,
                )
            if include_debug or not update_tracker:
                raise RuntimeError(
                    "configured detector does not support structured debug results"
                )
            return detector.detect(frame.image, captured_ns=frame.captured_ns)  # type: ignore[call-arg]

    @staticmethod
    def _snapshot_for_result(
        result: HybridBoardResult | ClassicalBoardResult | BoardObservation | None,
        frame: Frame,
    ) -> DetectionSnapshot:
        if isinstance(result, ClassicalBoardResult):
            observation = None
            if result.detected and result.center_px is not None:
                observation = BoardObservation(
                    captured_ns=result.timestamp_ns,
                    corners_px=tuple(result.corners_px),
                    center_px=result.center_px,
                    confidence=result.confidence,
                    homography_valid=result.homography_valid,
                )
            rejection_reasons = tuple(getattr(result, "rejection_reasons", ())) or tuple(
                reason
                for candidate in result.candidates
                for reason in candidate.rejection_reasons
            )
            return DetectionSnapshot(
                enabled=True, detected=result.detected,
                source_sequence=result.source_sequence, observation=observation,
                target_valid=result.target_valid, tracking_state=result.tracking_state,
                observation_source=CameraTuningService._enum_value(result.observation_source) or "NONE",
                confidence=float(result.confidence),
                scale_px_per_mm=result.scale_px_per_mm,
                velocity_px_s=result.velocity_px_s,
                predicted_frames=int(result.predicted_frames),
                source_age_us=int(result.source_age_us),
                near_image_edge=bool(result.near_image_edge),
                partially_outside=bool(result.partially_outside),
                rejection_reasons=rejection_reasons,
                model_state="READY", model_backend="classical", model_path=None,
                combined_score=float(result.confidence),
                candidate_count=len(result.candidates),
                failure_reason=CameraTuningService._enum_value(result.failure_reason),
                inference_ms=float(result.timings_ms.get("detection", 0.0)),
                geometry_ms=float(result.timings_ms.get("geometry", 0.0)),
                total_ms=float(result.timings_ms.get("total", 0.0)),
                homography_valid=result.homography_valid,
                target_x_mm=result.target_x_mm, target_y_mm=result.target_y_mm,
                corners_px=tuple(result.corners_px), center_px=result.center_px,
                candidates=tuple(
                    CameraTuningService._classical_candidate_snapshot(candidate)
                    for candidate in result.candidates
                ),
            )
        if not isinstance(result, HybridBoardResult):
            return DetectionSnapshot(
                enabled=True, detected=result is not None,
                source_sequence=frame.sequence, observation=result,
            )
        observation = None
        corners = tuple(result.corners_px or ())
        if result.detected and corners and result.center_px is not None:
            observation = BoardObservation(
                captured_ns=result.timestamp_ns,
                corners_px=corners,
                center_px=result.center_px,
                confidence=result.combined_score,
                homography_valid=result.homography_valid,
            )
        candidates = tuple(
            CameraTuningService._candidate_snapshot(candidate)
            for candidate in result.candidates
        )
        return DetectionSnapshot(
            enabled=True,
            detected=result.detected,
            source_sequence=result.source_sequence,
            observation=observation,
            target_valid=result.target_valid,
            tracking_state=result.tracking_state,
            model_state=result.model_state,
            model_backend=result.model_backend,
            model_path=result.model_path,
            model_confidence=result.model_confidence,
            geometry_score=result.geometry_score,
            edge_support_score=result.edge_support_score,
            structure_score=result.structure_score,
            combined_score=result.combined_score,
            candidate_count=result.candidate_count,
            confirmation_count=int(getattr(result, "confirmation_count", 0)),
            miss_count=int(getattr(result, "miss_count", 0)),
            failure_reason=CameraTuningService._enum_value(result.failure_reason),
            inference_ms=result.inference_ms,
            geometry_ms=result.geometry_ms,
            total_ms=result.total_ms,
            temporal_score=result.temporal_score,
            homography_valid=result.homography_valid,
            target_x_mm=result.target_x_mm,
            target_y_mm=result.target_y_mm,
            corners_px=corners,
            center_px=result.center_px,
            candidates=candidates,
        )

    @staticmethod
    def _debug_snapshot_for_result(
        result: HybridBoardResult | ClassicalBoardResult | BoardObservation | None,
        frame: Frame,
    ) -> DetectionDebugSnapshot | None:
        if not isinstance(result, (HybridBoardResult, ClassicalBoardResult)):
            raise RuntimeError("configured detector does not support structured debug results")
        if result.source_sequence != frame.sequence:
            return None
        return DetectionDebugSnapshot(
            result.source_sequence, copy_debug_images(result.debug_images)
        )

    @staticmethod
    def _classical_candidate_snapshot(
        candidate: ClassicalCandidateEvaluation,
    ) -> DetectionCandidateSnapshot:
        xs = tuple(point[0] for point in candidate.corners_px)
        ys = tuple(point[1] for point in candidate.corners_px)
        xyxy = (min(xs), min(ys), max(xs), max(ys)) if xs and ys else (0.0, 0.0, 0.0, 0.0)
        return DetectionCandidateSnapshot(
            xyxy_px=tuple(float(value) for value in xyxy),
            accepted=candidate.accepted, model_confidence=0.0,
            geometry_score=float(candidate.geometry_score),
            edge_support_score=float(candidate.ring_score),
            structure_score=float(candidate.white_score),
            temporal_score=float(candidate.temporal_score),
            combined_score=float(candidate.combined_score),
            failure_reason=CameraTuningService._enum_value(candidate.failure_reason),
        )

    @staticmethod
    def _candidate_snapshot(candidate: CandidateEvaluation) -> DetectionCandidateSnapshot:
        return DetectionCandidateSnapshot(
            xyxy_px=tuple(float(value) for value in candidate.model.xyxy_px),
            accepted=candidate.geometry.accepted,
            model_confidence=float(candidate.model.confidence),
            geometry_score=float(candidate.geometry.geometry_score),
            edge_support_score=float(candidate.geometry.edge_support_score),
            structure_score=float(candidate.geometry.structure_score),
            temporal_score=float(candidate.temporal_score),
            combined_score=float(candidate.combined_score),
            failure_reason=CameraTuningService._enum_value(
                candidate.geometry.failure_reason
            ),
        )

    @staticmethod
    def _enum_value(value: object | None) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "value", value))

    def _reset_detector(self) -> None:
        detector = self._detector
        reset = getattr(detector, "reset", None) if detector is not None else None
        if callable(reset):
            with self._detector_lock:
                reset()

    def _wake_analysis_workers(self) -> None:
        self._notify_analysis_worker(self._diagnostics_wakeup)
        self._notify_analysis_worker(self._detection_wakeup)

    @staticmethod
    def _analysis_wake_generation(wakeup: _AnalysisWakeup) -> int:
        with wakeup.condition:
            return wakeup.generation

    @staticmethod
    def _notify_analysis_worker(wakeup: _AnalysisWakeup) -> None:
        with wakeup.condition:
            wakeup.generation += 1
            wakeup.condition.notify_all()

    @staticmethod
    def _wait_for_analysis(
        stop_event: threading.Event,
        wakeup: _AnalysisWakeup,
        observed_generation: int,
        timeout_s: float,
        *,
        wake_on_change: bool,
    ) -> None:
        with wakeup.condition:
            if wake_on_change:
                wakeup.condition.wait_for(
                    lambda: stop_event.is_set()
                    or wakeup.generation != observed_generation,
                    timeout_s,
                )
            else:
                wakeup.condition.wait_for(stop_event.is_set, timeout_s)

    def _diagnostics_key_is_current_locked(self, key: tuple[int, int]) -> bool:
        # Acquisition can advance many frames while diagnostics are computed.
        # A completed result remains valid for the active camera session even
        # when it is no longer the newest acquired frame.
        return self._session_active and key[0] == self._session_generation

    def _detection_key_is_current_locked(self, key: tuple[int, int, int]) -> bool:
        # Detection is intentionally decoupled from acquisition rate. Reject
        # results from an obsolete session/config generation, but publish a
        # completed frame even if acquisition advanced while it was processed.
        return (
            self._session_active
            and self._detection_enabled
            and not self._detection_reload_in_progress
            and key[0] == self._session_generation
            and key[1] == self._detection_generation
        )

    def _reset_derived_locked(self) -> None:
        self._latest_diagnostics = None
        self._detection_generation += 1
        self._detection_computed_ns = None
        self._latest_detection_frame = None
        self._latest_detection_debug = None
        self._latest_detection = DetectionSnapshot(
            enabled=self._detection_enabled, detected=False
        )

    def _publish_frozen_frame_locked(
        self, frozen: Frame, *, reset_sequence: bool = False
    ) -> None:
        now_ns = self._clock_ns()
        sequence = int(frozen.sequence) & 0xFFFFFFFF
        if reset_sequence:
            self._last_sequence = None
        if self._last_sequence is not None:
            delta = (sequence - self._last_sequence) & 0xFFFFFFFF
            if delta == 0 or delta >= 0x80000000:
                return
            self._sequence_gap_count += delta - 1
        self._last_sequence = sequence
        if sequence != frozen.sequence:
            frozen = Frame(
                sequence=sequence,
                captured_ns=frozen.captured_ns,
                image=frozen.image,
            )
        self._latest_frame = frozen
        self._latest_frame_received_ns = now_ns
        self._frame_count += 1
        self._acquisition_rate.record(now_ns)

    def _record_timeout(self, generation: int) -> None:
        reset_detector = False
        with self._lock:
            if generation != self._session_generation:
                return
            self._timeout_count += 1
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self._disconnect_timeout_threshold:
                reset_detector = not self._camera_fault_active
                self._camera_fault_active = True
                self._state = "Disconnected"
                self._last_error = (
                    f"camera read reached {self._consecutive_timeouts} consecutive timeouts"
                )
        if reset_detector:
            self._reset_detector()

    def _record_camera_error(self, error: BaseException, generation: int) -> None:
        reset_detector = False
        with self._lock:
            if generation != self._session_generation:
                return
            self._consecutive_timeouts = 0
            reset_detector = not self._camera_fault_active
            self._camera_fault_active = True
            self._state = "Disconnected"
            self._last_error = f"camera read failed: {error}"
        if reset_detector:
            self._reset_detector()

    def _record_camera_success_locked(self) -> None:
        self._consecutive_timeouts = 0
        if self._camera_fault_active:
            self._camera_fault_active = False
            self._state = "Connected"
            self._last_error = None

    def _record_analysis_error(
        self, worker: str, error: BaseException, key: tuple[int, int]
    ) -> None:
        with self._lock:
            if worker == "diagnostics" and self._diagnostics_key_is_current_locked(key):
                self._last_error = f"{worker} failed: {error}"

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
        if self._detection_computed_ns is not None and result.source_sequence is not None:
            age_ms = max(
                0.0, (now_ns - self._detection_computed_ns) / 1_000_000.0
            )
        return replace(result, result_age_ms=age_ms)

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _set_state(self, state: str, error: str | None) -> None:
        with self._lock:
            self._state = state
            self._last_error = error

    @staticmethod
    def _freeze_frame(frame: Frame) -> Frame:
        image = np.array(frame.image, copy=True)
        image.setflags(write=False)
        return Frame(
            sequence=int(frame.sequence) & 0xFFFFFFFF,
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
