from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ev_vision.gimbal_usb.protocol import GimbalTargetCommand
from ev_vision.gimbal_usb.safety import GateDecision
from ev_vision.gimbal_usb.serial_transport import SerialStats


@dataclass(frozen=True)
class GimbalWorkerSnapshot:
    """Bounded diagnostics for the latest-only gimbal output worker."""

    running: bool
    ticks: int
    overruns: int
    sent_valid: int
    sent_invalid: int
    transport_deferred: int
    real_observations: int
    predicted_observations: int
    reasons: Mapping[str, int]
    last_decision: GateDecision | None
    last_error: str | None
    serial: SerialStats

    @property
    def cycles(self) -> int:
        """Compatibility alias for the implementation-plan terminology."""

        return self.ticks


class GimbalOutputWorker:
    """Send the newest detection to the gimbal at a fixed deadline cadence.

    Production serial ownership belongs exclusively to the daemon thread created
    by :meth:`start`: that thread calls ``transmit`` and closes the transport in
    its ``finally`` block. ``run_tick`` and ``run_for_test_ticks`` are synchronous
    deterministic test hooks and must not be called while the worker is running.
    """

    def __init__(
        self,
        service,
        gate,
        transport,
        *,
        output_hz: float = 50.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wait: Callable[[float], bool] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        if (
            isinstance(output_hz, bool)
            or not isinstance(output_hz, (int, float))
            or not math.isfinite(float(output_hz))
            or float(output_hz) <= 0.0
        ):
            raise ValueError("output_hz must be positive and finite")
        if not callable(clock_ns):
            raise ValueError("clock_ns must be callable")
        if wait is not None and not callable(wait):
            raise ValueError("wait must be callable")

        self.service = service
        self.gate = gate
        self.transport = transport
        self._period_ns = round(1_000_000_000 / float(output_hz))
        self._clock_ns = clock_ns
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        self._wait = wait if wait is not None else self._stop_event.wait

        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._ticks = 0
        self._overruns = 0
        self._sent_valid = 0
        self._sent_invalid = 0
        self._transport_deferred = 0
        self._real_observations = 0
        self._predicted_observations = 0
        self._reasons: Counter[str] = Counter()
        self._last_decision: GateDecision | None = None
        self._last_error: str | None = None

    @property
    def thread(self) -> threading.Thread | None:
        with self._lifecycle_lock:
            return self._thread

    @staticmethod
    def _error_text(operation: str, error: BaseException) -> str:
        detail = str(error).strip()
        name = type(error).__name__
        return f"{operation}: {name}: {detail}" if detail else f"{operation}: {name}"

    @staticmethod
    def _safe_decision(reason: str) -> GateDecision:
        return GateDecision(
            command=GimbalTargetCommand.safe(),
            reason=reason,
            observation_source="NONE",
            result_age_ms=None,
            source_age_us=0,
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name="gimbal-usb-output",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def _record_overruns(self, count: int) -> None:
        if count <= 0:
            return
        with self._state_lock:
            self._overruns += count

    def _advance_deadline(self, deadline_ns: int) -> tuple[int, float]:
        deadline_ns += self._period_ns
        now_ns = int(self._clock_ns())
        if now_ns > deadline_ns:
            expired = (now_ns - deadline_ns) // self._period_ns + 1
            self._record_overruns(expired)
            deadline_ns += expired * self._period_ns
        delay_s = max(0.0, (deadline_ns - int(self._clock_ns())) / 1_000_000_000)
        return deadline_ns, delay_s

    def _run(self) -> None:
        try:
            deadline_ns = int(self._clock_ns())
            while not self._stop_event.is_set():
                self.run_tick()
                deadline_ns, delay_s = self._advance_deadline(deadline_ns)
                if delay_s > 0.0 and self._wait(delay_s):
                    break
        except Exception as exc:
            # Keep cleanup fail-closed even if an injected clock/wait dependency
            # unexpectedly fails outside the per-tick exception boundaries.
            with self._state_lock:
                self._last_error = self._error_text("worker loop failed", exc)
        finally:
            try:
                self.transport.close(safe_frames=5)
            except Exception as exc:
                with self._state_lock:
                    self._last_error = self._error_text("transport close failed", exc)
            finally:
                with self._lifecycle_lock:
                    self._closed = True

    def run_tick(self) -> None:
        """Run one latest-only output tick without propagating runtime faults."""

        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            raise RuntimeError("run_tick cannot run concurrently with the worker thread")

        error_text: str | None = None
        try:
            detection = self.service.latest_detection()
        except Exception as exc:
            decision = self._safe_decision("detection_provider_error")
            error_text = self._error_text("latest_detection failed", exc)
        else:
            try:
                decision = self.gate.evaluate(detection)
            except Exception as exc:
                decision = self._safe_decision("safety_gate_error")
                error_text = self._error_text("safety gate failed", exc)

        try:
            sent = bool(self.transport.transmit(decision.command))
        except Exception as exc:
            sent = False
            error_text = self._error_text("serial transmit failed", exc)

        with self._state_lock:
            self._ticks += 1
            self._last_decision = decision
            self._reasons[decision.reason] += 1
            self._real_observations += int(decision.reason == "real_observation")
            self._predicted_observations += int(
                decision.reason == "predicted_observation"
            )
            if sent:
                if decision.command.tracking:
                    self._sent_valid += 1
                else:
                    self._sent_invalid += 1
            else:
                self._transport_deferred += 1
            if error_text is not None:
                self._last_error = error_text

    def run_for_test_ticks(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        deadline_ns = int(self._clock_ns())
        for index in range(count):
            self.run_tick()
            if index + 1 == count:
                break
            deadline_ns, delay_s = self._advance_deadline(deadline_ns)
            if delay_s > 0.0:
                self._wait(delay_s)

    def stop(self) -> None:
        """Stop, join, and let the owner thread emit five shutdown safe frames."""

        self._stop_event.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

    def snapshot(self) -> GimbalWorkerSnapshot:
        with self._lifecycle_lock:
            thread = self._thread
            running = thread is not None and thread.is_alive()
        with self._state_lock:
            ticks = self._ticks
            overruns = self._overruns
            sent_valid = self._sent_valid
            sent_invalid = self._sent_invalid
            transport_deferred = self._transport_deferred
            real_observations = self._real_observations
            predicted_observations = self._predicted_observations
            reasons = dict(self._reasons)
            last_decision = self._last_decision
            last_error = self._last_error
        return GimbalWorkerSnapshot(
            running=running,
            ticks=ticks,
            overruns=overruns,
            sent_valid=sent_valid,
            sent_invalid=sent_invalid,
            transport_deferred=transport_deferred,
            real_observations=real_observations,
            predicted_observations=predicted_observations,
            reasons=reasons,
            last_decision=last_decision,
            last_error=last_error,
            serial=self.transport.snapshot(),
        )
