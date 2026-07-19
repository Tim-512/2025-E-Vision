from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pytest

from ev_vision.gimbal_usb.protocol import GimbalTargetCommand
from ev_vision.gimbal_usb.safety import GateDecision
from ev_vision.gimbal_usb.serial_transport import SerialStats
from ev_vision.tuning.models import DetectionSnapshot


@dataclass
class FakeService:
    detections: list[DetectionSnapshot | BaseException]

    def __post_init__(self) -> None:
        self.latest_detection_calls = 0

    def latest_detection(self) -> DetectionSnapshot:
        self.latest_detection_calls += 1
        item = self.detections[min(self.latest_detection_calls - 1, len(self.detections) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class FakeGate:
    def __init__(self, decisions: list[GateDecision | BaseException]) -> None:
        self._decisions = decisions
        self.calls: list[DetectionSnapshot] = []

    def evaluate(self, detection: DetectionSnapshot) -> GateDecision:
        self.calls.append(detection)
        item = self._decisions[min(len(self.calls) - 1, len(self._decisions) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class FakeTransport:
    def __init__(self, outcomes: list[bool | BaseException]) -> None:
        self._outcomes = outcomes
        self.commands: list[GimbalTargetCommand] = []
        self.transmit_thread_ids: list[int] = []
        self.close_calls: list[int] = []
        self.close_thread_ids: list[int] = []
        self.transmitted = threading.Event()

    def transmit(self, command: GimbalTargetCommand) -> bool:
        self.commands.append(command)
        self.transmit_thread_ids.append(threading.get_ident())
        self.transmitted.set()
        item = self._outcomes[min(len(self.commands) - 1, len(self._outcomes) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, *, safe_frames: int = 5) -> None:
        self.close_calls.append(safe_frames)
        self.close_thread_ids.append(threading.get_ident())

    def snapshot(self) -> SerialStats:
        return SerialStats(
            connected=not self.close_calls,
            sent_frames=sum(isinstance(item, bool) and item for item in self._outcomes),
            connect_count=1,
            reconnects=0,
            open_errors=0,
            write_errors=0,
            sequence=0,
            last_error=None,
            closed=bool(self.close_calls),
        )


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance_s(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


class AdvancingWait:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.requested_delays: list[float] = []

    def __call__(self, delay_s: float) -> bool:
        self.requested_delays.append(delay_s)
        self.clock.advance_s(delay_s)
        return False


def detection() -> DetectionSnapshot:
    return DetectionSnapshot(enabled=True, detected=True)


def valid_decision() -> GateDecision:
    return GateDecision(
        command=GimbalTargetCommand(1.25, -2.5, True),
        reason="real_observation",
        observation_source="FULL_BOARD",
        result_age_ms=5.0,
        source_age_us=0,
    )


def invalid_decision() -> GateDecision:
    return GateDecision(
        command=GimbalTargetCommand.safe(),
        reason="target_invalid",
        observation_source="NONE",
        result_age_ms=None,
        source_age_us=0,
    )


def worker_type():
    from ev_vision.gimbal_usb.worker import GimbalOutputWorker

    return GimbalOutputWorker


def test_cycle_reads_latest_exactly_once_and_transmits_gate_decision() -> None:
    service = FakeService([detection()])
    gate = FakeGate([valid_decision()])
    transport = FakeTransport([True])
    worker = worker_type()(service, gate, transport, output_hz=50.0)

    worker.run_tick()

    assert service.latest_detection_calls == 1
    assert gate.calls == [service.detections[0]]
    assert transport.commands == [valid_decision().command]
    status = worker.snapshot()
    assert status.ticks == 1
    assert status.sent_valid == 1
    assert status.sent_invalid == 0
    assert status.transport_deferred == 0
    assert status.real_observations == 1
    assert status.reasons == {"real_observation": 1}


def test_worker_is_latest_only_without_a_detection_queue() -> None:
    newest = detection()
    service = FakeService([newest, newest, newest])
    gate = FakeGate([valid_decision()])
    transport = FakeTransport([True])
    worker = worker_type()(service, gate, transport)

    worker.run_for_test_ticks(3)

    assert service.latest_detection_calls == 3
    assert len(gate.calls) == 3
    assert len(transport.commands) == 3
    assert not hasattr(worker, "queue")
    assert not hasattr(worker, "_queue")


def test_transport_deferred_is_not_counted_as_sent() -> None:
    worker = worker_type()(
        FakeService([detection()]),
        FakeGate([valid_decision()]),
        FakeTransport([False]),
    )

    worker.run_tick()

    status = worker.snapshot()
    assert status.sent_valid == 0
    assert status.sent_invalid == 0
    assert status.transport_deferred == 1


def test_invalid_command_is_counted_only_after_successful_transport() -> None:
    worker = worker_type()(
        FakeService([detection(), detection()]),
        FakeGate([invalid_decision()]),
        FakeTransport([False, True]),
    )

    worker.run_tick()
    worker.run_tick()

    status = worker.snapshot()
    assert status.sent_invalid == 1
    assert status.transport_deferred == 1


def test_provider_exception_sends_safe_command_and_continues() -> None:
    service = FakeService([RuntimeError("camera snapshot failed"), detection()])
    gate = FakeGate([valid_decision()])
    transport = FakeTransport([True, True])
    worker = worker_type()(service, gate, transport)

    worker.run_tick()
    worker.run_tick()

    assert transport.commands == [GimbalTargetCommand.safe(), valid_decision().command]
    status = worker.snapshot()
    assert status.ticks == 2
    assert status.sent_invalid == 1
    assert status.sent_valid == 1
    assert status.reasons["detection_provider_error"] == 1
    assert "camera snapshot failed" in (status.last_error or "")


def test_gate_exception_sends_safe_command_and_continues() -> None:
    gate = FakeGate([ValueError("gate failed"), valid_decision()])
    transport = FakeTransport([True, True])
    worker = worker_type()(FakeService([detection(), detection()]), gate, transport)

    worker.run_tick()
    worker.run_tick()

    assert transport.commands == [GimbalTargetCommand.safe(), valid_decision().command]
    status = worker.snapshot()
    assert status.reasons["safety_gate_error"] == 1
    assert status.sent_invalid == 1
    assert status.sent_valid == 1


def test_transport_exception_does_not_stop_future_ticks() -> None:
    transport = FakeTransport([OSError("usb unplugged"), True])
    worker = worker_type()(
        FakeService([detection(), detection()]), FakeGate([valid_decision()]), transport
    )

    worker.run_tick()
    worker.run_tick()

    status = worker.snapshot()
    assert status.ticks == 2
    assert status.transport_deferred == 1
    assert status.sent_valid == 1
    assert "usb unplugged" in (status.last_error or "")


def test_scheduler_uses_50_hz_accumulated_deadlines() -> None:
    clock = FakeClock()
    wait = AdvancingWait(clock)
    worker = worker_type()(
        FakeService([detection()] * 4),
        FakeGate([valid_decision()]),
        FakeTransport([True]),
        output_hz=50.0,
        clock_ns=clock,
        wait=wait,
    )

    worker.run_for_test_ticks(4)

    assert wait.requested_delays == pytest.approx([0.02, 0.02, 0.02])
    assert worker.snapshot().ticks == 4


def test_scheduler_skips_expired_periods_without_catchup_burst() -> None:
    clock = FakeClock()
    wait = AdvancingWait(clock)

    class SlowService(FakeService):
        def latest_detection(self) -> DetectionSnapshot:
            result = super().latest_detection()
            if self.latest_detection_calls == 2:
                clock.advance_s(0.055)
            return result

    worker = worker_type()(
        SlowService([detection()] * 4),
        FakeGate([valid_decision()]),
        FakeTransport([True]),
        output_hz=50.0,
        clock_ns=clock,
        wait=wait,
    )

    worker.run_for_test_ticks(4)

    assert wait.requested_delays == pytest.approx([0.02, 0.005, 0.02], abs=1e-9)
    assert all(delay > 0.0 for delay in wait.requested_delays)
    assert worker.snapshot().overruns == 2


@pytest.mark.parametrize("output_hz", [0, -1, float("nan"), float("inf"), True])
def test_output_hz_must_be_positive_finite_number(output_hz: Any) -> None:
    with pytest.raises(ValueError, match="output_hz"):
        worker_type()(
            FakeService([detection()]),
            FakeGate([valid_decision()]),
            FakeTransport([True]),
            output_hz=output_hz,
        )


def test_start_is_idempotent_and_creates_one_daemon_thread() -> None:
    transport = FakeTransport([True])
    worker = worker_type()(
        FakeService([detection()]), FakeGate([valid_decision()]), transport
    )

    worker.start()
    first_thread = worker.thread
    worker.start()

    assert first_thread is worker.thread
    assert first_thread is not None
    assert first_thread.daemon is True
    assert first_thread.name == "gimbal-usb-output"
    assert transport.transmitted.wait(1.0)
    worker.stop()


def test_stop_joins_and_worker_thread_closes_transport_with_five_safe_frames() -> None:
    transport = FakeTransport([True])
    worker = worker_type()(
        FakeService([detection()]), FakeGate([valid_decision()]), transport
    )
    main_thread_id = threading.get_ident()

    worker.start()
    assert transport.transmitted.wait(1.0)
    worker.stop()

    assert worker.snapshot().running is False
    assert worker.thread is None
    assert transport.close_calls == [5]
    assert transport.close_thread_ids == transport.transmit_thread_ids[:1]
    assert transport.close_thread_ids[0] != main_thread_id


def test_stop_is_idempotent_and_does_not_restart_closed_worker() -> None:
    transport = FakeTransport([True])
    worker = worker_type()(
        FakeService([detection()]), FakeGate([valid_decision()]), transport
    )

    worker.start()
    assert transport.transmitted.wait(1.0)
    worker.stop()
    worker.stop()
    worker.start()

    assert transport.close_calls == [5]
    assert worker.thread is None
    assert worker.snapshot().running is False


def test_custom_stop_event_is_used() -> None:
    stop_event = threading.Event()
    worker = worker_type()(
        FakeService([detection()]),
        FakeGate([valid_decision()]),
        FakeTransport([True]),
        stop_event=stop_event,
    )

    worker.start()
    assert worker.transport.transmitted.wait(1.0)
    worker.stop()

    assert stop_event.is_set()
