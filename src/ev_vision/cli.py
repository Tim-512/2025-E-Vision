from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ev_vision.config import ConfigError, load_config
from ev_vision.control import VisualServo
from ev_vision.models import BoardObservation, GimbalFeedback, LaserMode
from ev_vision.protocol import MessageType, crc16_ccitt_false, decode_frame, encode_frame
from ev_vision.runtime import CycleInput, VisionRuntime
from ev_vision.state_machine import Event, VisionStateMachine


def _machine_in_track() -> VisionStateMachine:
    machine = VisionStateMachine()
    for event in (Event.BOOT_COMPLETE, Event.START, Event.BOARD_FOUND, Event.BOARD_STABLE, Event.LASER_CALIBRATED, Event.CENTERED, Event.TRACK_REQUESTED):
        machine.handle(event)
    return machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev-vision")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--hardware-required", action="store_true")
    commands.add_parser("protocol-selftest")
    mock = commands.add_parser("mock-run")
    mock.add_argument("--cycles", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        try:
            load_config(args.config, hardware_required=args.hardware_required)
        except ConfigError as exc:
            print(f"invalid configuration: {exc}")
            return 2
        print("configuration valid")
        return 0
    if args.command == "protocol-selftest":
        frame = encode_frame(MessageType.HEARTBEAT, 1, b"selftest")
        decoded = decode_frame(frame)
        ok = crc16_ccitt_false(b"123456789") == 0x29B1 and decoded.payload == b"selftest"
        print("protocol self-test passed" if ok else "protocol self-test failed")
        return 0 if ok else 1
    if args.command == "mock-run":
        if args.cycles <= 0:
            print("cycles must be positive")
            return 2
        runtime = VisionRuntime(_machine_in_track(), VisualServo.simple(kp=1.0, max_rate=20.0, max_accel=1000.0))
        last = None
        for index in range(args.cycles):
            now_ns = index * 10_000_000
            board = BoardObservation(now_ns, ((0, 0), (1, 0), (1, 1), (0, 1)), (640, 512), 1.0, True)
            last = runtime.step(CycleInput(now_ns, board, (0.1, -0.1), GimbalFeedback(received_ns=now_ns)))
        runtime.state_machine.handle(Event.STOP)
        shutdown = runtime.step(CycleInput(args.cycles * 10_000_000, None, None, GimbalFeedback()))
        assert last is not None
        print(f"cycles={args.cycles} final_rates=({shutdown.command.yaw_rate_deg_s:.1f},{shutdown.command.pitch_rate_deg_s:.1f}) laser={shutdown.command.laser_mode.name}")
        return 0 if shutdown.command.laser_mode is LaserMode.OFF else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
