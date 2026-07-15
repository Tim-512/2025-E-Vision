from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

import uvicorn

from ev_vision.camera.hikrobot import HikrobotCamera, create_native_api
from ev_vision.config import CameraConfig, load_config
from ev_vision.detection.board_geometry import BoardGeometryDetector
from ev_vision.tuning.models import CameraIdentity, EditableCameraParameters, ParameterBounds
from ev_vision.tuning.service import CameraTuningService
from ev_vision.tuning.storage import TuningStorage
from ev_vision.web.camera_tuning_app import create_camera_tuning_app

_FIXED_FORMAT = (1280, 1024, "BayerRG8", 2)
_DEFAULT_SERIAL = "00G02809155"
_CAMERA_MODEL = "MV-CA013-21UC"


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _port(text: str) -> int:
    value = _positive_int(text)
    if value > 65535:
        raise argparse.ArgumentTypeError("must be within 1..65535")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ev-camera-tuning",
        description=(
            "Jetson Hikrobot camera tuning dashboard. Bind 0.0.0.0 only on a trusted LAN; "
            "the 405 nm laser must remain disconnected and OFF."
        ),
        epilog=(
            "Default binding is local-only. If --host 0.0.0.0 is used, never expose or "
            "port-forward this unauthenticated development service."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--serial", default=_DEFAULT_SERIAL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/camera-tuning"))
    parser.add_argument("--preview-fps", type=_positive_float, default=20.0)
    parser.add_argument("--detection-fps", type=_positive_float, default=15.0)
    parser.add_argument("--diagnostic-fps", type=_positive_float, default=10.0)
    parser.add_argument("--timeout-ms", type=_positive_int, default=100)
    parser.add_argument(
        "--shutdown-timeout",
        type=_positive_float,
        default=2.0,
        help="seconds allowed for acquisition stop and camera close during parameter apply",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def _require_fixed_format(camera: CameraConfig) -> None:
    actual = (camera.width, camera.height, camera.pixel_format, camera.buffer_size)
    if actual != _FIXED_FORMAT:
        raise ValueError(
            "camera tuning requires fixed format 1280x1024, BayerRG8, buffer 2; "
            f"got {camera.width}x{camera.height}, {camera.pixel_format}, buffer {camera.buffer_size}"
        )


def build_application(args: argparse.Namespace):
    config = load_config(args.config)
    _require_fixed_format(config.camera)

    defaults = EditableCameraParameters.from_camera_config(config.camera)
    bounds = ParameterBounds()
    bounds.validate(defaults)

    native_api = create_native_api()

    def camera_factory(camera_config: CameraConfig) -> HikrobotCamera:
        return HikrobotCamera(native_api, camera_config, serial_number=args.serial)

    service = CameraTuningService(
        camera_factory=camera_factory,
        base_config=config.camera,
        detector=BoardGeometryDetector(),
        bounds=bounds,
        camera_identity=CameraIdentity(model=_CAMERA_MODEL, serial=args.serial),
        read_timeout_ms=args.timeout_ms,
        diagnostics_fps=args.diagnostic_fps,
        detection_fps=args.detection_fps,
        shutdown_timeout_s=args.shutdown_timeout,
    )
    storage = TuningStorage(args.output, bounds=bounds)
    return create_camera_tuning_app(
        service,
        storage,
        defaults,
        preview_fps=args.preview_fps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = build_application(args)
    except Exception as exc:
        print(f"camera tuning startup failed: {exc}", file=sys.stderr)
        return 2

    print(f"Camera tuning dashboard: http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("SAFETY: 0.0.0.0 is for a trusted LAN only; do not expose or port-forward this service.")
    else:
        print("Network: local-only binding. Use --host 0.0.0.0 only on a trusted LAN.")
    print("SAFETY: keep the 405 nm laser physically disconnected and OFF during tuning.")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
