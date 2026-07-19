from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import threading
from types import FrameType
from typing import Callable, Sequence

from ev_vision.gimbal_usb.config import GimbalUsbConfig, load_gimbal_usb_config
from ev_vision.gimbal_usb.safety import GateLimits, GimbalControlGate
from ev_vision.gimbal_usb.serial_transport import GimbalSerialTransport
from ev_vision.gimbal_usb.target_angles import (
    AngleConverter,
    TargetAngleConverter,
    UnavailableTargetAngleConverter,
)
from ev_vision.gimbal_usb.worker import GimbalOutputWorker
from ev_vision.preview.local import run_local_preview
from ev_vision.tuning.runtime import CameraRuntime, DEFAULT_SERIAL, build_camera_runtime


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ev-gimbal-vision",
        description=(
            "Jetson-local target detection and fail-closed USB gimbal output "
            "without FastAPI or Uvicorn."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/jetson-local.yaml"),
        help="camera and detection YAML",
    )
    parser.add_argument(
        "--gimbal-config",
        type=Path,
        default=Path("config/gimbal_usb.yaml"),
        help="USB gimbal and calibration YAML",
    )
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument(
        "--display",
        action="store_true",
        help="show the existing low-latency local OpenCV preview",
    )
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--display-fps", type=_positive_float, default=45.0)
    parser.add_argument("--detection-fps", type=_positive_float, default=50.0)
    parser.add_argument("--timeout-ms", type=_positive_int, default=100)
    parser.add_argument("--shutdown-timeout", type=_positive_float, default=2.0)
    return parser


def build_angle_converter(
    config: GimbalUsbConfig,
    *,
    image_size: tuple[int, int],
) -> AngleConverter:
    """Load calibration without allowing calibration failure to stop vision."""
    try:
        return TargetAngleConverter.from_file(
            config.calibration_path,
            image_size=image_size,
            max_rms_px=config.max_calibration_rms_px,
            yaw_sign=config.yaw_sign,
            pitch_sign=config.pitch_sign,
        )
    except Exception as exc:
        return UnavailableTargetAngleConverter(str(exc))


def build_worker(
    camera_runtime: CameraRuntime,
    config: GimbalUsbConfig,
    *,
    converter: AngleConverter | None = None,
) -> GimbalOutputWorker:
    camera = camera_runtime.config.camera
    image_size = (camera.width, camera.height)
    if converter is None:
        converter = build_angle_converter(config, image_size=image_size)
    gate = GimbalControlGate(
        converter,
        GateLimits(
            max_result_age_ms=config.max_result_age_ms,
            predicted_control_max_frames=config.predicted_control_max_frames,
            predicted_control_max_age_us=round(
                config.predicted_control_max_age_ms * 1000.0
            ),
            predicted_max_angle_step_deg=config.predicted_max_angle_step_deg,
            image_size=image_size,
        ),
    )
    transport = GimbalSerialTransport(
        config.port,
        baudrate=config.baudrate,
        reconnect_interval_s=config.reconnect_interval_s,
    )
    return GimbalOutputWorker(
        camera_runtime.service,
        gate,
        transport,
        output_hz=config.output_hz,
    )


def _new_stop_event() -> threading.Event:
    return threading.Event()


def _install_stop_handlers(
    stop_event: threading.Event,
) -> Callable[[], None]:
    """Make SIGINT/SIGTERM request an orderly worker-first shutdown."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    previous: dict[signal.Signals, object] = {}

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        stop_event.set()

    for candidate in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[candidate] = signal.getsignal(candidate)
            signal.signal(candidate, request_stop)
        except (AttributeError, OSError, RuntimeError, ValueError):
            continue

    def restore() -> None:
        for candidate, handler in previous.items():
            try:
                signal.signal(candidate, handler)
            except (OSError, RuntimeError, ValueError):
                pass

    return restore


def _print_camera_parameters(camera: object) -> None:
    print("Camera parameters loaded from YAML:")
    print(
        f"  format={camera.width}x{camera.height} "
        f"{camera.pixel_format}, buffer={camera.buffer_size}"
    )
    print(f"  exposure_us={camera.exposure_us}")
    print(f"  gain_db={camera.gain_db}")
    print(f"  acquisition_fps={camera.acquisition_fps}")
    print(f"  auto_exposure={camera.auto_exposure}")
    print(f"  auto_gain={camera.auto_gain}")
    print(f"  auto_white_balance={camera.auto_white_balance}")


def _print_calibration(
    converter: AngleConverter,
    config: GimbalUsbConfig,
) -> None:
    if isinstance(converter, TargetAngleConverter):
        calibration = converter.calibration
        matrix = calibration.camera_matrix
        print("Camera calibration:")
        print(f"  path={config.calibration_path}")
        print(f"  image_size={calibration.image_size[0]}x{calibration.image_size[1]}")
        print(f"  rms_px={calibration.rms_px:.4f} (maximum {config.max_calibration_rms_px:.4f})")
        print(
            "  "
            f"fx={float(matrix[0, 0]):.4f}, fy={float(matrix[1, 1]):.4f}, "
            f"cx={float(matrix[0, 2]):.4f}, cy={float(matrix[1, 2]):.4f}"
        )
        return

    reason = getattr(converter, "reason", "calibration unavailable")
    print("=" * 72, file=sys.stderr)
    print(f"CALIBRATION ERROR: {reason}", file=sys.stderr)
    print(
        "USB CONTROL IS FAIL-CLOSED: camera, detection, and preview continue, "
        "but every USB command remains all-zero with tracking=0.",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)


def print_startup_summary(
    camera_runtime: CameraRuntime,
    config: GimbalUsbConfig,
    converter: AngleConverter,
) -> None:
    _print_camera_parameters(camera_runtime.config.camera)
    _print_calibration(converter, config)
    print("USB gimbal output:")
    print(f"  stable_port={config.port}")
    print(f"  serial={config.baudrate} baud, 8N1, output={config.output_hz:g} Hz")
    print(f"  reconnect_interval={config.reconnect_interval_s:g} s")
    print(
        "  valid-result maximum age="
        f"{config.max_result_age_ms:.1f} ms"
    )
    print(
        "  pure prediction maximum="
        f"{config.predicted_control_max_frames} frames / "
        f"{config.predicted_control_max_age_ms:.1f} ms"
    )
    print(
        "  prediction angular step maximum="
        f"{config.predicted_max_angle_step_deg:g} deg per axis"
    )
    print(f"  yaw_sign={config.yaw_sign}, pitch_sign={config.pitch_sign}")
    print("  distance=0, fire=0, invalid output is all-zero tracking=0")
    print(
        "SAFETY: the 405 nm laser is hardware-always-on; software fire=0 cannot "
        "turn it off. Physically disconnect it or reliably cover it during debugging."
    )


def format_runtime_snapshot(camera_runtime: CameraRuntime, worker: GimbalOutputWorker) -> str:
    camera = camera_runtime.service.runtime_snapshot()
    output = worker.snapshot()
    serial_state = "up" if output.serial.connected else "down"
    error = output.last_error or output.serial.last_error or "none"
    return (
        f"camera={camera.state} acquisition={camera.acquisition_fps:.1f}fps "
        f"detection={camera.detection_fps:.1f}fps | usb={serial_state} "
        f"ticks={output.ticks} valid={output.sent_valid} safe={output.sent_invalid} "
        f"overruns={output.overruns} error={error}"
    )


def run_application(
    camera_runtime: CameraRuntime,
    worker: GimbalOutputWorker,
    *,
    display: bool,
    width: int = 640,
    display_fps: float = 45.0,
    stop_event: threading.Event | None = None,
) -> int:
    stop_event = stop_event if stop_event is not None else _new_stop_event()
    try:
        camera_runtime.service.start()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"gimbal vision camera startup failed: {exc}", file=sys.stderr)
        return 2

    worker_started = False
    try:
        try:
            worker.start()
            worker_started = True
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(f"gimbal worker startup failed: {exc}", file=sys.stderr)
            return 3

        if display:
            run_local_preview(
                camera_runtime.service,
                max_width=width,
                display_fps=display_fps,
            )
        else:
            while not stop_event.wait(1.0):
                print(format_runtime_snapshot(camera_runtime, worker), flush=True)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"gimbal vision runtime failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if worker_started:
            worker.stop()
        camera_runtime.service.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.display
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
    ):
        print(
            "gimbal vision startup failed: DISPLAY is not set; use headless mode "
            "or run --display from the Jetson desktop session",
            file=sys.stderr,
        )
        return 2

    try:
        gimbal_config = load_gimbal_usb_config(args.gimbal_config)
        camera_runtime = build_camera_runtime(
            config_path=args.config,
            serial=args.serial,
            read_timeout_ms=args.timeout_ms,
            detection_fps=args.detection_fps,
            diagnostics_fps=None,
            shutdown_timeout_s=args.shutdown_timeout,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"gimbal vision startup failed: {exc}", file=sys.stderr)
        return 2

    image_size = (
        camera_runtime.config.camera.width,
        camera_runtime.config.camera.height,
    )
    converter = build_angle_converter(gimbal_config, image_size=image_size)
    try:
        worker = build_worker(camera_runtime, gimbal_config, converter=converter)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"gimbal vision startup failed: {exc}", file=sys.stderr)
        return 2

    print_startup_summary(camera_runtime, gimbal_config, converter)
    stop_event = _new_stop_event()
    restore_handlers = (
        (lambda: None)
        if args.display
        else _install_stop_handlers(stop_event)
    )
    try:
        return run_application(
            camera_runtime,
            worker,
            display=args.display,
            width=args.width,
            display_fps=args.display_fps,
            stop_event=stop_event,
        )
    finally:
        restore_handlers()


if __name__ == "__main__":
    raise SystemExit(main())
