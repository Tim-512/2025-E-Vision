from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Sequence

from ev_vision.preview.local import run_local_preview
from ev_vision.tuning.runtime import DEFAULT_SERIAL, build_camera_runtime


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ev-camera-preview",
        description="Low-memory Jetson-local camera detection preview without FastAPI or Uvicorn.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--display-fps", type=_positive_float, default=30.0)
    parser.add_argument("--detection-fps", type=_positive_float, default=30.0)
    parser.add_argument("--timeout-ms", type=_positive_int, default=100)
    parser.add_argument("--shutdown-timeout", type=_positive_float, default=2.0)
    return parser


def _print_parameters(camera) -> None:
    print("Local preview camera parameters loaded from YAML:")
    print(f"  exposure_us={camera.exposure_us}")
    print(f"  gain_db={camera.gain_db}")
    print(f"  acquisition_fps={camera.acquisition_fps}")
    print(f"  auto_exposure={camera.auto_exposure}")
    print(f"  auto_gain={camera.auto_gain}")
    print(f"  auto_white_balance={camera.auto_white_balance}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print(
            "local preview startup failed: DISPLAY is not set; run from the Jetson desktop session",
            file=sys.stderr,
        )
        return 2
    try:
        runtime = build_camera_runtime(
            config_path=args.config,
            serial=args.serial,
            read_timeout_ms=args.timeout_ms,
            detection_fps=args.detection_fps,
            diagnostics_fps=None,
            shutdown_timeout_s=args.shutdown_timeout,
        )
    except Exception as exc:
        print(f"local preview startup failed: {exc}", file=sys.stderr)
        return 2

    _print_parameters(runtime.config.camera)
    print("Controls: Q or Esc closes the local preview.")
    print("SAFETY: the 405 nm laser is hardware-always-on; physically disconnect or reliably cover it during debugging.")
    try:
        runtime.service.start()
        run_local_preview(
            runtime.service,
            max_width=args.width,
            display_fps=args.display_fps,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"local preview failed: {exc}", file=sys.stderr)
        return 3
    finally:
        runtime.service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())