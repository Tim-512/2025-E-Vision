from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2

from ev_vision.calibration_capture.local import (
    PoseSignature,
    analyze_chessboard_frame,
    poses_are_similar,
    remove_last_saved,
    render_capture_overlay,
    save_original_frame,
)
from ev_vision.camera.hikrobot import HikrobotCamera, create_native_api
from ev_vision.config import AppConfig, load_config
from ev_vision.tuning.runtime import require_fixed_format


WINDOW_NAME = "EV Vision - camera calibration capture"
DEFAULT_SERIAL = "00G02809155"


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _corner_count(text: str) -> int:
    value = _positive_int(text)
    if value <= 1:
        raise argparse.ArgumentTypeError("must be at least 2")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ev-camera-calibration-capture",
        description="Capture varied Jetson-local chessboard images for camera calibration.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/jetson-local.yaml"))
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--columns", type=_corner_count, default=8)
    parser.add_argument("--rows", type=_corner_count, default=5)
    parser.add_argument("--square-mm", type=_positive_float, default=22.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/images"))
    parser.add_argument("--width", type=_positive_int, default=960)
    parser.add_argument("--timeout-ms", type=_positive_int, default=100)
    return parser


def build_camera(config_path: Path, serial: str) -> tuple[HikrobotCamera, AppConfig]:
    config = load_config(config_path)
    require_fixed_format(config.camera)
    camera = HikrobotCamera(create_native_api(), config.camera, serial_number=serial)
    return camera, config


def run_capture_session(
    camera: HikrobotCamera,
    *,
    pattern_size: tuple[int, int],
    square_mm: float,
    output: Path,
    preview_width: int,
    timeout_ms: int,
    cv: Any = cv2,
    session_signatures: list[PoseSignature] | None = None,
) -> str:
    del square_mm
    saved_paths: list[Path] = []
    signatures = session_signatures if session_signatures is not None else []

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    try:
        camera.open()
        while True:
            try:
                frame = camera.read(timeout_ms=timeout_ms)
            except TimeoutError:
                continue

            analysis = analyze_chessboard_frame(frame.image, pattern_size=pattern_size)
            duplicate = bool(
                analysis.pose_signature is not None
                and any(
                    poses_are_similar(analysis.pose_signature, previous)
                    for previous in signatures
                )
            )
            rendered = render_capture_overlay(
                frame.image,
                analysis,
                len(saved_paths),
                duplicate,
                pattern_size=pattern_size,
            )
            if rendered.shape[1] > preview_width:
                preview_height = max(
                    1,
                    round(rendered.shape[0] * preview_width / rendered.shape[1]),
                )
                rendered = cv.resize(
                    rendered,
                    (preview_width, preview_height),
                    interpolation=cv.INTER_AREA,
                )
            cv.imshow(WINDOW_NAME, rendered)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return "key"
            if cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
                return "window"
            if key == ord(" ") and analysis.save_allowed:
                path = save_original_frame(output, frame.image, index=len(saved_paths) + 1)
                saved_paths.append(path)
                if analysis.pose_signature is not None:
                    signatures.append(analysis.pose_signature)
            elif key in (ord("r"), ord("R")) and saved_paths:
                remove_last_saved(saved_paths)
                if signatures:
                    signatures.pop()
    finally:
        try:
            camera.close()
        finally:
            cv.destroyWindow(WINDOW_NAME)


def _print_startup_guidance(config: AppConfig) -> None:
    camera = config.camera
    print("Camera parameters loaded from YAML:")
    print(f"  exposure_us={camera.exposure_us}")
    print(f"  gain_db={camera.gain_db}")
    print(f"  acquisition_fps={camera.acquisition_fps}")
    print("Capture 20-25 images and keep at least 15 usable varied poses.")
    print("Include center, corners, near, far, rotated, and tilted board poses.")
    print("Controls: SPACE saves, R removes the latest image from this session, Q/Esc quits.")
    print(
        "SAFETY: physically disconnect or reliably cover the hardware-always-on "
        "405 nm laser before calibration."
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print(
            "calibration capture startup failed: DISPLAY is not set; "
            "run from the Jetson desktop session",
            file=sys.stderr,
        )
        return 2

    try:
        camera, config = build_camera(args.config, args.serial)
    except Exception as exc:
        print(f"calibration capture startup failed: {exc}", file=sys.stderr)
        return 2

    _print_startup_guidance(config)
    try:
        run_capture_session(
            camera,
            pattern_size=(args.columns, args.rows),
            square_mm=args.square_mm,
            output=args.output,
            preview_width=args.width,
            timeout_ms=args.timeout_ms,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"calibration capture failed: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
