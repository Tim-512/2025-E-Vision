from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2

from ev_vision.camera.hikrobot import HikrobotCamera, create_native_api
from ev_vision.camera.smoke import CameraSmokeStats, run_camera_smoke
from ev_vision.config import load_config


def run_with_camera(camera, *, duration_s: float, timeout_ms: int):
    latest_image = None

    def remember(frame) -> None:
        nonlocal latest_image
        latest_image = frame.image.copy()

    with camera:
        stats = run_camera_smoke(
            camera,
            duration_s=duration_s,
            timeout_ms=timeout_ms,
            on_frame=remember,
        )
    return stats, latest_image


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save sample image: {path}")


def stats_dict(stats: CameraSmokeStats) -> dict[str, object]:
    result = asdict(stats)
    result.update(
        fps=stats.fps,
        attempts=stats.attempts,
        timeout_rate=stats.timeout_rate,
        sequence_gap_rate=stats.sequence_gap_rate,
        rss_growth_bytes=stats.rss_growth_bytes,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sustained Hikrobot capture acceptance test (laser must remain OFF)",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--serial", default="00G02809155")
    parser.add_argument("--duration", type=float, default=60.0, help="capture duration in seconds")
    parser.add_argument("--timeout-ms", type=int, default=100)
    parser.add_argument("--sample", type=Path, help="optional latest BGR sample image")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    camera = HikrobotCamera(
        create_native_api(),
        cfg.camera,
        serial_number=args.serial,
    )
    stats, latest_image = run_with_camera(
        camera,
        duration_s=args.duration,
        timeout_ms=args.timeout_ms,
    )
    if args.sample is not None and latest_image is not None:
        save_image(args.sample, latest_image)
    print(json.dumps(stats_dict(stats), indent=2, ensure_ascii=False))
    return 2 if stats.disconnect is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
