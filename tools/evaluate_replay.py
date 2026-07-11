from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from ev_vision.detection.board_geometry import BoardGeometryDetector
from ev_vision.detection.laser_spot import LaserSpotDetector


def iter_images(path: Path):
    extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for image_path in sorted(item for item in path.iterdir() if item.suffix.lower() in extensions):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            yield image_path, image


def evaluate(input_dir: Path, output_path: Path) -> dict[str, float | int]:
    board_detector = BoardGeometryDetector()
    laser_detector = LaserSpotDetector()
    board_detections = 0
    laser_detections = 0
    frame_count = 0
    durations_ms: list[float] = []

    for _, image in iter_images(input_dir):
        started = time.perf_counter_ns()
        board = board_detector.detect(image, captured_ns=frame_count)
        if frame_count == 0:
            laser_detector.update_background(image)
            laser = None
        else:
            laser = laser_detector.detect(image, captured_ns=frame_count)
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        frame_count += 1
        board_detections += board is not None
        laser_detections += laser is not None

    metrics: dict[str, float | int] = {
        "frames": frame_count,
        "board_detections": board_detections,
        "laser_detections": laser_detections,
        "mean_processing_ms": float(np.mean(durations_ms)) if durations_ms else 0.0,
        "p95_processing_ms": float(np.percentile(durations_ms, 95)) if durations_ms else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate classical board/laser detection on an image replay directory")
    parser.add_argument("input", type=Path, help="directory containing ordered replay images")
    parser.add_argument("--output", type=Path, default=Path("replay-metrics.json"))
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")
    metrics = evaluate(args.input, args.output)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
